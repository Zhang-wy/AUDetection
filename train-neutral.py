#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
# from numpy.core.numeric import load, require
# from numpy.lib.type_check import imag
from torchsummary import summary
import yaml
import numpy as np
import os

# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad, gradcheck

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

from tools.utils import funcs, loss_neutral

from tensorboardX import SummaryWriter

import torchvision.models as models
from net.baseline import Model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def smooth_one_hot(true_label, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    num_class = true_label.size(-1)
   
    with torch.no_grad():
        true_dist = (1.0 - smoothing) * true_label.data.clone()
        true_dist += smoothing / (num_class - 1)
    return true_dist


class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition (Adding Neutral Face)
    """
    def init_environment(self):
        
        super().init_environment()
        self.best_f1 = np.zeros(self.arg.model_args['num_class'])
        self.best_acc = np.zeros(self.arg.model_args['num_class'])
        self.best_aver_f1 = 0
        self.best_aver_acc = 0

        torch.manual_seed(self.arg.seed)
        torch.cuda.manual_seed_all(self.arg.seed)
        torch.cuda.manual_seed(self.arg.seed)
        np.random.seed(self.arg.seed)

    def load_data(self):

        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
            self.arg.test_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()

        dataset_train = Feeder(**self.arg.train_feeder_args)
        dataset_val = Feeder(**self.arg.test_feeder_args)
        print('len(dataset_train):', len(dataset_train))
        print('len(dataset_val):', len(dataset_val))
      
        self.sampler_train = torch.utils.data.RandomSampler(dataset_train)
        self.sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if self.arg.phase == 'train':
            batch_sampler_train = torch.utils.data.BatchSampler(
                self.sampler_train, self.arg.batch_size, drop_last=True)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=dataset_train,
                batch_size=self.arg.batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker *
                torchlight.ngpu(self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=dataset_val,
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker *
                torchlight.ngpu(self.arg.device),
                drop_last=False)

    def load_model(self):

        self.train_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'train'),
                                          comment='train')
        self.validation_logger = SummaryWriter(log_dir=os.path.join(
            self.arg.work_dir, 'validation'),
                                               comment='validation')

        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        
        if self.arg.resume and os.path.isfile(self.arg.resume):
            if self.arg.backbone_only:
                print("Loading checkpoint for backbone '{}'".format(
                    self.arg.resume))

                print('=====> init backbone weights')
                checkpoint = torch.load(self.arg.resume)
                model_dict = self.model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in checkpoint.items()
                    if k in model_dict and 'encoder' in k
                }
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)
                print('updated params:{}'.format(len(pretrained_dict)))

                for k, v in self.model.named_parameters():
                    if 'encoder' in k:
                        v.requires_grad = False
            else:
                print("Loading checkpoint '{}'".format(self.arg.resume))
                checkpoint = torch.load(self.arg.resume)
                self.model.load_state_dict(checkpoint)
        else:   # self.arg.resume=='', enter here
            pretrained_dict = models.resnet34(pretrained=True).state_dict()
            model_dict = self.model.state_dict()
            update_dict = {}
            for k, v in pretrained_dict.items():
                if "layer1" in k:
                    update_dict[k.replace("layer1", "encoder.4", 1)] = v
                elif "layer2" in k:
                    update_dict[k.replace("layer2", "encoder.5", 1)] = v
                elif "layer3" in k:
                    update_dict[k.replace("layer3", "encoder.6", 1)] = v
                elif k == 'conv1.weight':
                    update_dict['encoder.0.weight'] = v
                elif k == 'bn1.weight':
                    update_dict['encoder.1.weight'] = v
                elif k == 'bn1.bias':
                    update_dict['encoder.1.bias'] = v
                elif k == 'bn1.running_mean':
                    update_dict['encoder.1.running_mean'] = v
                elif k == 'bn1.running_var':
                    update_dict['encoder.1.running_var'] = v
                elif k == 'bn1.num_batches_tracked':
                    update_dict['encoder.1.num_batches_tracked'] = v
            print('updated params:{}'.format(len(update_dict)))
            model_dict.update(update_dict)
            self.model.load_state_dict(model_dict)

            print("Random initialization")

        if self.arg.loss == 'NeutralLoss':
            self.loss = loss_neutral.NeutralLoss(
                lam_sub = self.arg.lam_sub, 
                lam_ort = self.arg.lam_ort, 
                lam_clf = self.arg.lam_clf
            )
        else:
            raise ValueError()

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       nesterov=self.arg.nesterov,
                                       weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=self.arg.base_lr,
                                        weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(),
                                           lr=self.arg.base_lr,
                                           alpha=0.9,
                                           momentum=0,
                                           weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (self.arg.lr_decay**np.sum(
                self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        result_frag = []
        label_frag = []

        sub_loss_value = []
        ort_loss_value = []
        clf_loss_value = []

        print("training dataloader length: ", len(loader))

        for data, label, image, subid in loader:
            
            # get data
            # print('data:', list(data.size()))   # [batch_size, 2, 1, 22]
            # print('label:', list(label.size())) # [batch_size, 12] 待改？？？
            # print('image:', list(image.size())) # [batch_size, 2, 3, 256, 256]

            data = data.float().to(self.dev)
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)
            
            # forward
            # summary(self.model, image.shape[1:])
            feature, output, x_subject, x_au = self.model(image)
            # print('feature size:', feature.shape)       # feature: [batch_size, 2, 256]
            # print("output size:", output.shape)         # output: [batch_size*2, num_classes=12]
            # print('x_subject size:', x_subject.shape)   # [batch_size*2, 256]
            # print('x_au size:', x_au.shape)             # [batch_size*2, 256]

            output_cur = output.view(output.shape[0]//2, 2, -1)[:, 1]

            result_frag.append(output_cur.data.cpu().numpy())
            label_frag.append(label.data.cpu().numpy())

            if self.arg.smoothing > 0:
                smooth_label = smooth_one_hot(label, self.arg.smoothing)
                loss, sub_loss, ort_loss, clf_loss = self.loss(smooth_label, output, x_subject, x_au, self.arg.BCE_weight)
            else:
                loss, sub_loss, ort_loss, clf_loss = self.loss(label, output, x_subject, x_au, self.arg.BCE_weight)
            
            if self.arg.flood > 0:
                loss = (loss - self.arg.flood).abs() + self.arg.flood

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['sub_loss'] = sub_loss.data.item()
            self.iter_info['ort_loss'] = ort_loss.data.item()
            self.iter_info['clf_loss'] = clf_loss.data.item()
            loss_value.append(self.iter_info['loss'])
            sub_loss_value.append(self.iter_info['sub_loss'])
            ort_loss_value.append(self.iter_info['ort_loss'])
            clf_loss_value.append(self.iter_info['clf_loss'])
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.epoch_info['mean_sub_loss'] = np.mean(sub_loss_value)
        self.epoch_info['mean_ort_loss'] = np.mean(ort_loss_value)
        self.epoch_info['mean_clf_loss'] = np.mean(clf_loss_value)
        
        self.show_epoch_info()
        self.io.print_timer()

        self.train_logger.add_scalar('loss', self.epoch_info['mean_loss'], self.meta_info['epoch'])
        self.train_logger.add_scalar('sub_loss', self.epoch_info['mean_sub_loss'], self.meta_info['epoch'])
        self.train_logger.add_scalar('ort_loss', self.epoch_info['mean_ort_loss'], self.meta_info['epoch'])
        self.train_logger.add_scalar('clf_loss', self.epoch_info['mean_clf_loss'], self.meta_info['epoch'])
        
        # visualize loss and metrics
        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)
        f1_score, accuracy, train_f1, train_acc = funcs.record_metrics(
            self.result, self.label, self.epoch_info['mean_loss'],
            self.arg.model_args['num_class'], self.arg.work_dir, 'train')
        self.train_logger.add_scalar('loss', self.epoch_info['mean_loss'],
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train-acc', train_acc,
                                     self.meta_info['epoch'])
        self.train_logger.add_scalar('train-F1', train_f1,
                                     self.meta_info['epoch'])
        

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        result_frag = []
        label_frag = []

        loss_value = []
        sub_loss_value = []
        ort_loss_value = []
        clf_loss_value = []

        print("validation dataloader length: ", len(loader))
        for data, label, image, subid in loader:

            # get data
            data = data.float().to(self.dev)
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)

            # inference
            with torch.no_grad():
                feature, output, x_subject, x_au = self.model(image)
            output_cur = output.view(output.shape[0]//2, 2, -1)[:, 1]
            result_frag.append(output_cur.data.cpu().numpy())

            # get loss
            if evaluation:
                if self.arg.smoothing > 0:
                    smooth_label = smooth_one_hot(label, self.arg.smoothing)
                    ret_losses = self.loss(smooth_label, output, x_subject, x_au, self.arg.BCE_weight)
                    loss, sub_loss, ort_loss, clf_loss = Variable(ret_losses[0], requires_grad=True), ret_losses[1], ret_losses[2], ret_losses[3]
                else:   # loss = Variable(self.loss(feature, label), requires_grad=True)
                    ret_losses = self.loss(label, output, x_subject, x_au, self.arg.BCE_weight)
                    loss, sub_loss, ort_loss, clf_loss = Variable(ret_losses[0], requires_grad=True), ret_losses[1], ret_losses[2], ret_losses[3]

                loss_value.append(loss.item())
                sub_loss_value.append(sub_loss.data.item())
                ort_loss_value.append(ort_loss.data.item())
                clf_loss_value.append(clf_loss.data.item())
                label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.epoch_info['mean_sub_loss'] = np.mean(sub_loss_value)
            self.epoch_info['mean_ort_loss'] = np.mean(ort_loss_value)
            self.epoch_info['mean_clf_loss'] = np.mean(clf_loss_value)
            self.show_epoch_info()

            # compute f1 score
            f1_score, accuracy, val_f1, val_acc = funcs.record_metrics(
                self.result, self.label, self.epoch_info['mean_loss'],
                self.arg.model_args['num_class'], self.arg.work_dir, 'val')
            if self.best_aver_f1 < val_f1:
                self.best_aver_f1 = val_f1
                self.best_f1 = f1_score
            np.savetxt(os.path.join(self.arg.work_dir, 'best_f1.txt'),
                       self.best_f1,
                       fmt='%f',
                       delimiter='+')
            if self.best_aver_acc < val_acc:
                self.best_aver_acc = val_acc
                self.best_acc = accuracy
            np.savetxt(os.path.join(self.arg.work_dir, 'best_acc.txt'),
                       self.best_acc,
                       fmt='%f',
                       delimiter='+')

            self.validation_logger.add_scalar('loss',
                                              self.epoch_info['mean_loss'],
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('sub_loss',
                                              self.epoch_info['mean_sub_loss'], 
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('ort_loss',
                                              self.epoch_info['mean_ort_loss'],
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('clf_loss',
                                              self.epoch_info['mean_clf_loss'],
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('val-acc', val_acc,
                                              self.meta_info['epoch'])
            self.validation_logger.add_scalar('val-F1', val_f1,
                                              self.meta_info['epoch'])
            

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Supervised Contrastive Loss with Resnet')

        # region arguments yapf: disable
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--lr_decay', type=float, default=0.1, help='lr decay for optimizer')
        parser.add_argument('--loss', type=str, default='Focal', help='loss for optimizer')
        parser.add_argument('--BCE_weight', type=int, default=[], nargs='+', help='weights for BCE loss')
        parser.add_argument('--smoothing', type=float, default=0.0, help='label smoothing rate')
        parser.add_argument('--flood', type=float, default=0.0, help='flooding for training')
        parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
        parser.add_argument('--backbone_only', type=str2bool, default=True, help='only use backbone weights')
        parser.add_argument('--seed', type=int, default=0, help='random seed')
        parser.add_argument('--lam_sub', type=float, default=1, help='lambda for subject loss')
        parser.add_argument('--lam_ort', type=float, default=1, help='lambda for orthogonal loss')
        parser.add_argument('--lam_clf', type=float, default=1, help='lambda for classification loss')
        # endregion yapf: enable

        return parser
