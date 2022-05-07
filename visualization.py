# t-SNE 将feature与AU/label的关系可视化
# subject feature w/ AU label, AU feature w/ subject label

import sys
import argparse
from sklearn import cluster
import yaml
import numpy as np
import os
import math
from tqdm import tqdm

# torch
import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn.cluster import KMeans

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor

from tools.utils import funcs, losses, loss_neutral

from tensorboardX import SummaryWriter

from itertools import zip_longest

import torchvision.models as models
from net.baseline import Model

import os
import argparse
import random
import torch
import torchvision
import numpy as np
from torch.utils import data
import copy

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

plt.switch_backend('agg')
import seaborn as sns

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

'''
def mine_nearest_neighbors(features, topk):
    # mine the topk nearest neighbors for every sample
    import faiss
    # features = features.cpu().numpy()
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included
    return indices
'''

def tsne(feature_vector, labels_vector, savepath='save/tsne.png'):
    print(feature_vector.shape, labels_vector.shape)

    def scatter(x, colors):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", 41)) ### change the number of clusters 41=14+(14+13)
        
        # subject feature + au feature in one graph
        pair_palette = np.array(sns.color_palette("Paired", 82))
        pair_palette_1 = torch.tensor(pair_palette).view(41, 2, 3)[:, 0, :]
        pair_palette_2 = torch.tensor(pair_palette).view(41, 2, 3)[:, 1, :]
        pair_palette = np.array(torch.cat((pair_palette_1, pair_palette_2), 0))

        # We create a scatter plot.
        f = plt.figure(figsize=(32, 32))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:, 0],
                        x[:, 1],
                        lw=4,
                        s=120,
                        c=pair_palette[colors.astype(np.int)-1]) ##### change when single/pair features #####
        my_x_ticks = np.arange(-100, 100, 25)
        my_y_ticks = np.arange(-100, 100, 25)
        plt.xticks(my_x_ticks, fontsize=100)
        plt.yticks(my_y_ticks, fontsize=100)

        # We add the labels for each cluster.
        txts = []
        for i in range(41): ### change the number of clusters (the number of subjects)
            # Position of each label.
            xtext, ytext = np.median(x[colors == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=60)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=8, foreground="w"),
                PathEffects.Normal()
            ])
            txts.append(txt)
        return f, ax, sc, txts

    print('tsne start')
    lo_embedded = TSNE(n_components=2).fit_transform(feature_vector)
    print(lo_embedded.shape)
    print('tsne end')

    kmeans = KMeans(n_clusters=41).fit(lo_embedded) ### change the number of clusters # 14?
    predictions = kmeans.labels_
    ari = metrics.adjusted_rand_score(labels_vector, predictions)
    print('ari: ', ari)

    f, ax, sc, txts = scatter(lo_embedded, labels_vector)
    f.savefig(savepath)
    return


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
        if self.arg.resume:
            print('Loading checkpoint "{}"'.format(self.arg.resume))
            checkpoint = torch.load(self.arg.resume)
            self.model.load_state_dict(checkpoint)

    def test(self, evaluation=True):

        self.model.eval()
        
        # loader = self.data_loader['test']
        # new feeder - set shuffle = True
        Feeder = import_class(self.arg.feeder)
        dataset_val = Feeder(**self.arg.test_feeder_args)
        print('len(dataset_val):', len(dataset_val))
        loader = torch.utils.data.DataLoader(
            dataset=dataset_val,
            batch_size=self.arg.test_batch_size,
            shuffle=True,   
            num_workers=self.arg.num_worker *
            torchlight.ngpu(self.arg.device),
            drop_last=False)
        print("validation dataloader length: ", len(loader))

        subject_vectors = []
        au_vectors = []
        pair_vectors = []
        labels = []
        subids = []
        cnt = 0

        for data, label, image, subid in loader:
            
            # get data
            data = data.float().to(self.dev)
            label = label.float().to(self.dev)
            image = image.float().to(self.dev)
            subid = subid.float().to(self.dev)

            # tSNE
            with torch.no_grad():
                feature, output, x_subject, x_au = self.model(image)
                # print("x_subject shape:", x_subject.shape) # [8, 256]
                # print("x_au shape:", x_au.shape) # [8, 256]
                subject_vectors.append(F.normalize(x_subject.cpu()).numpy())
                au_vectors.append(F.normalize(x_au.cpu()).numpy())
                
                pair_vectors.append(F.normalize(x_subject.cpu()).numpy())
                pair_vectors.append(F.normalize(x_au.cpu()).numpy())

                labels.append(label.squeeze(1).cpu().numpy())
                
                subid = subid.cpu().numpy().tolist()
                # print("subid:", subid)
                # double - neutral+current - subject feature only
                double_subid = []
                for i in range(len(subid)):
                    double_subid.append(subid[i])
                    double_subid.append(subid[i])
                # print("double_subid:", double_subid)

                # subject feature+au feature - double subid
                pair_double_subid = double_subid
                for i in range(len(subid)):
                    pair_double_subid.append(subid[i]+41)
                    pair_double_subid.append(subid[i]+41)
                # print("double_subid:", double_subid)

                subids.append(np.array(pair_double_subid))  ##### change when single/pair features #####
            
            cnt += 1
            if cnt >= 5000:
                break
            
        # tSNE
        ##### change when single/pair features #####
        feature_vectors = np.concatenate(pair_vectors).reshape(-1, 256)
        
        # label = np.concatenate(labels)
        # clf = KMeans(n_clusters=24)
        # clf.fit(label)
        # label = clf.labels_
        
        label_vectors = np.concatenate(labels, axis=0)
        subid_vectors = np.concatenate(subids, axis=0)
        
        tsne(feature_vector=feature_vectors, labels_vector=subid_vectors, savepath='tsne_dir/pair_ort+sub_subf+auf_tSNE{}.png'.format(cnt))


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