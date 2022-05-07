from cProfile import label
from doctest import OutputChecker
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import losses, loss_sup_contrast


# feature分为两部分，subject和au
# subject部分，对应被试人脸，计算相似度，希望相似度尽可能高
class SubjectLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SubjectLoss, self).__init__()

    def forward(self, x_subject, *args):
        # x_subject shape: [2*N, d_m]  (2 for [neutral image, current image])

        # 隔行选出，将x_subject交错分成两部分，neutral和current，计算两个feature的相似性
        x_subject_n = x_subject.view(x_subject.shape[0]//2, 2, -1)[:, 0]
        x_subject_c = x_subject.view(x_subject.shape[0]//2, 2, -1)[:, 1]

        x_subject_n_norm = F.normalize(x_subject_n)
        x_subject_c_norm = F.normalize(x_subject_c)

        N, d_m = x_subject_n.shape

        loss = 1 - torch.cosine_similarity(x_subject_n_norm, x_subject_c_norm, dim=1).mean()
        # dim=1 计算行向量之间的相似度
        
        # TODO: try L1Loss, MSELoss ...

        # print('subjectloss:', loss)
        return loss


# 对于每个图像，subject和au两特征正交
class OrthogonalLoss(nn.Module):
    def __init__(self, **kwargs):
        super(OrthogonalLoss, self).__init__()
        self.L2norm = nn.MSELoss().cuda()

    def forward(self, x_subject, x_au, *args):
        # x_subject.shape=([8, 256])=x_au.shape
        # dim=0 对列；dim=1 对行
        sim = torch.mm(F.normalize(x_subject.t(), p=2, dim=0),
                       F.normalize(x_au, p=2, dim=1))
        sim = torch.diagonal(sim, 0)    # 取对角线
        loss = torch.log(self.L2norm(sim, torch.zeros_like(sim).to(x_subject.device)))*(-1)
        # print('orthogonalloss:', loss)
        return loss


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=0.0, p=1, reduction='sum'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        num = torch.sum(torch.mul(predict, target)) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weights: An array of shape [num_classes,]
        ignore_index: class index to ignore
        outputs: A tensor of shape [N, C, *]
        target: A tensor of ground-truth labels
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weights=None, ignore_index=None, lambda_dice=1, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        # self.weights = torch.tensor(weights)
        self.weights = weights
        self.ignore_index = ignore_index
        self.lambda_dice = lambda_dice

    def forward(self, outputs, targets, *args):
        num_class = outputs.size()[-1]
        outputs = outputs.view(-1, num_class)
        targets = targets.view(-1, num_class)

        N, num_class = outputs.size()

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0

        for i in range(num_class):
            if i != self.ignore_index:
                dice_loss = dice(outputs[:, i], targets[:, i])
                if self.weights is not None:
                    assert self.weights.shape[0] == targets.shape[1], \
                        'Expect weights shape [{}], get[{}]'.format(targets.shape[1], self.weights.shape[0])
                    dice_loss *= 1 - self.weights[i]
                total_loss += dice_loss

        return self.lambda_dice * (total_loss / targets.shape[1])


# au部分照原来方式类似地做clf
class CLFLoss(nn.Module):
    def __init__(self, **kwargs):
        super(CLFLoss, self).__init__()

    def forward(self, outputs, labels, x_au, weights):
        
        loss = 0
        # TODO: clf related losses, try SRERL, SupContrast...

        weight = np.array(weights)
        weight = torch.from_numpy(weight).float()
        outputs_c = outputs.view(outputs.shape[0]//2, 2, -1)[:, 1]  # (4,12)
        srerl_loss = losses.SRERL_loss(weight=weight)(outputs_c, labels)
        
        # 4.7 UPDATE: ADD DICE LOSS ---------- delete
        # loss += DiceLoss(weights=weight).forward(outputs_c, labels)
        
        # add supCon loss - TO DO
        supcon_loss = loss_sup_contrast()

        # print('CLFloss:', loss)
        return loss


# 三个loss合到一起作为neutral的loss
class NeutralLoss(nn.Module):
    def __init__(self, lam_sub=1, lam_ort=1, lam_clf=1, **kwargs):
        super(NeutralLoss, self).__init__()
        self.lambda_sub = lam_sub
        self.lambda_ort = lam_ort
        self.lambda_clf = lam_clf
    
    def forward(self, labels, output, x_subject, x_au, weight):
        sub_loss = SubjectLoss().forward(x_subject)
        ort_loss = OrthogonalLoss().forward(x_subject, x_au)
        clf_loss = CLFLoss().forward(output, labels, x_au, weight)
        loss = self.lambda_sub * sub_loss + self.lambda_ort * ort_loss + self.lambda_clf * clf_loss
        return loss, sub_loss, ort_loss, clf_loss