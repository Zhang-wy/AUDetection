from cProfile import label
from doctest import OutputChecker
from re import sub
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.utils import losses, loss_sup_contrast

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# feature分为两部分，subject和au
# subject部分，对应被试人脸
class SubjectLoss(nn.Module):
    def __init__(self, **kwargs):
        super(SubjectLoss, self).__init__()

    def forward(self, subids, x_subject, *args):
        # x_subject shape: [2*N, d_m]  (2 for [neutral image, current image])

        # 隔行选出，将x_subject交错分成两部分，neutral和current，计算两个feature的相似性
        x_subject_n = F.normalize(x_subject.view(x_subject.shape[0]//2, 2, -1)[:, 0])
        x_subject_c = F.normalize(x_subject.view(x_subject.shape[0]//2, 2, -1)[:, 1])

        N, d_m = x_subject_n.shape

        # 希望相同subid的feature尽可能相似，不同subid的feature尽可能不相似
        subid = subids[:, None] # (N, 1)
        target = torch.eq(subid, subid.t()).float().to(device)
        dot_product_n = torch.matmul(x_subject_n, x_subject_n.t()).to(device)
        dot_product_c = torch.matmul(x_subject_c, x_subject_c.t()).to(device)
        eye = torch.eye(N, N).bool().to(device) # 去掉对角线
        dot_product_n = dot_product_n.masked_fill(eye, 1)
        dot_product_c = dot_product_c.masked_fill(eye, 1)
        loss = abs(target-dot_product_n).sum()/(N*(N-1)) + abs(target-dot_product_c).sum()/(N*(N-1))

        # neutral和current
        loss += 1 - torch.cosine_similarity(x_subject_n, x_subject_c, dim=1).mean()

        # print('subject loss:', loss)
        return loss


# 对于每个图像，subject和au两特征正交
class OrthogonalLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalLoss, self).__init__()
        self.L2norm = nn.MSELoss().cuda()
        self.gamma = gamma

    def forward(self, x_subject, x_au, *args):
        # x_subject.shape=([8, 256])=x_au.shape
        # dim=0 对列；dim=1 对行
        # sim = torch.mm(F.normalize(x_subject.t(), p=2, dim=0),
        #                F.normalize(x_au, p=2, dim=1))
        # sim = torch.diagonal(sim, 0)    # 取对角线
        # loss = (-1) * torch.log(self.L2norm(sim, torch.zeros_like(sim).to(x_subject.device)))
        
        N = x_subject.shape[0]
        x_subject_norm = F.normalize(x_subject, p=2, dim=1)
        x_au_norm = F.normalize(x_au, p=2, dim=1)
        similarity = abs(torch.cosine_similarity(x_subject_norm, x_au_norm, dim=1))
        loss = (-1)*(torch.log(1-similarity)).sum() / N

        # print('orthogonal loss:', loss)
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
        # x_au = x_au.view(x_au.shape[0]//2, 2, -1)   # (4, 2, xxx)
        # supcon_loss = loss_sup_contrast.SupConLoss(AUchannels=256, lam_clf=0, lam_ctr=1)(
        #     outputs=outputs_c,
        #     features=x_au,
        #     targets=labels,
        #     use_ctr=True,
        #     weight=weight
        # )

        loss = srerl_loss

        # print('CLF loss:', loss)
        return loss


# 三个loss合到一起作为neutral的loss
class NeutralLoss(nn.Module):
    def __init__(self, lam_sub=1, lam_ort=1, lam_clf=1, **kwargs):
        super(NeutralLoss, self).__init__()
        self.lambda_sub = lam_sub
        self.lambda_ort = lam_ort
        self.lambda_clf = lam_clf
    
    def forward(self, labels, subids, output, x_subject, x_au, weight):
        sub_loss = SubjectLoss().forward(subids, x_subject)
        ort_loss = OrthogonalLoss().forward(x_subject, x_au)
        clf_loss = CLFLoss().forward(output, labels, x_au, weight)
        loss = self.lambda_sub * sub_loss + self.lambda_ort * ort_loss + self.lambda_clf * clf_loss
        return loss, sub_loss, ort_loss, clf_loss