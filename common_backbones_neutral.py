import torch
import torch.nn as nn
import torch.nn.functional as F

# pre-trained backbone
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    r"""Baseline

    Args:
        num_class (int): Number of classes for the classification task
        backbone (str): choose from 'simple', 'resnet50', 'resnet101', 'vgg16', 'alexnet'
        pooling (bool): pooling or not
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, (T_{in}), C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, num_class)` 
          
    """
    def __init__(self,
                 num_class,
                 backbone='resnet34',
                 pooling=False,
                 d_m=256,
                 **kwargs):
        super().__init__()

        self.backbone = backbone
        self.pooling = pooling
        self.d_m = d_m

        if self.backbone == 'alexnet':
            self.encoder = nn.Sequential(
                *list(models.alexnet(
                    pretrained=False).children())[0],  # [N, 256, 6, 6]
            )
            self.output_channel = 256
            self.output_size = 6

        elif self.backbone == 'vgg16':
            self.encoder = nn.Sequential(
                *list(models.vgg16(
                    pretrained=False).children())[0],  # [N, 512, 8, 8]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'vgg19':
            self.encoder = nn.Sequential(
                *list(models.vgg19(
                    pretrained=False).children())[0],  # [N, 512, 8, 8]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'densenet':
            self.encoder = nn.Sequential(
                *list(models.densenet161(
                    pretrained=False).children())[0],  # [N, 2208, 8, 8]
            )
            self.output_channel = 2208
            self.output_size = 8

        elif self.backbone == 'inception_v3':  # error
            self.encoder = nn.Sequential(
                *list(models.inception_v3(
                    pretrained=False).children())[:-3],  # [N, 512, 7, 7]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'googlenet':  # error
            self.encoder = nn.Sequential(
                *list(models.googlenet(
                    pretrained=False).children())[:-3],  # [N, 512, 7, 7]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'squeezenet':
            self.encoder = nn.Sequential(
                *list(models.squeezenet1_0(
                    pretrained=False).children())[0],  # [N, 512, 15, 15]
            )
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'resnet18-trunc':
            self.encoder = nn.Sequential(
                *list(models.resnet18(pretrained=False).children())
                [:-3],  # [N, 256, image_size // (2^4), _]
            )
            self.output_channel = 256
            self.output_size = 16

        elif self.backbone == 'resnet34-trunc':
            self.encoder = nn.Sequential(
                *list(models.resnet34(pretrained=False).children())
                [:-3],  # [N, 256, image_size // (2^4), _]
            )
            self.output_channel = 256
            self.output_size = 16

        elif self.backbone == 'resnet50-trunc':
            self.encoder = nn.Sequential(
                *list(models.resnet50(pretrained=False).children())
                [:-3],  # [N, 1024, image_size // (2^4), _]
            )
            self.output_channel = 1024
            self.output_size = 16

        elif self.backbone == 'resnet18':
            self.encoder = nn.Sequential(
                *list(models.resnet18(pretrained=False).children())
                [:-1],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 16

        elif self.backbone == 'resnet34':
            self.encoder = nn.Sequential(
                *list(models.resnet34(pretrained=False).children())
                [:-2],  # [N, 512, image_size // (2^4), _]
            ) #### changed -1 -> -2
            # print(self.encoder)
            self.output_channel = 512
            self.output_size = 8

        elif self.backbone == 'resnet50':
            self.encoder = nn.Sequential(
                *list(models.resnet50(pretrained=False).children())
                [:-1],  # [N, 1024, image_size // (2^4), _]
            )
            self.output_channel = 2048
            self.output_size = 16

        elif self.backbone == 'resnet101':
            self.encoder = nn.Sequential(
                *list(models.resnet101(
                    pretrained=False).children())[:-1],  # [N, 2048, 1, 1]
            )
            self.output_channel = 2048
            self.output_size = 1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.pooling == False:
            ### here
            self.mlp_subject = nn.Linear(
                self.output_channel * self.output_size * self.output_size,
                self.d_m)
            self.mlp_au = nn.Linear(
                self.output_channel * self.output_size * self.output_size,
                self.d_m)
            self.final = nn.Sequential(
                #nn.Linear(self.d_m, 64),
                nn.LeakyReLU(inplace=True),
                #nn.Linear(64, num_class),
                nn.Linear(self.d_m, num_class),
                nn.Sigmoid(),
            )
        else:
            self.mlp_subject = nn.Linear(self.output_channel, self.d_m)
            self.mlp_au = nn.Linear(self.output_channel, self.d_m)
            self.final = nn.Sequential(
                #nn.Linear(self.d_m, 64),
                nn.LeakyReLU(inplace=True),
                #nn.Linear(64, num_class),
                nn.Linear(self.d_m, num_class),
                nn.Sigmoid(),
            )

    def forward(self, image, subject_infos=None):
        '''
        image: [N, 2, C, H, W] (2 for [neutral image, current image])
               [batch_size, 2, 3, 256, 256]
        '''
        N, T, C, H, W = image.shape
        x = image.view(-1, C, H, W) # [batch_size*2, 3, 256, 256]
        # neutral image, current image 相邻

        x = self.encoder(x)
        if self.pooling == False:
            ### here
            # print('after encoder:', x.shape)    # [batch_size*2, 512, 8, 8]
            x = x.view(x.shape[0], -1)
            # print('after view:', x.shape)   # [batch_size*2, 512*8*8]
            x_subject = self.mlp_subject(x) # [batch_size*2, 256]
            x_au = self.mlp_au(x)           # [batch_size*2, 256]
        else:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x_subject = self.mlp_subject(x)
            x_au = self.mlp_au(x)

        feature = x_au.view(N, 2, -1)       # [batch_size, 2, 256]

        output = self.final(x_au)           # [8, 12]   # 要不要改成只有current face可以有final的output，变成输出[4, 12]

        return feature, output, x_subject, x_au