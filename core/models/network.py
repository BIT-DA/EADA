import torch
import torch.nn as nn
from torchvision import models


class ResNetFc(nn.Module):

    def __init__(self, bottleneck_dim=256, class_num=1000, cfg=None):
        super(ResNetFc, self).__init__()

        self.cfg = cfg

        # ImageNet pretrain model
        if self.cfg.MODEL.BACKBONE.NAME == 'resnet18':
            self.model_resnet = models.resnet18(pretrained=True)
        elif self.cfg.MODEL.BACKBONE.NAME == 'resnet34':
            self.model_resnet = models.resnet34(pretrained=True)
        elif self.cfg.MODEL.BACKBONE.NAME == 'resnet50':
            self.model_resnet = models.resnet50(pretrained=True)
        elif self.cfg.MODEL.BACKBONE.NAME == 'resnet101':
            self.model_resnet = models.resnet101(pretrained=True)
        else:
            raise RuntimeError("Backbone not available: {}".format(self.cfg.MODEL.BACKBONE.NAME))

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)

        self.classifier = nn.Linear(bottleneck_dim, class_num)

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.bn2(x)
        y = self.classifier(x)
        if return_feat:
            return x, y
        else:
            return y

    def output_num(self):
        return self.__in_features

    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.conv1.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.bn1.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.maxpool.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.layer1.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.layer2.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.layer3.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.layer4.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.avgpool.parameters(), 'lr': lr * self.cfg.OPTIM.BASE_LR_MULT},
            {'params': self.bottleneck.parameters()},
            {'params': self.classifier.parameters()},
        ]

        return parameter_list




