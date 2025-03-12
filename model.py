import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def nan_checker(inpt):
    if torch.any(torch.isnan(inpt)):
        raise ValueError("NaN found")
    if torch.any(torch.isinf(inpt)):
        raise ValueError("Infinite found")


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        if layer.weight.ndimension() == 4:
            (n_out, n_in, height, width) = layer.weight.size()
            n = n_in * height * width
        elif layer.weight.ndimension() == 3:
            (n_out, n_in, height) = layer.weight.size()
            n = n_in * height
        elif layer.weight.ndimension() == 2:
            (n_out, n) = layer.weight.size()

        std = math.sqrt(2. / n)
        scale = std * math.sqrt(3.)
        layer.weight.data.uniform_(-scale, scale)

        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    else:
        pass


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer, init_type="ones"):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    if init_type == "ones":
        layer.weight.data.fill_(1.)
    elif init_type == "eye":
        eye = nn.Parameter(torch.eye(4))
        layer.weight = eye
    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class LightSERNetMoE(nn.Module):
    def __init__(self, skip_final_fc=False):
        super(LightSERNetMoE, self).__init__()
        self.mod_neutral = LightSERNet(moe=True)
        self.mod_happy = LightSERNet(moe=True)
        self.mod_sad = LightSERNet(moe=True)
        self.mod_angry = LightSERNet(moe=True)

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a


class LightSERNetMoEFF1(nn.Module):
    def __init__(self, skip_final_fc=False, dropout=.3):
        super(LightSERNetMoEFF1, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=1)
        self.mod_happy = LightSERNet(moe=True, fusion_level=1)
        self.mod_sad = LightSERNet(moe=True, fusion_level=1)
        self.mod_angry = LightSERNet(moe=True, fusion_level=1)

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.p1 = nn.Identity()
            mod.p2 = nn.Identity()
            mod.p3 = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.avgPool22_1 = nn.Identity()

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        xa = self.p1(input[0].clone())
        xb = self.p2(input[0].clone())
        xc = self.p3(input[0].clone())

        pad_p1 = F.pad(xa, (2, 3, 2, 3))
        pad_p2 = F.pad(xb, (2, 3, 2, 3))
        pad_p3 = F.pad(xc, (2, 3, 2, 3))

        p1 = self.avgPool22_1(pad_p1)
        p2 = self.avgPool22_1(pad_p2)
        p3 = self.avgPool22_1(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a


class LightSERNetMoEFF2(nn.Module):
    def __init__(self, skip_final_fc=False, dropout=.3):
        super(LightSERNetMoEFF2, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=2)
        self.mod_happy = LightSERNet(moe=True, fusion_level=2)
        self.mod_sad = LightSERNet(moe=True, fusion_level=2)
        self.mod_angry = LightSERNet(moe=True, fusion_level=2)

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))

        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.p1 = nn.Identity()
            mod.p2 = nn.Identity()
            mod.p3 = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.avgPool22_1 = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.avgPool22_2 = nn.Identity()

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        xa = self.p1(input[0].clone())
        xb = self.p2(input[0].clone())
        xc = self.p3(input[0].clone())

        pad_p1 = F.pad(xa, (2, 3, 2, 3))
        pad_p2 = F.pad(xb, (2, 3, 2, 3))
        pad_p3 = F.pad(xc, (2, 3, 2, 3))

        p1 = self.avgPool22_1(pad_p1)
        p2 = self.avgPool22_1(pad_p2)
        p3 = self.avgPool22_1(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22_2(pad_x)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a


class LightSERNetMoEFF3(nn.Module):
    def __init__(self, skip_final_fc=False, dropout=.3):
        super(LightSERNetMoEFF3, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=3)
        self.mod_happy = LightSERNet(moe=True, fusion_level=3)
        self.mod_sad = LightSERNet(moe=True, fusion_level=3)
        self.mod_angry = LightSERNet(moe=True, fusion_level=3)

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.p1 = nn.Identity()
            mod.p2 = nn.Identity()
            mod.p3 = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.avgPool22_1 = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.avgPool22_2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.avgPool22_3 = nn.Identity()

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        xa = self.p1(input[0].clone())
        xb = self.p2(input[0].clone())
        xc = self.p3(input[0].clone())

        pad_p1 = F.pad(xa, (2, 3, 2, 3))
        pad_p2 = F.pad(xb, (2, 3, 2, 3))
        pad_p3 = F.pad(xc, (2, 3, 2, 3))

        p1 = self.avgPool22_1(pad_p1)
        p2 = self.avgPool22_1(pad_p2)
        p3 = self.avgPool22_1(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22_2(pad_x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22_3(pad_x)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a



class LightSERNetMoEFF4(nn.Module):
    def __init__(self, skip_final_fc=False, dropout=.3):
        super(LightSERNetMoEFF4, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=4)
        self.mod_happy = LightSERNet(moe=True, fusion_level=4)
        self.mod_sad = LightSERNet(moe=True, fusion_level=4)
        self.mod_angry = LightSERNet(moe=True, fusion_level=4)

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.p1 = nn.Identity()
            mod.p2 = nn.Identity()
            mod.p3 = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.avgPool22_1 = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.avgPool22_2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.avgPool22_3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.avgPool21_4 = nn.Identity()

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        xa = self.p1(input[0].clone())
        xb = self.p2(input[0].clone())
        xc = self.p3(input[0].clone())

        pad_p1 = F.pad(xa, (2, 3, 2, 3))
        pad_p2 = F.pad(xb, (2, 3, 2, 3))
        pad_p3 = F.pad(xc, (2, 3, 2, 3))

        p1 = self.avgPool22_1(pad_p1)
        p2 = self.avgPool22_1(pad_p2)
        p3 = self.avgPool22_1(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22_2(pad_x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22_3(pad_x)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21_4(pad_x)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a


class LightSERNetMoEFF5(nn.Module):
    def __init__(self, skip_final_fc=False, dropout=.3):
        super(LightSERNetMoEFF5, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=5)
        self.mod_happy = LightSERNet(moe=True, fusion_level=5)
        self.mod_sad = LightSERNet(moe=True, fusion_level=5)
        self.mod_angry = LightSERNet(moe=True, fusion_level=5)

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.p1 = nn.Identity()
            mod.p2 = nn.Identity()
            mod.p3 = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.avgPool22_1 = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.avgPool22_2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.avgPool22_3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.avgPool21_4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.avgPool21_5 = nn.Identity()

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        xa = self.p1(input[0].clone())
        xb = self.p2(input[0].clone())
        xc = self.p3(input[0].clone())

        pad_p1 = F.pad(xa, (2, 3, 2, 3))
        pad_p2 = F.pad(xb, (2, 3, 2, 3))
        pad_p3 = F.pad(xc, (2, 3, 2, 3))

        p1 = self.avgPool22_1(pad_p1)
        p2 = self.avgPool22_1(pad_p2)
        p3 = self.avgPool22_1(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22_2(pad_x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22_3(pad_x)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21_4(pad_x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21_5(pad_x)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a


class LightSERNetMoEFF6(nn.Module):
    def __init__(self, skip_final_fc=False, dropout=.3):
        super(LightSERNetMoEFF6, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=6)
        self.mod_happy = LightSERNet(moe=True, fusion_level=6)
        self.mod_sad = LightSERNet(moe=True, fusion_level=6)
        self.mod_angry = LightSERNet(moe=True, fusion_level=6)

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=1,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(in_channels=160, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)

        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.p1 = nn.Identity()
            mod.p2 = nn.Identity()
            mod.p3 = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.avgPool22_1 = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.avgPool22_2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.avgPool22_3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.avgPool21_4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.avgPool21_5 = nn.Identity()
            mod.conv6 = nn.Identity()
            mod.bn6 = nn.Identity()

        if skip_final_fc:
            self.fc = nn.Identity()
            self.act = nn.Identity()
        else:
            self.fc = nn.Linear(in_features=4, out_features=4)
            self.act = nn.Sigmoid()

    def forward(self, *input):
        xa = self.p1(input[0].clone())
        xb = self.p2(input[0].clone())
        xc = self.p3(input[0].clone())

        pad_p1 = F.pad(xa, (2, 3, 2, 3))
        pad_p2 = F.pad(xb, (2, 3, 2, 3))
        pad_p3 = F.pad(xc, (2, 3, 2, 3))

        p1 = self.avgPool22_1(pad_p1)
        p2 = self.avgPool22_1(pad_p2)
        p3 = self.avgPool22_1(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22_2(pad_x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22_3(pad_x)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21_4(pad_x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21_5(pad_x)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))
        x = torch.reshape(x, (x.shape[0], x.shape[1]))

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.fc(self.act(x))

        return x, n, h, s, a


class LightSERNetMoEBF7(nn.Module):
    def __init__(self):
        super(LightSERNetMoEBF7, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=-7)
        self.mod_happy = LightSERNet(moe=True, fusion_level=-7)
        self.mod_sad = LightSERNet(moe=True, fusion_level=-7)
        self.mod_angry = LightSERNet(moe=True, fusion_level=-7)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()
            mod.drop = nn.Identity()
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=320*4, out_features=4)
        self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(self.act(x)))

        return x


class LightSERNetMoEBF6(nn.Module):
    def __init__(self):
        super(LightSERNetMoEBF6, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=-6)
        self.mod_happy = LightSERNet(moe=True, fusion_level=-6)
        self.mod_sad = LightSERNet(moe=True, fusion_level=-6)
        self.mod_angry = LightSERNet(moe=True, fusion_level=-6)

        self.conv6 = nn.Conv2d(in_channels=160*4, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()
            mod.conv6 = nn.Identity()
            mod.bn6 = nn.Identity()
            mod.drop = nn.Identity()
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(in_features=320, out_features=4)
        self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(self.act(x)))

        return x


class LightSERNetMoEBF5(nn.Module):
    def __init__(self):
        super(LightSERNetMoEBF5, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=-5)
        self.mod_happy = LightSERNet(moe=True, fusion_level=-5)
        self.mod_sad = LightSERNet(moe=True, fusion_level=-5)
        self.mod_angry = LightSERNet(moe=True, fusion_level=-5)

        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.conv5 = nn.Conv2d(in_channels=128*4, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(in_channels=160, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()
            mod.conv6 = nn.Identity()
            mod.bn6 = nn.Identity()
            mod.avgPool21_5 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.drop = nn.Identity()
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(in_features=320, out_features=4)
        self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21_5(pad_x)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(self.act(x)))

        return x


class LightSERNetMoEBF4(nn.Module):
    def __init__(self):
        super(LightSERNetMoEBF4, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=-4)
        self.mod_happy = LightSERNet(moe=True, fusion_level=-4)
        self.mod_sad = LightSERNet(moe=True, fusion_level=-4)
        self.mod_angry = LightSERNet(moe=True, fusion_level=-4)

        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.conv4 = nn.Conv2d(in_channels=96*4, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(in_channels=160, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()
            mod.conv6 = nn.Identity()
            mod.bn6 = nn.Identity()
            mod.avgPool21_5 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.avgPool21_4 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.drop = nn.Identity()
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(in_features=320, out_features=4)
        self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21_4(pad_x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21_5(pad_x)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(self.act(x)))

        return x


class LightSERNetMoEBF3(nn.Module):
    def __init__(self):
        super(LightSERNetMoEBF3, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=-3)
        self.mod_happy = LightSERNet(moe=True, fusion_level=-3)
        self.mod_sad = LightSERNet(moe=True, fusion_level=-3)
        self.mod_angry = LightSERNet(moe=True, fusion_level=-3)

        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.conv3 = nn.Conv2d(in_channels=64*4, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(in_channels=160, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()
            mod.conv6 = nn.Identity()
            mod.bn6 = nn.Identity()
            mod.avgPool21_5 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.avgPool21_4 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.avgPool22_3 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.drop = nn.Identity()
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(in_features=320, out_features=4)
        self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22_3(pad_x)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21_4(pad_x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21_5(pad_x)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(self.act(x)))

        return x


class LightSERNetMoEBF2(nn.Module):
    def __init__(self):
        super(LightSERNetMoEBF2, self).__init__()
        self.mod_neutral = LightSERNet(moe=True, fusion_level=-2)
        self.mod_happy = LightSERNet(moe=True, fusion_level=-2)
        self.mod_sad = LightSERNet(moe=True, fusion_level=-2)
        self.mod_angry = LightSERNet(moe=True, fusion_level=-2)

        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.conv2 = nn.Conv2d(in_channels=32*4, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(in_channels=160, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()
            mod.conv6 = nn.Identity()
            mod.bn6 = nn.Identity()
            mod.avgPool21_5 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.avgPool21_4 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.avgPool22_3 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.avgPool22_2 = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.drop = nn.Identity()
        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(in_features=320, out_features=4)
        self.act = nn.Sigmoid()

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n, h, s, a), 1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22_2(pad_x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22_3(pad_x)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21_4(pad_x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21_5(pad_x)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(self.act(x)))

        return x


class LightSERNet(nn.Module):
    # from https://github.com/AryaAftab/LIGHT-SERNET
    def __init__(self, dropout=.3, input_type="mfcc", moe=False,
                 fusion_level=None):
        super(LightSERNet, self).__init__()
        self.fusion_level = fusion_level
        if input_type == "mfcc":
            number_of_channel = 1
        elif input_type == "spectrogram":
            number_of_channel = 1
        elif input_type == "mel_spectrogram":
            number_of_channel = 1
        else:
            raise ValueError('input_type not valid!')

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=number_of_channel,
                                          out_channels=32, kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=number_of_channel,
                                          out_channels=32, kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=number_of_channel,
                                          out_channels=32, kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32),
                                nn.ReLU())

        self.avgPool22_1 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_2 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool22_3 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21_4 = nn.AvgPool2d(kernel_size=(2, 1))
        self.avgPool21_5 = nn.AvgPool2d(kernel_size=(2, 1))

        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=160,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(in_channels=160, out_channels=320,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)

        if moe:
            self.fc = nn.Linear(in_features=320, out_features=1)
        else:
            self.fc = nn.Linear(in_features=320, out_features=4)

    def forward(self, *input):
        if self.fusion_level:
            if self.fusion_level < 0:
                p1 = self.p1(input[0].clone())
                p2 = self.p2(input[0].clone())
                p3 = self.p3(input[0].clone())

                pad_p1 = F.pad(p1, (2, 3, 2, 3))
                pad_p2 = F.pad(p2, (2, 3, 2, 3))
                pad_p3 = F.pad(p3, (2, 3, 2, 3))

                p1 = self.avgPool22_1(pad_p1)
                p2 = self.avgPool22_1(pad_p2)
                p3 = self.avgPool22_1(pad_p3)

                x = torch.cat((p1, p2, p3), dim=-1)
            else:
                x = input[0]
        else:
            p1 = self.p1(input[0].clone())
            p2 = self.p2(input[0].clone())
            p3 = self.p3(input[0].clone())

            pad_p1 = F.pad(p1, (2, 3, 2, 3))
            pad_p2 = F.pad(p2, (2, 3, 2, 3))
            pad_p3 = F.pad(p3, (2, 3, 2, 3))

            p1 = self.avgPool22_1(pad_p1)
            p2 = self.avgPool22_1(pad_p2)
            p3 = self.avgPool22_1(pad_p3)

            x = torch.cat((p1, p2, p3), dim=-1)

        if self.fusion_level:
            if 0 < self.fusion_level < 2 or self.fusion_level < -2:
                x = self.conv2(x)
                x = self.relu(self.bn2(x))
                pad_x = F.pad(x, (51, 51, 7, 8))
                x = self.avgPool22_2(pad_x)
        else:
            x = self.conv2(x)
            x = self.relu(self.bn2(x))
            pad_x = F.pad(x, (51, 51, 7, 8))
            x = self.avgPool22_2(pad_x)

        if self.fusion_level:
            if 0 < self.fusion_level < 3 or self.fusion_level < -3:
                x = self.conv3(x)
                x = self.relu(self.bn3(x))
                pad_x = F.pad(x, (51, 51, 8, 7))
                x = self.avgPool22_3(pad_x)
        else:
            x = self.conv3(x)
            x = self.relu(self.bn3(x))
            pad_x = F.pad(x, (51, 51, 8, 7))
            x = self.avgPool22_3(pad_x)

        if self.fusion_level:
            if 0 < self.fusion_level < 4 or self.fusion_level < -4:
                x = self.conv4(x)
                x = self.relu(self.bn4(x))
                pad_x = F.pad(x, (0, 0, 7, 8))
                x = self.avgPool21_4(pad_x)
        else:
            x = self.conv4(x)
            x = self.relu(self.bn4(x))
            pad_x = F.pad(x, (0, 0, 7, 8))
            x = self.avgPool21_4(pad_x)

        if self.fusion_level:
            if 0 < self.fusion_level < 5 or self.fusion_level < -5:
                x = self.conv5(x)
                x = self.relu(self.bn5(x))
                pad_x = F.pad(x, (0, 0, 8, 7))
                x = self.avgPool21_5(pad_x)
        else:
            x = self.conv5(x)
            x = self.relu(self.bn5(x))
            pad_x = F.pad(x, (0, 0, 8, 7))
            x = self.avgPool21_5(pad_x)

        if self.fusion_level:
            if 0 < self.fusion_level < 6 or self.fusion_level < -6:
                x = self.conv6(x)
                x = self.gap(self.relu(self.bn6(x)))
                x = torch.reshape(x, (x.shape[0], x.shape[1]))
        else:
            x = self.conv6(x)
            x = self.gap(self.relu(self.bn6(x)))
            x = torch.reshape(x, (x.shape[0], x.shape[1]))

        x = self.fc(self.drop(x))

        return x


class LightSERNetX4(nn.Module):
    # from https://github.com/AryaAftab/LIGHT-SERNET
    def __init__(self, dropout=.3, input_type="mfcc", moe=False):
        super(LightSERNetX4, self).__init__()
        if input_type == "mfcc":
            number_of_channel = 1
        elif input_type == "spectrogram":
            number_of_channel = 1
        elif input_type == "mel_spectrogram":
            number_of_channel = 1
        else:
            raise ValueError('input_type not valid!')

        self.p1 = nn.Sequential(nn.Conv2d(in_channels=number_of_channel,
                                          out_channels=32*2,
                                          kernel_size=(11, 1),
                                          stride=(1, 1), padding=(5, 0)),
                                nn.BatchNorm2d(32*2),
                                nn.ReLU())
        self.p2 = nn.Sequential(nn.Conv2d(in_channels=number_of_channel,
                                          out_channels=32*2,
                                          kernel_size=(1, 9),
                                          stride=(1, 1), padding=(0, 4)),
                                nn.BatchNorm2d(32*2),
                                nn.ReLU())
        self.p3 = nn.Sequential(nn.Conv2d(in_channels=number_of_channel,
                                          out_channels=32*2,
                                          kernel_size=(3, 3),
                                          stride=(1, 1), padding=(1, 1)),
                                nn.BatchNorm2d(32*2),
                                nn.ReLU())

        self.avgPool22 = nn.AvgPool2d(kernel_size=(2, 2))
        self.avgPool21 = nn.AvgPool2d(kernel_size=(2, 1))
        self.drop = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(in_channels=32*2, out_channels=64*2,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64*2)

        self.conv3 = nn.Conv2d(in_channels=64*2, out_channels=96*2,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(96*2)

        self.conv4 = nn.Conv2d(in_channels=96*2, out_channels=128*2,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(128*2)

        self.conv5 = nn.Conv2d(in_channels=128*2, out_channels=160*2,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(160*2)

        self.conv6 = nn.Conv2d(in_channels=160*2, out_channels=320*2,
                               kernel_size=(1, 1), stride=(1, 1),
                               padding=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(320*2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)

        if moe:
            self.fc = nn.Linear(in_features=320, out_features=1)
        else:
            self.fc = nn.Linear(in_features=320*2, out_features=4)

    def forward(self, *input):
        p1 = self.p1(input[0].clone())
        p2 = self.p2(input[0].clone())
        p3 = self.p3(input[0].clone())

        pad_p1 = F.pad(p1, (2, 3, 2, 3))
        pad_p2 = F.pad(p2, (2, 3, 2, 3))
        pad_p3 = F.pad(p3, (2, 3, 2, 3))

        p1 = self.avgPool22(pad_p1)
        p2 = self.avgPool22(pad_p2)
        p3 = self.avgPool22(pad_p3)

        x = torch.cat((p1, p2, p3), dim=-1)

        x = self.conv2(x)
        x = self.relu(self.bn2(x))
        pad_x = F.pad(x, (51, 51, 7, 8))
        x = self.avgPool22(pad_x)

        x = self.conv3(x)
        x = self.relu(self.bn3(x))
        pad_x = F.pad(x, (51, 51, 8, 7))
        x = self.avgPool22(pad_x)

        x = self.conv4(x)
        x = self.relu(self.bn4(x))
        pad_x = F.pad(x, (0, 0, 7, 8))
        x = self.avgPool21(pad_x)

        x = self.conv5(x)
        x = self.relu(self.bn5(x))
        pad_x = F.pad(x, (0, 0, 8, 7))
        x = self.avgPool21(pad_x)

        x = self.conv6(x)
        x = self.gap(self.relu(self.bn6(x)))

        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        x = self.fc(self.drop(x))

        return x


class MACNNMoE(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 skip_final_fc=False, fc_bias=False, act=None):
        super(MACNNMoE, self).__init__()
        self.skip_final_fc = skip_final_fc
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True)
        self.act = act

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=fc_bias)

        if self.skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        n, pre_n = self.mod_neutral(input[0].clone())
        h, pre_h = self.mod_happy(input[0].clone())
        s, pre_s = self.mod_sad(input[0].clone())
        a, pre_a = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEFF1(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 fusion_level=1, skip_final_fc=False):
        super(MACNNMoEFF1, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv1a = nn.Identity()
            mod.conv1b = nn.Identity()
            mod.bn1a = nn.Identity()
            mod.bn1b = nn.Identity()

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=False)
        if skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.relu(self.bn1a(xa))
        xb = self.conv1b(input[0])
        xb = self.relu(self.bn1b(xb))
        x = torch.cat((xa, xb), 1)

        n, pre_n = self.mod_neutral(x.clone())
        h, pre_h = self.mod_happy(x.clone())
        s, pre_s = self.mod_sad(x.clone())
        a, pre_a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEFF2(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256, fusion_level=2,
                 skip_final_fc=False):
        super(MACNNMoEFF2, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16,
                               out_channels=32, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv1a = nn.Identity()
            mod.conv1b = nn.Identity()
            mod.bn1a = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=False)

        if skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.relu(self.bn1a(xa))
        xb = self.conv1b(input[0])
        xb = self.relu(self.bn1b(xb))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        n, pre_n = self.mod_neutral(x.clone())
        h, pre_h = self.mod_happy(x.clone())
        s, pre_s = self.mod_sad(x.clone())
        a, pre_a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEFF3(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256, fusion_level=3,
                 skip_final_fc=False):
        super(MACNNMoEFF3, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16,
                               out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32,
                               out_channels=48, padding=1)

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv1a = nn.Identity()
            mod.conv1b = nn.Identity()
            mod.bn1a = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=False)

        if skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.relu(self.bn1a(xa))
        xb = self.conv1b(input[0])
        xb = self.relu(self.bn1b(xb))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        n, pre_n = self.mod_neutral(x.clone())
        h, pre_h = self.mod_happy(x.clone())
        s, pre_s = self.mod_sad(x.clone())
        a, pre_a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEFF4(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256, fusion_level=4,
                 skip_final_fc=False):
        super(MACNNMoEFF4, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16,
                               out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32,
                               out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48,
                               out_channels=64, padding=1)

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv1a = nn.Identity()
            mod.conv1b = nn.Identity()
            mod.bn1a = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=False)

        if skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.relu(self.bn1a(xa))
        xb = self.conv1b(input[0])
        xb = self.relu(self.bn1b(xb))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        x = self.relu(self.bn4(self.conv4(x)))

        n, pre_n = self.mod_neutral(x.clone())
        h, pre_h = self.mod_happy(x.clone())
        s, pre_s = self.mod_sad(x.clone())
        a, pre_a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEFF5(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256, fusion_level=5,
                 skip_final_fc=False):
        super(MACNNMoEFF5, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16,
                               out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32,
                               out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48,
                               out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64,
                               out_channels=80, padding=1)

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv1a = nn.Identity()
            mod.conv1b = nn.Identity()
            mod.bn1a = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=False)

        if skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.relu(self.bn1a(xa))
        xb = self.conv1b(input[0])
        xb = self.relu(self.bn1b(xb))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        x = self.relu(self.bn4(self.conv4(x)))

        x = self.relu(self.bn5(self.conv5(x)))

        n, pre_n = self.mod_neutral(x.clone())
        h, pre_h = self.mod_happy(x.clone())
        s, pre_s = self.mod_sad(x.clone())
        a, pre_a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEFF6(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 fusion_level=6, skip_final_fc=False):
        super(MACNNMoEFF6, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True,
                               fusion_level=fusion_level)

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16,
                               out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32,
                               out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48,
                               out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64,
                               out_channels=80, padding=1)

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv1a = nn.Identity()
            mod.conv1b = nn.Identity()
            mod.bn1a = nn.Identity()
            mod.bn1b = nn.Identity()
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(in_features=4,
                            out_features=4, bias=False)

        if skip_final_fc:
            self.act = None
            self.fc = nn.Identity()

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.relu(self.bn1a(xa))
        xb = self.conv1b(input[0])
        xb = self.relu(self.bn1b(xb))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        x = self.relu(self.bn4(self.conv4(x)))

        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = F.softmax(torch.mul(q, k), dim=1)
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        n, pre_n = self.mod_neutral(x.clone())
        h, pre_h = self.mod_happy(x.clone())
        s, pre_s = self.mod_sad(x.clone())
        a, pre_a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        if self.act:
            x = self.act(x)
        x = self.fc(x)

        return x, n, h, s, a


class MACNNMoEBF7(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256):
        super(MACNNMoEBF7, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=True)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=True)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=True)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()

        self.fc = nn.Linear(in_features=64*4,
                            out_features=4)

    def forward(self, *input):
        n, _ = self.mod_neutral(input[0].clone())
        h, _ = self.mod_happy(input[0].clone())
        s, _ = self.mod_sad(input[0].clone())
        a, _ = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        x = self.fc(x)

        return x


class MACNNMoEBF6(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 fusion_level=-6):
        super(MACNNMoEBF6, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=False,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=False,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80 * 4,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80 * 4,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80 * 4,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        self.fc = nn.Linear(in_features=self.attention_hidden,
                            out_features=4)

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)
            attention = F.softmax(torch.mul(q, k), dim=1)
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMoEBF5(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 fusion_level=-5):
        super(MACNNMoEBF5, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=False,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=False,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)

        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64 * 4,
                               out_channels=80, padding=1)
        self.bn5 = nn.BatchNorm2d(80)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        self.fc = nn.Linear(in_features=self.attention_hidden,
                            out_features=4)

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = F.softmax(torch.mul(q, k), dim=1)
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMoEBF4(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 fusion_level=-4):
        super(MACNNMoEBF4, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=False,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=False,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)

        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48 * 4,
                               out_channels=64, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64,
                               out_channels=80, padding=1)
        self.bn5 = nn.BatchNorm2d(80)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        self.fc = nn.Linear(in_features=self.attention_hidden,
                            out_features=4)

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = F.softmax(torch.mul(q, k), dim=1)
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMoEBF3(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
        fusion_level=-3):
        super(MACNNMoEBF3, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=False,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=False,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32 * 4,
                               out_channels=48, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48,
                               out_channels=64, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64,
                               out_channels=80, padding=1)
        self.bn5 = nn.BatchNorm2d(80)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        self.fc = nn.Linear(in_features=self.attention_hidden,
                            out_features=4)

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = F.softmax(torch.mul(q, k), dim=1)
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMoEBF2(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256,
                 fusion_level=-2):
        super(MACNNMoEBF2, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 preMACNN_classifier=False,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             preMACNN_classifier=False,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               preMACNN_classifier=False,
                               fusion_level=fusion_level)

        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16 * 4,
                               out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32,
                               out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48,
                               out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64,
                               out_channels=80, padding=1)

        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv2 = nn.Identity()
            mod.bn2 = nn.Identity()
            mod.conv3 = nn.Identity()
            mod.bn3 = nn.Identity()
            mod.conv4 = nn.Identity()
            mod.bn4 = nn.Identity()
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        self.fc = nn.Linear(in_features=self.attention_hidden,
                            out_features=4)

    def forward(self, *input):
        n = self.mod_neutral(input[0].clone())
        h = self.mod_happy(input[0].clone())
        s = self.mod_sad(input[0].clone())
        a = self.mod_angry(input[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()),
                      dim=1)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)
            attention = F.softmax(torch.mul(q, k), dim=1)
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNN(nn.Module):
    # adapted from https://github.com/lessonxmk/head_fusion
    def __init__(self, attention_heads=8, attention_hidden=256, num_emo=4,
                 preMACNN_classifier=False, fusion_level=None):
        super(MACNN, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.fusion_level = fusion_level
        self.preMACNN_classifier = preMACNN_classifier

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16,
                               out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32,
                               out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48,
                               out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64,
                               out_channels=80, padding=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)

        self.relu = nn.ReLU()

        if self.preMACNN_classifier:
            self.fc = nn.Linear(in_features=self.attention_hidden,
                                out_features=1)
        else:
            self.fc = nn.Linear(in_features=self.attention_hidden,
                                out_features=num_emo)

        self.dropout = nn.Dropout(0.5)

        if not self.fusion_level or 0 < self.fusion_level < 6:
            self.gap = nn.AdaptiveAvgPool2d(1)

            self.attention_query = nn.ModuleList()
            self.attention_key = nn.ModuleList()
            self.attention_value = nn.ModuleList()

            for i in range(self.attention_heads):
                self.attention_query.append(
                    nn.Conv2d(in_channels=80,
                              out_channels=self.attention_hidden,
                              kernel_size=1))
                self.attention_key.append(
                    nn.Conv2d(in_channels=80,
                              out_channels=self.attention_hidden,
                              kernel_size=1))
                self.attention_value.append(
                    nn.Conv2d(in_channels=80,
                              out_channels=self.attention_hidden,
                              kernel_size=1))

    def forward(self, *input):
        xa = self.relu(self.bn1a(self.conv1a(input[0])))
        xb = self.relu(self.bn1b(self.conv1b(input[0])))

        if not self.fusion_level:
            x = torch.cat((xa, xb), 1)
            x = self.relu(self.bn2(self.conv2(x)))
        else:
            if self.fusion_level < 0:
                x = torch.cat((xa, xb), 1)
                x = self.relu(self.bn2(self.conv2(x)))
            else:
                x = xa
                x = self.relu(self.bn2(self.conv2(x)))

        if self.fusion_level:
            if 0 < self.fusion_level < 2 or self.fusion_level < -2:
                x = self.maxp(x)
        else:
            x = self.maxp(x)

        if self.fusion_level:
            if 0 < self.fusion_level < 3 or self.fusion_level < -3:
                x = self.relu(self.bn3(self.conv3(x)))
                x = self.maxp(x)
        else:
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.maxp(x)

        if self.fusion_level:
            if 0 < self.fusion_level < 4 or self.fusion_level < -4:
                x = self.relu(self.bn4(self.conv4(x)))
        else:
            x = self.relu(self.bn4(self.conv4(x)))

        if self.fusion_level:
            if 0 < self.fusion_level < 5 or self.fusion_level < -5:
                x = self.relu(self.bn5(self.conv5(x)))
        else:
            x = self.relu(self.bn5(self.conv5(x)))

        if not self.fusion_level or 0 < self.fusion_level < 6:
            attn = None
            for i in range(self.attention_heads):
                q = self.attention_query[i](x)
                k = self.attention_key[i](x)
                v = self.attention_value[i](x)

                attention = F.softmax(torch.mul(q, k), dim=1)
                attention = torch.mul(attention, v)

                if attn is None:
                    attn = attention
                else:
                    attn = torch.cat((attn, attention), 2)
            x = self.gap(self.relu(attn))

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        if self.preMACNN_classifier:
            outpt = self.fc(x.clone())
            return outpt, x
        else:
            x = self.fc(x)
        return x


class MACNN4timesParams(nn.Module):
    def __init__(self, attention_heads=8, attention_hidden=256, num_emo=4):
        super(MACNN4timesParams, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.fusion_level = None

        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1,
                                out_channels=8*2, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1,
                                out_channels=8*2, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16*2,
                               out_channels=32*2, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32*2,
                               out_channels=48*2, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48*2,
                               out_channels=64*2, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64*2,
                               out_channels=80*2, padding=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))

        self.bn1a = nn.BatchNorm2d(8*2)
        self.bn1b = nn.BatchNorm2d(8*2)
        self.bn2 = nn.BatchNorm2d(32*2)
        self.bn3 = nn.BatchNorm2d(48*2)
        self.bn4 = nn.BatchNorm2d(64*2)
        self.bn5 = nn.BatchNorm2d(80*2)

        self.fc = nn.Linear(in_features=self.attention_hidden*2,
                            out_features=num_emo)

        self.dropout = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.attention_heads):
            self.attention_query.append(nn.Conv2d(in_channels=80*2,
                                                  out_channels=self.attention_hidden*2,
                                                  kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80*2,
                                                out_channels=self.attention_hidden*2,
                                                kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80*2,
                                                  out_channels=self.attention_hidden*2,
                                                  kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        if not self.fusion_level:
            x = torch.cat((xa, xb), 1)
        else:
            if self.fusion_level < 0:
                x = torch.cat((xa, xb), 1)
            else:
                x = xa
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        if self.fusion_level:
            if 0 < self.fusion_level < 2 or self.fusion_level < -2:
                x = self.maxp(x)
        else:
            x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x = F.relu(x)
        if self.fusion_level:
            if 0 < self.fusion_level < 3 or self.fusion_level < -3:
                x = self.maxp(x)
        else:
            x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        attn = None
        for i in range(self.attention_heads):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K), dim=1)
            attention = torch.mul(attention, V)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x
