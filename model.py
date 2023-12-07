import torch
import torch.nn as nn


class MACNN(nn.Module):
    # model code adapted from https://github.com/lessonxmk/head_fusion
    def __init__(self, attention_heads=4, attention_hidden=64,
                 fusion_level=None, moe=False):
        super(MACNN, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        self.fusion_level = fusion_level
        self.moe = moe

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
        self.soft = nn.Softmax(dim=1)

        if self.moe:
            self.fc = nn.Linear(in_features=self.attention_hidden,
                                out_features=1)
        else:
            self.fc = nn.Linear(in_features=self.attention_hidden,
                                out_features=4)

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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))

        if not self.fusion_level:
            x = torch.cat((xa, xb), 1)
        else:
            if self.fusion_level < 0:
                x = torch.cat((xa, xb), 1)
            else:
                x = xa
        x = self.relu(self.bn2(self.conv2(x)))

        if self.fusion_level:
            if 0 < self.fusion_level < 2 or self.fusion_level < -2:
                x = self.maxp(x)
        else:
            x = self.maxp(x)
        x = self.relu(self.bn3(self.conv3(x)))

        if self.fusion_level:
            if 0 < self.fusion_level < 3 or self.fusion_level < -3:
                x = self.maxp(x)
        else:
            x = self.maxp(x)
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.relu(self.bn5(self.conv5(x)))

        if not self.fusion_level or 0 < self.fusion_level < 6:
            attn = None
            for i in range(self.attention_heads):
                q = self.attention_query[i](x)
                k = self.attention_key[i](x)
                v = self.attention_value[i](x)

                attention = self.soft(torch.mul(q, k))
                attention = torch.mul(attention, v)

                if attn is None:
                    attn = attention
                else:
                    attn = torch.cat((attn, attention), 2)
            x = self.gap(self.relu(attn))

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x


class MACNNMixtureOfExperts(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64):
        super(MACNNMixtureOfExperts, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 fusion_level=None,
                                 moe=True)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=None,
                               moe=True)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             fusion_level=None,
                             moe=True)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=None,
                               moe=True)

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsForwardFusion1(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, fusion_level=1):
        super(MACNNMixtureOfExpertsForwardFusion1, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 moe=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             moe=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
        x = torch.cat((xa, xb), 1)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsForwardFusion2(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, fusion_level=2):
        super(MACNNMixtureOfExpertsForwardFusion2, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 moe=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             moe=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsForwardFusion3(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, fusion_level=3):
        super(MACNNMixtureOfExpertsForwardFusion3, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 moe=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             moe=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsForwardFusion4(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, fusion_level=4):
        super(MACNNMixtureOfExpertsForwardFusion4, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 moe=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             moe=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        x = self.relu(self.bn4(self.conv4(x)))

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsForwardFusion5(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, fusion_level=5):
        super(MACNNMixtureOfExpertsForwardFusion5, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 moe=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             moe=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
        x = torch.cat((xa, xb), 1)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.maxp(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)

        x = self.relu(self.bn4(self.conv4(x)))

        x = self.relu(self.bn5(self.conv5(x)))

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsForwardFusion6(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, fusion_level=6):
        super(MACNNMixtureOfExpertsForwardFusion6, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 moe=True,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             moe=True,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               moe=True,
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
        self.soft = nn.Softmax(dim=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
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

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)

        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        n = self.mod_neutral(x.clone())
        h = self.mod_happy(x.clone())
        s = self.mod_sad(x.clone())
        a = self.mod_angry(x.clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        return x, n, h, s, a


class MACNNMixtureOfExpertsBackwardFusion7(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, num_moe=4):
        super(MACNNMixtureOfExpertsBackwardFusion7, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()

        self.fc = nn.Linear(in_features=64*num_moe,
                            out_features=4)

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)
        x = self.fc(x)

        return x


class MACNNMixtureOfExpertsBackwardFusion6(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64,
                 fusion_level=-6, num_moe=4):
        super(MACNNMixtureOfExpertsBackwardFusion6, self).__init__()

        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden
        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80 * num_moe,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80 * num_moe,
                          out_channels=self.attention_hidden,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80 * num_moe,
                          out_channels=self.attention_hidden,
                          kernel_size=1))

        self.fc = nn.Linear(in_features=self.attention_hidden,
                            out_features=4)

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMixtureOfExpertsBackwardFusion5(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64,
                 fusion_level=-5, num_moe=4):
        super(MACNNMixtureOfExpertsBackwardFusion5, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level,)

        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64 * num_moe,
                               out_channels=80, padding=1)
        self.bn5 = nn.BatchNorm2d(80)

        for mod in [self.mod_neutral, self.mod_happy, self.mod_sad,
                    self.mod_angry]:
            mod.conv5 = nn.Identity()
            mod.bn5 = nn.Identity()
            mod.fc = nn.Identity()

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMixtureOfExpertsBackwardFusion4(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, num_moe=4,
                 fusion_level=-4,):
        super(MACNNMixtureOfExpertsBackwardFusion4, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)

        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48 * num_moe,
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
        self.soft = nn.Softmax(dim=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMixtureOfExpertsBackwardFusion3(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, num_moe=4,
                 fusion_level=-3):
        super(MACNNMixtureOfExpertsBackwardFusion3, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)

        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32 * num_moe,
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
        self.soft = nn.Softmax(dim=1)

        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxp(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))

        attn = None
        for i in range(self.attention_heads):
            q = self.attention_query[i](x)
            k = self.attention_key[i](x)
            v = self.attention_value[i](x)

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNNMixtureOfExpertsBackwardFusion2(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64, num_moe=4,
                 fusion_level=-2):
        super(MACNNMixtureOfExpertsBackwardFusion2, self).__init__()
        self.mod_neutral = MACNN(attention_heads=attention_heads,
                                 attention_hidden=attention_hidden,
                                 fusion_level=fusion_level)
        self.mod_happy = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)
        self.mod_sad = MACNN(attention_heads=attention_heads,
                             attention_hidden=attention_hidden,
                             fusion_level=fusion_level)
        self.mod_angry = MACNN(attention_heads=attention_heads,
                               attention_hidden=attention_hidden,
                               fusion_level=fusion_level)

        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16 * num_moe,
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
        self.soft = nn.Softmax(dim=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.gap = nn.AdaptiveAvgPool2d(1)

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

    def forward(self, *inpt):
        n = self.mod_neutral(inpt[0].clone())
        h = self.mod_happy(inpt[0].clone())
        s = self.mod_sad(inpt[0].clone())
        a = self.mod_angry(inpt[0].clone())

        x = torch.cat((n.clone(), h.clone(), s.clone(), a.clone()), dim=1)
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

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = self.gap(self.relu(attn))

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)

        return x


class MACNN4timesParams(nn.Module):
    def __init__(self, attention_heads=4, attention_hidden=64):
        super(MACNN4timesParams, self).__init__()
        self.attention_heads = attention_heads
        self.attention_hidden = attention_hidden

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

        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=1)

        self.fc = nn.Linear(in_features=self.attention_hidden*2,
                            out_features=4)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.attention_heads):
            self.attention_query.append(
                nn.Conv2d(in_channels=80*2,
                          out_channels=self.attention_hidden*2,
                          kernel_size=1))
            self.attention_key.append(
                nn.Conv2d(in_channels=80*2,
                          out_channels=self.attention_hidden*2,
                          kernel_size=1))
            self.attention_value.append(
                nn.Conv2d(in_channels=80*2,
                          out_channels=self.attention_hidden*2,
                          kernel_size=1))

    def forward(self, *inpt):
        xa = self.relu(self.bn1a(self.conv1a(inpt[0])))
        xb = self.relu(self.bn1b(self.conv1b(inpt[0])))
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

            attention = self.soft(torch.mul(q, k))
            attention = torch.mul(attention, v)

            if attn is None:
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = self.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x
