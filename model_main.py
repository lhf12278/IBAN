import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
        self.l = nn.Conv2d(channel, channel,1)

    def forward(self,t_o, t, x):
        t_att = self.fc(t)
        x_t   = x*t_att + x

        x, t_o =normalize(x), normalize(t_o)
        co_sim = torch.cosine_similarity(x, t_o, dim=1)
        return x_t.unsqueeze(dim=1), co_sim

class Sim_r(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Sim_r, self).__init__()
    def forward(self, t_r, x):
        x, t_r =normalize(x),normalize(t_r)
        co_sim = torch.cosine_similarity(x, t_r, dim=1)
        return co_sim

class temporal_feat_learning(nn.Module):
    def __init__(self,  ):
        super(temporal_feat_learning, self).__init__()
        dim = 2048
        self.se_1 = SELayer(2048)
        self.se_2 = SELayer(2048)
        self.se_3 = SELayer(2048)
        self.se_4 = SELayer(2048)
        self.se_5 = SELayer(2048)
        self.se_6 = SELayer(2048)

        self.s_1 = Sim_r(2048)
        self.s_2 = Sim_r(2048)
        self.s_3 = Sim_r(2048)
        self.s_4 = Sim_r(2048)
        self.s_5 = Sim_r(2048)
        self.s_6 = Sim_r(2048)

        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.d = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.f = nn.Linear(dim, dim)

    def forward(self, t_o, t_r, x):#(t, x_l, x_h)
        t1_o = self.a(t_o)+x[0]
        t2_o = self.b(t_o)+x[1]
        t3_o = self.c(t_o)+x[2]
        t4_o = self.d(t_o)+x[3]
        t5_o = self.e(t_o)+x[4]
        t6_o = self.f(t_o)+x[5]

        f1,sim1_o = self.se_1(t_o, t1_o/2, x[0])
        f2,sim2_o = self.se_2(t_o, t2_o/2, x[1])
        f3,sim3_o = self.se_3(t_o, t3_o/2, x[2])
        f4,sim4_o = self.se_4(t_o, t4_o/2, x[3])
        f5,sim5_o = self.se_5(t_o, t5_o/2, x[4])
        f6,sim6_o = self.se_6(t_o, t6_o/2, x[5])

        sim1 = ((self.s_1(t_r, x[0]) + sim1_o)/2).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,1,2048)
        sim2 = ((self.s_2(t_r, x[1]) + sim2_o)/2).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,1,2048)
        sim3 = ((self.s_3(t_r, x[2]) + sim3_o)/2).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,1,2048)
        sim4 = ((self.s_4(t_r, x[3]) + sim4_o)/2).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,1,2048)
        sim5 = ((self.s_5(t_r, x[4]) + sim5_o)/2).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,1,2048)
        sim6 = ((self.s_6(t_r, x[5]) + sim6_o)/2).unsqueeze(dim=1).unsqueeze(dim=2).repeat(1,1,2048)


        f = torch.cat((torch.mul(sim1,f1)+f1, torch.mul(sim2,f2)+f2, torch.mul(sim3,f3)+f3, torch.mul(sim4,f4)+f4, torch.mul(sim5,f5)+f5, torch.mul(sim6,f6)+f6), dim=1)#拼接
        f = f.mean(dim=1)#平均
        return f



class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv3d
        max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        bn = nn.BatchNorm3d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential( conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                                    bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x,x2):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x2).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class Dual_Cross_Atten(nn.Module):
    def __init__(self, in_channels):
        super(Dual_Cross_Atten, self).__init__()
        self.cross_atten = NonLocalBlockND(in_channels)
    def forward(self, x1, x2):
        B, c, h, w = x1.size() #(96,512,36,18)
        seq_len = 6
        x1 = x1.view(int(B//seq_len), c, seq_len, h, w)
        x2 = x2.view(int(B//seq_len), c, seq_len, h, w)

        x1 = self.cross_atten(x1, x2)

        x1 = x1.view(int(B),c, h, w)
        x2 = x2.view(int(B),c, h, w)
        return x1, x2



class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x
class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        self.layer4 = copy.deepcopy(self.base.layer4)

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x
class base_resnet_fir(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_fir, self).__init__()
        model_base = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        return x
class base_resnet_sec(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet_sec, self).__init__()
        model_base = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
    def forward(self, x):
        x = self.base.layer4(x)
        return x

class chonggou_fir(nn.Module):
    def __init__(self):
        super(chonggou_fir,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1,bias=False)#3——1
        )
    def forward(self,x):
        x=self.conv(x)
        return x
class chonggou_sec(nn.Module):
    def __init__(self):
        super(chonggou_sec,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0,bias=False)#3——1
        )
    def forward(self,x):
        x=self.conv(x)
        return x


def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)
def conv1x1(conv,x):
    x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    x = conv(x)
    x = x.squeeze()
    return x


class embed_net(nn.Module):
    def __init__(self,  low_dim,  class_num, drop=0.2, part = 3, alpha=0.2, nheads=4, arch='resnet50', wpa = False):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.thermal_module_f = thermal_module(arch=arch)
        self.visible_module_f = visible_module(arch=arch)

        self.base_resnet = base_resnet(arch=arch)
        self.base_resnet_fir = base_resnet_fir(arch=arch)
        self.base_resnet_sec = base_resnet_sec(arch=arch)
        self.base_resnet_fir_f = base_resnet_fir(arch=arch)
        self.base_resnet_sec_f = base_resnet_sec(arch=arch)

        self.chonggou_v2t_fir = chonggou_fir()
        self.chonggou_t2v_fir = chonggou_fir()
        self.chonggou_v2t_sec = chonggou_sec()
        self.chonggou_t2v_sec = chonggou_sec()

        self.cross_atten_fir = Dual_Cross_Atten(1024)
        self.cross_atten_sec = Dual_Cross_Atten(2048)


        pool_dim = 2048
        self.dropout = drop
        self.part = part
        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_f = nn.BatchNorm1d(pool_dim)
        self.bottleneck_f.bias.requires_grad_(False)
        self.bottleneck_f.apply(weights_init_kaiming)

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_f = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_mean = nn.AdaptiveAvgPool2d((2048, 1))
        self.avgpool_mean_f = nn.AdaptiveAvgPool2d((2048, 1))

        self.lstm_o = nn.LSTM(2048, 2048, 2)
        self.lstm_r = nn.LSTM(2048, 2048, 2)
        self.temporal_feat_learning = temporal_feat_learning()

    def forward(self, x1, x2, f1, f2, modal=0, seq_len = 6):
        b, c, h, w = x1.size()
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w) #(24,3,288,144)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)
        f1 = f1.view(int(b * seq_len), int(c / seq_len), h, w) #!
        f2 = f2.view(int(b * seq_len), int(c / seq_len), h, w)

        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

            f1 = self.visible_module_f(f1)
            f2 = self.thermal_module_f(f2)
            f = torch.cat((f1, f2), 0)

        elif modal == 1:
            x = self.visible_module(x1)
            f = self.visible_module_f(f1)
        elif modal == 2:
            x = self.thermal_module(x2)
            f = self.thermal_module_f(f2)

        x_fir = self.base_resnet_fir(x) #(48,1024,18,9)
        f_fir = self.base_resnet_fir_f(f)
        x_fir_att, f_fir = self.cross_atten_fir(x_fir, f_fir)

        x = self.base_resnet_sec(x_fir_att) #(8,2048,18,9)
        f = self.base_resnet_sec_f(f_fir)
        x, f = self.cross_atten_sec(x, f)

#----------------------------------------------------------------------------------------------------------------------#
        x_o = self.avgpool(x).squeeze() #(2*b*s, 2048)
        x_o = x_o.view(x_o.size(0)//seq_len, seq_len, -1).permute(1, 0, 2)

        x_r = []
        for i in range(seq_len-1, -1, -1):
            single = x_o[i].unsqueeze(0)
            x_r.append(single)
        x_r = torch.cat(x_r, dim=0)
#----------------------------------------------------------------------------------------------------------------------#

        h0 = torch.zeros(2, x_o.shape[1], x_o.shape[2]).cuda()
        c0 = torch.zeros(2, x_o.shape[1], x_o.shape[2]).cuda()

        if self.training:
            self.lstm_o.flatten_parameters()
            self.lstm_r.flatten_parameters()

        out_o, (hn, cn) = self.lstm_o(x_o, (h0, c0))
        out_r, (hn, cn) = self.lstm_r(x_r, (h0, c0))
        t_o = out_o[-1]
        t_r = out_r[-1]

        x_t = self.temporal_feat_learning(t_o, t_r, x_o)

#----------------------------------------------------------------------------------------------------------------------#
        x_feat = self.bottleneck(x_t)

        if self.training:
            B = f.size(0)//2

            f_v2t_fir = self.chonggou_v2t_fir(f_fir[:B])#f:浮雕，b = x1.size(0)
            f_t2v_fir = self.chonggou_t2v_fir(f_fir[B:])
            f_v2t_sec = self.chonggou_v2t_sec(f[:B])#
            f_t2v_sec = self.chonggou_t2v_sec(f[B:])

            f_pool = self.avgpool_f(f).squeeze() #(48,2048)
            f_pool = f_pool.view(int(f_pool.size(0)/seq_len), seq_len, f_pool.size(1))#(8,6,2048)
            f_pool = f_pool.permute(0, 2, 1)#(8,2048,6)
            f_pool = self.avgpool_mean_f(f_pool).squeeze()#(8,2048)
            f_feat = self.bottleneck_f(f_pool)

            return x_t, self.classifier(x_feat), f_pool, self.classifier(f_feat), f_v2t_fir, f_t2v_fir, f_v2t_sec, f_t2v_sec, f_fir, f#改成一个分类器
            #池化，池化+降维， 池化，池化+降维， 重构v，重构i，原始f
        else:
            return self.l2norm(x_feat)