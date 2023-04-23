import torch.nn as nn
from torchvision.ops import stochastic_depth
from functools import partial

class SE(nn.Module):
    def __init__(self,
                 in_planes,
                 expand_planes,
                 se_ratio = 0.25):
        super(SE, self).__init__()

        squeeze_planes = int(in_planes * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_planes, squeeze_planes, 1)
        self.silu = nn.SiLU()
        self.conv_expand = nn.Conv2d(squeeze_planes, expand_planes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x.mean((2, 3), keepdim=True)
        out = self.conv_reduce(out)
        out = self.silu(out)
        out = self.conv_expand(out)
        out = self.sigmoid(out)
        return out * x

class Conv_BN_Act(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size = 3,
                 stride = 1,
                 groups = 1,
                 bn_layer = partial(nn.BatchNorm2d, momentum=0.01),
                 activation_layer = nn.SiLU):
        super(Conv_BN_Act, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(kernel_size-1)//2,
                              groups=groups,
                              bias=False)

        self.bn = bn_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        return out

class MBConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 stride,
                 drop_rate):
        super(MBConv, self).__init__()

        self.skip_connection = (stride == 1 and in_planes == out_planes)

        expanded_plances = in_planes * expand_ratio

        self.expand_conv = Conv_BN_Act(in_planes, expanded_plances, kernel_size=1)
        self.dw_conv = Conv_BN_Act(expanded_plances, expanded_plances, stride=stride, groups=expanded_plances)
        self.se = SE(in_planes, expanded_plances, 0.25)
        self.project_conv = Conv_BN_Act(expanded_plances, out_planes, kernel_size=1, activation_layer=nn.Identity) 

        if self.skip_connection:
            self.dropout = partial(stochastic_depth, p=drop_rate, model = 'batch')

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.dw_conv(out)
        out = self.se(out)
        out = self.project_conv(out)

        if self.skip_connection:
            out = self.dropout(out)
            out += x

        return out
    
class FusedMBConv(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 expand_ratio,
                 stride,
                 drop_rate):
        super(FusedMBConv, self).__init__()

        self.skip_connection = (stride == 1 and in_planes == out_planes)
        self.expansion = expand_ratio != 1

        expanded_plances = in_planes * expand_ratio

        if self.expansion:
            self.expand_conv = Conv_BN_Act(in_planes, expanded_plances, stride=stride)
            self.project_conv = Conv_BN_Act(expanded_plances, out_planes, kernel_size=1, activation_layer=nn.Identity)
        else:
            self.project_conv = Conv_BN_Act(in_planes, out_planes, stride=stride)

        if self.skip_connection:
            self.dropout = partial(stochastic_depth, p=drop_rate, model = 'batch')

    def forward(self, x):
        if self.expansion:
            out = self.expand_conv(x)
            out = self.project_conv(out)
        else:
            out = self.project_conv(out)

        if self.skip_connection:
            out = self.dropout(out)
            out += x

        return out
    
class EfficientNetV2(nn.Module):

    cfgs = [[1,1,1,24,2],[1,4,2,48,4],[1,4,2,64,4],[0,4,2,128,6],[0,6,1,160,9],[0,6,2,256,15]]

    def __init__(self, 
                 num_features = 1280,
                 num_classes = 1000,
                 dropout_rate = 0.2,
                 survival_probability = 0.8):
        super(EfficientNetV2, self).__init__()

        self.stage0 = Conv_BN_Act(3, 24, stride=2)

        self.stage1_6 = self._make_stages(24, survival_probability)

        self.stage7 = nn.Sequential(
            Conv_BN_Act(256, num_features, kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, num_classes)
        )

        self.apply(self._init_weights)
    
    def _make_stages(self, in_planes, survival_probability):
        operator = [MBConv, FusedMBConv]
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            for i in range(cfg[4]):
                stages.append(operator[cfg[0]](input_c=in_planes,
                                               out_c=cfg[3],
                                               expand_ratio=cfg[1],
                                               stride=cfg[2] if i == 0 else 1,
                                               drop_rate=(1-survival_probability) * layer_id / 40))
                layer_id += 1
    
    def _init_weights(self, models):
        if isinstance(models, nn.Conv2d):
            nn.init.kaiming_normal_(models.weight, mode="fan_out")
            if models.bias is not None:
                nn.init.zeros_(models.bias)
        elif isinstance(models, nn.BatchNorm2d):
            nn.init.ones_(models.weight)
            nn.init.zeros_(models.bias)
        elif isinstance(models, nn.Linear):
            nn.init.normal_(models.weight, 0, 0.01)
            nn.init.zeros_(models.bias)


    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1_6(out)
        out = self.stage7(out)

        return out