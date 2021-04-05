from model import common
import time
import torch
import torch.nn as nn
import math

def make_model(args, parent=False):
    return RSAN(args)

class AdaptiveConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AdaptiveConv, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.paddings = padding
        self.dilations = dilation
        self.groups = groups
        self.adaptiveweight = nn.Sequential(*[nn.Linear(1+in_channels, out_channels//4), nn.ReLU(), nn.Linear(out_channels//4, out_channels), nn.Sigmoid()])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x, s):
        y = self.avg_pool(x).view(x.shape[0],x.shape[1])
        scaleprior = y.new_full((x.shape[0], 1), s)
        prior = torch.cat([y, scaleprior], dim=1)
        weight = self.adaptiveweight(prior)
        weight = weight.view(x.shape[0], self.weight.shape[0], 1, 1)
        res = []
        for i in range(x.shape[0]):
            newweight = self.weight * weight[i].view(self.weight.shape[0],1,1,1)
            res.append(torch.nn.functional.conv2d(x[i:i+1], newweight, self.bias, self.stride, self.paddings, self.dilations, self.groups))
        result = torch.cat(res, dim=0)
        return result

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=( kSize -1 ) //2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c* G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = AdaptiveConv(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x, s):
        out = x
        out = self.convs(x)
        return self.LFF(out, s) + x


class Pos2Weight(nn.Module):
    def __init__(self,inC, kernel_size=3, outC=3):
        super(Pos2Weight,self).__init__()
        self.inC = inC
        self.kernel_size=kernel_size
        self.outC = outC
        self.meta_block=nn.Sequential(
            nn.Linear(6,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,self.kernel_size*self.kernel_size*self.inC*self.outC)
        )
    def forward(self,x):

        output = self.meta_block(x)
        return output

class RSAN(nn.Module):
    def __init__(self, args):
        super(RSAN, self).__init__()
        G0 = args.G0
        kSize = args.RDNkSize
        self.scale = 1
        self.args = args
        self.scale_idx = 0
        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )
        # Global Feature Fusion

        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            AdaptiveConv(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        ## position to weight
        self.P2W = Pos2Weight(inC=G0)
        self.P2W1 = Pos2Weight(inC=3)
    def repeat_x(self,x):
        scale_int = math.ceil(self.scale)
        N,C,H,W = x.size()
        x = x.view(N,C,H,1,W,1)

        x = torch.cat([x]*scale_int,3)
        x = torch.cat([x]*scale_int,5).permute(0,3,5,1,2,4)

        return x.contiguous().view(-1, C, H, W)

    def forward(self, x, pos_mat=None):
        self.scalein = 1/self.scale
        x = self.sub_mean(x)
        up_img = self.repeat_x(x)    
        upimg = nn.functional.unfold(up_img, 3,padding=1)
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x, self.scalein)
            RDBs_out.append(x)
        out = torch.cat(RDBs_out, 1)
        out = self.GFF[0](out)
        out = self.GFF[1](out, self.scalein)
        x = out
        x += f__1
        local_weight = self.P2W(pos_mat.view(pos_mat.size(1),-1))   ###   (outH*outW, outC*inC*kernel_size*kernel_size)
        local_weight1 = self.P2W1(pos_mat.view(pos_mat.size(1),-1))
        #print(d2)
        up_x = self.repeat_x(x)     ### the output is (N*r*r,inC,inH,inW)

        cols = nn.functional.unfold(up_x, 3,padding=1)
        scale_int = math.ceil(self.scale)

        cols = cols.contiguous().view(cols.size(0)//(scale_int**2),scale_int**2, cols.size(1), cols.size(2), 1).permute(0,1, 3, 4, 2).contiguous()
        upimg = upimg.contiguous().view(upimg.size(0)//(scale_int**2),scale_int**2, upimg.size(1), upimg.size(2), 1).permute(0,1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight = local_weight.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)
        local_weight1 = local_weight1.contiguous().view(x.size(2),scale_int, x.size(3),scale_int,-1,3).permute(1,3,0,2,4,5).contiguous()
        local_weight1 = local_weight1.contiguous().view(scale_int**2, x.size(2)*x.size(3),-1, 3)
        out = torch.matmul(cols,local_weight).permute(0,1,4,2,3)
        out = out.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out = out.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out1 = torch.matmul(upimg,local_weight1).permute(0,1,4,2,3)
        out1 = out1.contiguous().view(x.size(0),scale_int,scale_int,3,x.size(2),x.size(3)).permute(0,3,4,1,5,2)
        out1 = out1.contiguous().view(x.size(0),3, scale_int*x.size(2),scale_int*x.size(3))
        out = out + out1
        out = self.add_mean(out)

        return out

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        if self.training:
            self.scale = self.args.scale[scale_idx]
        else:
            self.scale = self.args.testscale[scale_idx]


