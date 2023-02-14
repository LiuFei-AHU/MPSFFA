# baseline
import torch
from torch import nn
import numpy as np
import os


class ChannelAttention(nn.Module):
    def __init__(self, c_in, m_in):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(c_in, m_in)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(m_in, c_in)
        self.flatten = nn.Flatten()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        avg_x = torch.reshape(avg_x, [avg_x.shape[0], -1])
        max_x = torch.reshape(max_x, [max_x.shape[0], -1])
        avg_out = self.fc2(self.relu1(self.fc1(avg_x)))
        max_out = self.fc2(self.relu1(self.fc1(max_x)))
        out = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return self.sigmoid(out)
        # return out


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], 1)
        out = self.sigmoid(self.conv(out))
        return out * x


class CDAF_Block(nn.Module):
    def __init__(self, in_channel):
        super(CDAF_Block, self).__init__()
        # cross dimension
        self.conv_bcxyz = nn.Sequential(nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(in_channel * 2), nn.ReLU(),
                                        nn.Conv3d(in_channel * 2, in_channel, 1), nn.BatchNorm3d(in_channel), nn.ReLU())
        self.conv_bcyzx = nn.Sequential(nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(in_channel * 2), nn.ReLU(),
                                        nn.Conv3d(in_channel * 2, in_channel, 1), nn.BatchNorm3d(in_channel), nn.ReLU())
        self.conv_bcxzy = nn.Sequential(nn.Conv3d(in_channel, in_channel * 2, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(in_channel * 2), nn.ReLU(),
                                        nn.Conv3d(in_channel * 2, in_channel, 1), nn.BatchNorm3d(in_channel), nn.ReLU())
        self.reduce_channel = nn.Sequential(nn.Conv3d(in_channel * 3, in_channel, 1), nn.BatchNorm3d(in_channel),
                                            nn.ReLU())
        self.sp_xyz = SpatialAttention()
        self.sp_yzx = SpatialAttention()
        self.sp_xzy = SpatialAttention()

    def forward(self, x):
        x1 = self.sp_xyz(self.conv_bcxyz(x))
        x2 = self.sp_yzx(self.conv_bcyzx(x.transpose([0, 1, 3, 4, 2]))).transpose([0, 1, 4, 2, 3])
        x3 = self.sp_xzy(self.conv_bcxzy(x.transpose([0, 1, 2, 4, 3]))).transpose([0, 1, 2, 4, 3])
        block_all = torch.cat([x1, x2, x3], 1)
        return self.reduce_channel(block_all)


class MBAF_Block(nn.Module):
    def __init__(self, in_channel, branch=(1, 3, 5)):
        super(MBAF_Block, self).__init__()

        branches, local_att = [], []

        for b in branch:
            branches.append(nn.Sequential(
                nn.Conv3d(in_channel, in_channel * 2, kernel_size=b, padding=(b - 1) // 2),
                nn.BatchNorm3d(in_channel * 2),
                nn.ReLU(),
                nn.Conv3d(in_channel * 2, in_channel, kernel_size=1),
                nn.BatchNorm3d(in_channel),
                nn.ReLU()
            ))
            local_att.append(nn.Sequential(
                nn.Conv3d(in_channel, in_channel, kernel_size=1),
                nn.Sigmoid()
            ))
        self.branches = nn.ModuleList(branches)
        self.local_att = nn.ModuleList(local_att)
        self.reduce_channel = nn.Sequential(nn.Conv3d(in_channel * len(branch), in_channel, 1),
                                            nn.BatchNorm3d(in_channel), nn.ReLU())

    def forward(self, x):
        block_out = []
        for i in range(len(self.branches)):
            block = self.branches[i]
            local = self.local_att[i]
            b = block(x)
            y = b * local(b)
            block_out.append(y)
        block_all = torch.cat(block_out, 1)
        return self.reduce_channel(block_all)


class AttBlock(nn.Module):
    def __init__(self, in_channel, branch):
        super(AttBlock, self).__init__()

        self.att = nn.Sequential(
            CDAF_Block(in_channel),
            MBAF_Block(in_channel, branch)
        )

    def forward(self, x):
        return self.att(x)


class ConvLayer(nn.Module):
    def __init__(self, ch_in, ch_out, f=3, s=1):
        super(ConvLayer, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=f, stride=s, padding=(f - 1) // 2),
            nn.BatchNorm3d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, att_net=None):
        super(BasicBlock, self).__init__()

        self.conv = nn.Sequential(
            ConvLayer(ch_in, ch_out * 2, 1, 1),
            ConvLayer(ch_out * 2, ch_out, 3, 1)
        )
        self.att_net = att_net

    def forward(self, x):
        y = self.conv(x) + x
        if self.att_net is not None:
            return self.att_net(y) + x  # + y
        return y


class DownSample(nn.Module):
    def __init__(self, ch_in, ch_out, f=3, s=2):
        super(DownSample, self).__init__()

        self.down_sample = ConvLayer(ch_in, ch_out, f, s)
        self.ch_out = ch_out

    def forward(self, x):
        return self.down_sample(x)


class LayerWarp(nn.Module):
    def __init__(self, ch_in, ch_out, count, att=False, layer_no=0, is_print=False):
        super(LayerWarp, self).__init__()

        self.att = att
        self.layer_no = layer_no
        self.is_print = is_print
        blocks = []
        for i in range(1, count):
            att_block = None
            if self.att:
                if i % 2 == 0:
                    if i * 2 <= count:
                        branch = (1, 3, 5, 7)
                    else:
                        branch = (1, 3)
                    att_block = AttBlock(ch_in, branch)
            blocks.append(BasicBlock(ch_in, ch_in, att_block))
        self.blocks = nn.ModuleList(blocks)
        self.down_sample = DownSample(ch_in, ch_out)

    def forward(self, x, sid=None, path=None):
        for i, block in enumerate(self.blocks):
            x = block(x)
        if self.is_print:
            np.save(os.path.join(path, '{}.npy'.format(self.layer_no)), x.cpu().numpy())
        return self.down_sample(x)


class LayerDownSample(nn.Module):
    def __init__(self, ch_in, ch_out, att=None):
        super(LayerDownSample, self).__init__()
        self.blocks = nn.Sequential(BasicBlock(ch_in, ch_in, att), DownSample(ch_in, ch_out))

    def forward(self, x):
        return self.blocks(x)


class BackBone(nn.Module):
    def __init__(self, is_print=False):
        super(BackBone, self).__init__()

        self.stages = [(4, False, 64, 128), (6, True, 128, 256), (8, True, 256, 512), (4, False, 512, 1024)]
        self.first_layer = nn.Sequential(ConvLayer(1, 32, 4, 2), ConvLayer(32, 64))

        layers, layers_m = [], []
        for i, (count, att, c_in, c_out) in enumerate(self.stages):
            layers.append(LayerWarp(c_in, c_out, count, att, i, is_print))
            if 0 < i < len(self.stages):
                layers_m.append(LayerDownSample(c_in, c_out, CDAF_Block(c_in)))
        self.layers = nn.ModuleList(layers)
        self.layers_m = nn.ModuleList(layers_m)

        self.reduce_ch = nn.Sequential(ConvLayer(1024 * 2, 512, 3), ConvLayer(512, 128, 3), ConvLayer(128, 64, 1))

    def forward(self, x, sid=None, spath=None):
        y = self.first_layer(x)
        layer_out = None
        for i, layer in enumerate(self.layers):
            y = layer(y, sid, spath)
            if layer_out is None:
                layer_out = y
            if i < len(self.layers_m):
                layer_out = self.layers_m[i](layer_out)
        out = self.reduce_ch(torch.cat([layer_out, y], 1))
        return out


class BlockScore(nn.Module):
    def __init__(self, block_num, token_size):
        super(BlockScore, self).__init__()
        self.feature_fc = nn.Sequential(nn.Linear(token_size, 1), nn.Sigmoid())
        self.global_att = ChannelAttention(block_num, block_num * 2)
        self.flatten = torch.nn.Flatten(2)
        self.softmax = nn.Softmax()

    def forward(self, x, m):
        _shape = x.shape
        y = self.flatten(x)
        fea_score = []
        for i in range(64):
            s = self.feature_fc(y[:, i])
            s = torch.sqrt(torch.square(torch.subtract(s, m)))
            fea_score.append(1 - s)
        b_scores = torch.cat(fea_score, 1).unsqueeze(-1)
        att = self.flatten(self.global_att(x))
        scores = (att + b_scores) / 2
        out = torch.reshape(torch.multiply(y, scores), _shape)
        return out


class MPSFFA(nn.Module):
    def __init__(self, is_print=False):
        super(MPSFFA, self).__init__()
        self.backbone = BackBone(is_print)

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 5 * 4, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, 2))

        self.fea_weighted = BlockScore(64, 4 * 5 * 4)

    def forward(self, x, m, sid=None, spath=None):
        y = self.backbone(x, sid, spath)
        out = y
        y = self.fea_weighted(y, m)
        y = self.fc(torch.reshape(y, [y.shape[0], -1]))
        return y, out


