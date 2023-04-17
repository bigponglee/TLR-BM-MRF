import torch.nn as nn
from configs import TLR_BM_configs as cfg
from model.base import down_block, up_block, conv3x3, BasicBlock


class Encoder(nn.Module):
    '''
    Encoder
    编码器
    '''

    def __init__(self, inchannel=cfg.input_dim, outchannel=cfg.para_maps, channels_list=[512, 256, 128, 64, 32, 32, 16, 8, 4]):
        super(Encoder, self).__init__()
        self.head = nn.Sequential(nn.Conv2d(in_channels=inchannel, out_channels=channels_list[0], kernel_size=5, stride=1, padding=2),
                                  nn.BatchNorm2d(channels_list[0]),
                                  nn.ELU(inplace=True),
                                  )  # [batch_size, channels_list[0], 220, 220]
        # down 220-110-55-26-13 feature depth 512-256-128-64-32
        self.down1 = down_block(pool_size=2, pool_stride=2, inchannel=channels_list[0], outchannel=channels_list[1],
                                skip_channel=channels_list[7], conv_group=1, basic_1_num=1)  # [batch_size, channels_list[1], 110, 110]
        self.down2 = down_block(pool_size=2, pool_stride=2, inchannel=channels_list[1], outchannel=channels_list[2],
                                skip_channel=channels_list[6], conv_group=1, basic_1_num=1)  # [batch_size, channels_list[2], 55, 55]
        self.down3 = down_block(pool_size=5, pool_stride=2, inchannel=channels_list[2], outchannel=channels_list[3],
                                skip_channel=channels_list[5], conv_group=1, basic_1_num=1)  # [batch_size, channels_list[3], 26, 26]
        self.down4 = down_block(pool_size=2, pool_stride=2, inchannel=channels_list[3], outchannel=channels_list[4],
                                skip_channel=channels_list[4], conv_group=1, basic_1_num=1)  # [batch_size, channels_list[4], 13, 13]
        self.basic_block_1 = nn.Sequential(*[
            BasicBlock(inplanes=channels_list[4], planes=channels_list[4], stride=1, groups=1, downsample=None) for _ in range(2)])
        # up 13-26-55-110-220 feature depth 32-64-128-256-512
        self.up1 = up_block(up_size=(26, 26), inchannel=2*channels_list[4], outchannel=channels_list[5],
                            conv_group=1, basic_1_num=1)  # [batch_size, channels_list[3], 26, 26]
        self.up2 = up_block(up_size=(55, 55), inchannel=2*channels_list[5], outchannel=channels_list[6],
                            conv_group=1, basic_1_num=1)  # [batch_size, channels_list[2], 55, 55]
        self.up3 = up_block(up_size=(110, 110), inchannel=2*channels_list[6], outchannel=channels_list[7],
                            conv_group=1, basic_1_num=1)  # [batch_size, channels_list[1], 110, 110]
        self.up4 = up_block(up_size=(220, 220), inchannel=2*channels_list[7], outchannel=channels_list[8],
                            conv_group=1, basic_1_num=1)  # [batch_size, channels_list[0], 220, 220]
        self.tail = nn.Sequential(conv3x3(in_planes=channels_list[8], out_planes=outchannel, stride=1, groups=1),
                                  nn.Sigmoid(),
                                  )  # [batch_size, outchannel, 220, 220]

    def forward(self, x):
        x = self.head(x)  # [batch_size, channels_list[0], 220, 220]
        # x: [batch_size, channels_list[1], 110, 110] skip_1: [batch_size, channels_list[7], 110, 110]
        x, skip_1 = self.down1(x)
        # x: [batch_size, channels_list[2], 55, 55] skip_2: [batch_size, channels_list[6], 55, 55]
        x, skip_2 = self.down2(x)
        # x: [batch_size, channels_list[3], 26, 26] skip_3: [batch_size, channels_list[5], 26, 26]
        x, skip_3 = self.down3(x)
        # x: [batch_size, channels_list[4], 13, 13] skip_4: [batch_size, channels_list[4], 13, 13]
        x, skip_4 = self.down4(x)
        x = self.basic_block_1(x)  # x: [batch_size, channels_list[4], 13, 13]
        x = self.up1(x, skip_4)  # x: [batch_size, channels_list[5], 26, 26]
        x = self.up2(x, skip_3)  # x: [batch_size, channels_list[6], 55, 55]
        x = self.up3(x, skip_2)  # x: [batch_size, channels_list[7], 110, 110]
        x = self.up4(x, skip_1)  # x: [batch_size, channels_list[8], 220, 220]
        x = self.tail(x)  # x: [batch_size, outchannel, 220, 220]
        return x
