import torch.nn as nn
from configs import TLR_BM_configs as cfg
import torch


def real2complex(real_tensor, T=cfg.data_shape[2]):
    '''
    Args:
    real_tensor: [batch, 2*T , Kx, Ky]
    Returns:
    complex_tensor: [T,batch,Kx,Ky]
    '''
    complex_tensor = real_tensor[:, :T, :, :] + \
        1j*real_tensor[:, T:, :, :]
    return complex_tensor.permute(1, 0, 2, 3)


def complex2real(complex_tensor):
    '''
    Args:
    complex_tensor: [T,batch,Kx,Ky]
    Returns:
    real_tensor: [batch, 2*T , Kx, Ky]
    '''
    real_tensor = torch.cat((complex_tensor.real, complex_tensor.imag), dim=0)
    return real_tensor.permute(1, 0, 2, 3)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        downsample=None,
    ) -> None:
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes, stride=1, groups=groups)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Temporal_Attention(nn.Module):
    '''
    channel attention module
    [batch_size, outchannel, kx, ky] ----> [batch_size, outchannel, kx, ky]
    '''

    def __init__(self, inchannel=512) -> None:
        super(Temporal_Attention, self).__init__()
        self.conv_q = conv1x1(inchannel, inchannel)
        self.conv_k = conv1x1(inchannel, inchannel)
        self.conv_v = conv1x1(inchannel, inchannel)
        self.sig = nn.Softmax(dim=2)
        self.bn = nn.BatchNorm2d(inchannel)

    def forward(self, x):
        '''
        x: [batch_size, outchannel, kx, ky] ----> [batch_size, outchannel, kx, ky]
        '''

        # [batch_size, outchannel, 2]
        # [batch_size, outchannel, 1]
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)
        # [batch_size, outchannel, kx*ky]
        q = torch.reshape(q, (q.shape[0], q.shape[1], -1))
        # [batch_size, outchannel, kx*ky]
        k = torch.reshape(k, (k.shape[0], k.shape[1], -1))
        # [batch_size, outchannel, kx*ky]
        v = torch.reshape(v, (v.shape[0], v.shape[1], -1))
        # [batch_size, outchannel, outchannel]
        attention_map = torch.matmul(q, k.transpose(1, 2))
        attention_map = self.sig(attention_map)
        # [batch_size, outchannel, kx*ky]
        x_att = torch.matmul(attention_map, v)
        x_att = torch.reshape(x_att, x.shape)
        x = x + x_att
        x = self.bn(x)

        return x


class basic_module(nn.Module):

    def __init__(self, inchannel=512, outchannel=256, groups=2) -> None:
        super(basic_module, self).__init__()
        self.res_layer1 = BasicBlock(inplanes=inchannel, planes=outchannel, stride=1, groups=groups, downsample=nn.Sequential(
            conv3x3(in_planes=inchannel, out_planes=outchannel,
                    stride=1, groups=groups),
            nn.BatchNorm2d(outchannel),
        ))
        self.res_layer2 = BasicBlock(
            inplanes=outchannel, planes=outchannel, stride=1, groups=groups, downsample=None)

    def forward(self, x):
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        return x


class down_block(nn.Module):

    def __init__(self, pool_size=2, pool_stride=2, inchannel=None, outchannel=None, skip_channel=None, conv_group=1, basic_1_num=2) -> None:
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.skip = conv3x3(inchannel, skip_channel,
                            stride=1, groups=conv_group)
        self.basic_block_1 = nn.Sequential(*[
            BasicBlock(inplanes=inchannel, planes=inchannel,
                       stride=1, groups=conv_group, downsample=None)
            for _ in range(basic_1_num)])
        # self.temporal_attention = Temporal_Attention(inchannel=inchannel)
        self.basic_block_2 = BasicBlock(inplanes=inchannel, planes=outchannel,
                                        stride=1, groups=conv_group, downsample=nn.Sequential(
                                            conv1x1(
                                                in_planes=inchannel, out_planes=outchannel, stride=1, groups=conv_group),
                                            nn.BatchNorm2d(outchannel),
                                        ))
        self.basic_block_1_final = BasicBlock(
            inplanes=outchannel, planes=outchannel, stride=1, groups=conv_group, downsample=None)

    def forward(self, x):
        x = self.max_pool(x)
        skip = self.skip(x)
        x = self.basic_block_1(x)
        # x = self.temporal_attention(x)
        x = self.basic_block_2(x)
        x = self.basic_block_1_final(x)
        return x, skip


class up_block(nn.Module):

    def __init__(self, up_size=None, inchannel=None, outchannel=None, conv_group=1, basic_1_num=2) -> None:
        super().__init__()
        self.upsample = nn.Upsample(
            size=up_size, mode='bilinear', align_corners=True)
        self.basic_block_1 = nn.Sequential(*[
            BasicBlock(inplanes=inchannel, planes=inchannel,
                       stride=1, groups=conv_group, downsample=None)
            for _ in range(basic_1_num)])
        # self.temporal_attention = Temporal_Attention(inchannel=inchannel)
        self.basic_block_2 = BasicBlock(inplanes=inchannel, planes=outchannel,
                                        stride=1, groups=conv_group, downsample=nn.Sequential(
                                            conv1x1(
                                                in_planes=inchannel, out_planes=outchannel, stride=1, groups=conv_group),
                                            nn.BatchNorm2d(outchannel),
                                        ))
        self.basic_block_1_final = BasicBlock(
            inplanes=outchannel, planes=outchannel, stride=1, groups=conv_group, downsample=None)

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        x = self.upsample(x)
        x = self.basic_block_1(x)
        # x = self.temporal_attention(x)
        x = self.basic_block_2(x)
        x = self.basic_block_1_final(x)
        return x


class ReLU_mu(torch.nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU_mu}(x) = \min(\max(mu_min,x), mu_max)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``
        mu_max: max value of the ReLU_mu function
        mu_min: min value of the ReLU_mu function

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.
    """

    def __init__(self, inplace: bool = False, mu_max=None, mu_min=None) -> None:
        super(ReLU_mu, self).__init__(mu_min, mu_max, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def apply_complex(fr, fi, input_real, input_imag):
    '''
    apply complex function to complex input
    '''
    out_real = fr(input_real)-fi(input_imag)  # [batch, T, h, w]
    out_imag = fr(input_imag)+fi(input_real)  # [batch, T, h, w]
    out = torch.cat([out_real, out_imag], dim=1)  # [batch, 2T, h, w]
    return out


class ComplexConv2d(nn.Module):
    '''
    complex convolution 2d
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_real, input_imag):
        return apply_complex(self.conv_r, self.conv_i, input_real, input_imag)


class Conv_2d_OP(nn.Module):

    def __init__(self, n_in=16, n_out=16, ifactivate=False, T=cfg.data_shape[2]):
        super().__init__()
        self.conv = ComplexConv2d(
            n_in, n_out, 3, stride=1, padding='same', bias=False, groups=n_in)
        self.ifactivate = ifactivate
        if ifactivate == True:
            self.relu = nn.ELU()
        self.T = T

    def forward(self, input):
        '''
        Args:
            input: [batch, 2T, h, w]
        Returns:
            output: [batch, 2T, h, w]
        '''
        real_in = input[:, :self.T, :, :]
        imag_in = input[:, self.T:, :, :]
        res = self.conv(real_in, imag_in)
        if self.ifactivate == True:
            res = self.relu(res)
        return res
