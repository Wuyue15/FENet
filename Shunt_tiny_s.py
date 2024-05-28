import torch
import math
import numpy as np
# from Tool.attention import *
import cv2
from backbone.Shunted.SSA import *
import torch.nn as nn

channel_lay1 = 64
channel_lay2 = 128
channel_lay3 = 256
channel_lay4 = 512

#通过应用一个模糊滤波器来消除输入图像中的噪声，常使用一个高斯滤波器
def get_gaussian_kernel(k=3, mu=0, sigma=1, normalize=True):
    # compute 1 dimension gaussian
    gaussian_1D = np.linspace(-1, 1, k)
    # compute a grid distance from center
    x, y = np.meshgrid(gaussian_1D, gaussian_1D)
    distance = (x ** 2 + y ** 2) ** 0.5

    # compute the 2 dimension gaussian
    gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
    gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

    # normalize part (mathematically)
    if normalize:
        gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
    return gaussian_2D

#为了检测边缘，对图像应用一个滤波器来提取梯度。
def get_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x                                          #sobel算子的分子和分母
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1                             # 避免除以0
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator              #计算sobel算子
    return sobel_2D

#为了细化边缘，可以使用非最大抑制方法。在此之前，我们需要创建45°× 45°方向的kernel。
def get_thin_kernels(start=0, end=360, step=45):
    k_thin = 3
    k_increased = k_thin + 2                                         #在旋转时候有额外的空间，防止插值影响核的形状

    thin_kernel_0 = np.zeros((k_increased, k_increased))
    thin_kernel_0[k_increased // 2, k_increased // 2] = 1              #中心行设置为1，右侧部分设置为-1
    thin_kernel_0[k_increased // 2, k_increased // 2 + 1:] = -1

    thin_kernels = []
    for angle in range(start, end, step):
        # 旋转角度的核
        kernel_angle = np.zeros((k_thin, k_thin))
        for i in range(k_thin):
            for j in range(k_thin):
                # 计算当前角度下的坐标
                x = j - (k_thin - 1) / 2
                y = i - (k_thin - 1) / 2
                # 应用旋转                  使用三角函数计算每个角度下的旋转核，避免插值问题
                x_new = int(round(x * np.cos(np.radians(angle)) - y * np.sin(np.radians(angle)) + (k_thin - 1) / 2))
                y_new = int(round(x * np.sin(np.radians(angle)) + y * np.cos(np.radians(angle)) + (k_thin - 1) / 2))
                # 检查旋转后的坐标是否越界
                if 0 <= x_new < k_thin and 0 <= y_new < k_thin:
                    kernel_angle[y_new, x_new] = thin_kernel_0[i, j]
        thin_kernels.append(kernel_angle)
    return thin_kernels

class CannyFilter(nn.Module):
    def __init__(self,k_gaussian=3,mu=0,sigma=1,k_sobel=3,use_cuda=True):
        super(CannyFilter, self).__init__()
        # device
        self.device = 'cuda' if use_cuda else 'cpu'
        # gaussian
        gaussian_2D = get_gaussian_kernel(k_gaussian, mu, sigma)
        self.gaussian_filter = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=k_gaussian,padding=k_gaussian // 2,bias=False)
        gaussian_tensor = torch.from_numpy(gaussian_2D)
        gaussian_tensor_detach = gaussian_tensor.detach()
        self.gaussian_filter.weight.data.copy_(gaussian_tensor_detach)
        # sobel

        sobel_2D = get_sobel_kernel(k_sobel)                               #对图像应用一个滤波器来提取梯度。
        self.sobel_filter_x = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        sobel_filter_x = torch.from_numpy(sobel_2D)
        sobel_filter_x = sobel_filter_x.detach()
        self.sobel_filter_x.weight.data.copy_(sobel_filter_x)

        self.sobel_filter_y = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=k_sobel,padding=k_sobel // 2,bias=False)
        self.sobel_filter_y.weight.data.copy_(torch.from_numpy(sobel_2D.T))

        # thin

        thin_kernels = get_thin_kernels()
        directional_kernels = np.stack(thin_kernels)

        self.directional_filter = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=thin_kernels[0].shape,padding=thin_kernels[0].shape[-1] // 2,bias=False)
        self.directional_filter.weight[:, 0].data.copy_(torch.from_numpy(directional_kernels))

        # hysteresis

        hysteresis = np.ones((3, 3)) + 0.25
        self.hysteresis = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,padding=1,bias=False)
        self.hysteresis.weight[:].data.copy_(torch.from_numpy(hysteresis))

    def forward(self, img, low_threshold=None, high_threshold=None, hysteresis=False):
        # set the setps tensors
        B, C, H, W = img.shape
        blurred = torch.zeros((B, C, H, W)).to(self.device)         #创建一个这个形状的全为0的张量存在指定设备里
        grad_x = torch.zeros((B, 1, H, W)).to(self.device)
        grad_y = torch.zeros((B, 1, H, W)).to(self.device)
        grad_magnitude = torch.zeros((B, 1, H, W)).to(self.device)
        grad_orientation = torch.zeros((B, 1, H, W)).to(self.device)

        #  1. 使用高斯滤波器平滑图像
        for c in range(C):
            blurred[:, c:c + 1] = self.gaussian_filter(img[:, c:c + 1])
            blurred123 = blurred[:, c:c + 1].to(self.device)                         #1,1,10,10
            a = self.sobel_filter_x(blurred123)
            b = self.sobel_filter_y(blurred123)

            grad_x = grad_x + a            # 2. 计算图像梯度的幅度和方向
            grad_y = grad_y + b

        grad_x, grad_y = grad_x / C, grad_y / C
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_orientation = torch.atan2(grad_y,grad_x)
        grad_orientation = (torch.rad2deg(grad_orientation) + 180) % 360       # convert to degree      将弧度转换为角度
        grad_orientation = torch.round(grad_orientation / 45) * 45            # keep a split by 45   将梯度方向角度量化到最接近的45度的倍数。

        # thin edges

        directional = self.directional_filter(grad_magnitude)               # get indices of positive and negative directions
        positive_idx = (grad_orientation / 45) % 8
        negative_idx = ((grad_orientation / 45) + 4) % 8
        thin_edges = grad_magnitude.clone()
        # non maximum suppression direction by direction
        for pos_i in range(4):
            neg_i = pos_i + 4
            # get the oriented grad for the angle
            is_oriented_i = (positive_idx == pos_i) * 1
            is_oriented_i = is_oriented_i + (positive_idx == neg_i) * 1
            pos_directional = directional[:, pos_i]
            neg_directional = directional[:, neg_i]
            selected_direction = torch.stack([pos_directional, neg_directional])

            # get the local maximum pixels for the angle
            is_max = selected_direction.min(dim=0)[0] > 0.0
            is_max = torch.unsqueeze(is_max, dim=1)

            # apply non maximum suppression
            to_remove = (is_max == 0) * 1 * (is_oriented_i) > 0
            thin_edges[to_remove] = 0.0

        if low_threshold is not None:
            low = thin_edges > low_threshold

            if high_threshold is not None:
                high = thin_edges > high_threshold
                # get black/gray/white only
                thin_edges = low * 0.5 + high * 0.5

                if hysteresis:
                    # get weaks and check if they are high or not
                    weak = (thin_edges == 0.5) * 1
                    weak_is_high = (self.hysteresis(thin_edges) > 1) * weak
                    thin_edges = high * 1 + weak_is_high * 1
            else:
                thin_edges = low * 1

        return thin_edges

class CC(nn.Module):
    def __init__(self, inch1,inch2, use_cuda=True):
        super(CC, self).__init__()
        # self.device = 'cuda' if use_cuda else 'cpu'
        self.CannyFilter = CannyFilter()
        self.cb1 = convblock(inch1,inch1,3,1,1)
        self.up = Bup(inch1,inch2)

    def forward(self, x):
        x1 = self.CannyFilter(x)
        x2 = self.cb1(x)
        x = x2 * x1 + x2
        y = self.up(x)

        return y

class DSM(nn.Module):
    def __init__(self, inch1):
        super(DSM, self).__init__()
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()
        self.conv2 = nn.Sequential(
            nn.Conv2d(inch1, inch1, 1, 1, 0),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(inch1, inch1 // 2, 1, 1, 0),
            nn.Conv2d(inch1 // 2, inch1, 1, 1, 0),
        )

    def to_matrix(self, x):
            n, c, h, w = x.size()
            x = x.view(n, c, -1)
            return x

    def forward(self, g, d):
        h1 = self.avgpool1(g + d)
        h2 = self.sig(h1)
        h3 = self.conv2(h2)
        h4 = self.fc2(h3)
        h5 = self.sig(h4)
        x1 = h5 * g
        x2 = h5 * d
        y = x1 + x2
        return y

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

class Bup(nn.Module):
    def __init__(self,inch1,inch2 ):
        super(Bup, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inch1, inch2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, y):
        y1 = self.conv1(y)
        return y1

class shunttiny(nn.Module):
    def __init__(self):
        super(shunttiny, self).__init__()
        self.rgb_net = shunted_t(pretrained=True)
        # self.d_net = shunted_t(pretrained=True)

        self.cc1 = CC(channel_lay4, channel_lay3)
        self.cc2 = CC(channel_lay3, channel_lay2)
        self.cc3 = CC(channel_lay2, channel_lay1)
        self.cb = convblock(channel_lay1, channel_lay1,3,1,1)
        self.DSM3 = DSM(channel_lay3)
        self.DSM2 = DSM(channel_lay2)
        self.DSM1 = DSM(channel_lay1)

        ########                   jd              #######################
        self.jdy = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.jdx2 = nn.Sequential(
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True),
        )
        self.jdx3 = nn.Sequential(
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True),
        )

    def forward(self, rgb, d):
        d = torch.cat((d, d, d), dim=1)
        rgb_list = self.rgb_net(rgb)
        # depth_list = self.d_net(d)

        r1 = rgb_list[0]
        r2 = rgb_list[1]
        r3 = rgb_list[2]
        r4 = rgb_list[3]

        fus4 = r4
        x3 = self.cc1(fus4)
        fus3 = self.DSM3(x3, r3)
        x2 = self.cc2(fus3)
        fus2 = self.DSM2(x2, r2)
        x1 = self.cc3(fus2)
        y1 = self.DSM1(x1, r1)
        y = self.cb(y1)
        out1 = self.jdy(y)
        out2 = self.jdx2(x2)
        out3 = self.jdx3(x3)

        return out1,out2,out3,r1,r2,r3,r4,x3,x2,x1

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.rgb_net.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.rgb_net.load_state_dict(model_dict_r)

        # model_dict_d = self.d_net.state_dict()
        # state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        # model_dict_d.update(state_dict_d)
        # self.d_net.load_state_dict(model_dict_d)
        # ############################################################
        # sk = torch.load(pre_model)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in sk.items():
        #     name = k[9:]
        #     new_state_dict[name] = v
        # self.rgb_net.load_state_dict(new_state_dict, strict=False)
        # self.t_net.load_state_dict(new_state_dict, strict=False)
        # # self.rgb_depth.load_state_dict(new_state_dict, strict=False)
        print('self.rgb_uniforr loading', 'self.depth_unifor loading')

if __name__ == '__main__':
    rgb = torch.randn([1, 3, 320, 320]).cuda()                                   # batch_size=1，通道3，图片尺寸320*320
    depth = torch.randn([1, 1, 320, 320]).cuda()
    model = shunttiny().cuda()
    a = model(rgb, depth)
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    # print(a[3].shape)
    # print(a[4].shape)