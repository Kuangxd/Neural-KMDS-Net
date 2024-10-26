import torch
import torch.nn as nn
from collections import namedtuple
from ops.utils import soft_threshold
from tqdm import tqdm
KMDSNetParams = namedtuple('KMDSNetParams', ['inner_num', 'iter_num', 'use_kernel', 'kernel_depth', 'frame_num', 'use_ConvTranspose2d'])
import  numpy as np
import pdb


class KMDSNet4D(nn.Module):
    def __init__(self, params: KMDSNetParams):
        super(KMDSNet4D, self).__init__()
        self.inner_num = params.inner_num
        self.iter_num = params.iter_num
        self.use_kernel = params.use_kernel
        self.kernel_depth = params.kernel_depth
        self.frame_num = params.frame_num
        self.use_ConvTranspose2d = params.use_ConvTranspose2d

        if self.use_ConvTranspose2d:
            self.d1 = nn.Conv3d(self.frame_num, self.inner_num, kernel_size=3, padding=1, stride=2, bias=False)
        else:
            self.d1 = nn.Conv3d(self.frame_num, self.inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.dn1 = nn.BatchNorm3d(self.inner_num)
        self.d2 = nn.Conv3d(self.inner_num, self.inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.dn2 = nn.BatchNorm3d(self.inner_num)
        self.d3 = nn.Conv3d(self.inner_num, self.inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.dn3 = nn.BatchNorm3d(self.inner_num)
        self.d4 = nn.Conv3d(self.inner_num, self.inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.dn4 = nn.BatchNorm3d(self.inner_num)

        if self.use_ConvTranspose2d:
            self.c1 = nn.ConvTranspose3d(self.inner_num, self.frame_num, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.c1 = nn.Conv3d(self.inner_num, self.frame_num, kernel_size=3, padding=1, bias=False)
        self.cn1 = nn.BatchNorm3d(self.frame_num)
        self.c2 = nn.Conv3d(self.frame_num, self.frame_num, kernel_size=3, padding=1, bias=False)
        self.cn2 = nn.BatchNorm3d(self.frame_num)
        self.c3 = nn.Conv3d(self.frame_num, self.frame_num, kernel_size=3, padding=1, bias=False)
        self.cn3 = nn.BatchNorm3d(self.frame_num)
        self.c4 = nn.Conv3d(self.frame_num, self.frame_num, kernel_size=3, padding=1, bias=False)
        self.cn4 = nn.BatchNorm3d(self.frame_num)
        
        model = []
        for i in range(self.iter_num):
            model += [KMDBlock4D(self.frame_num, self.inner_num, self.use_ConvTranspose2d)]
        self.model = nn.Sequential(*model)

        if self.use_kernel:
            kernel_model_s = []
            kernel_model_e = []
            for i in range(self.kernel_depth):
                kernel_model_s += [CBRBlock3D(self.frame_num, self.frame_num)]
                kernel_model_e += [CBRBlock3D(self.frame_num, self.frame_num)]
            self.kernel_model_s = nn.Sequential(*kernel_model_s)
            self.kernel_model_e = nn.Sequential(*kernel_model_e)

    def forward(self, x):
        if self.use_kernel:
            x = self.kernel_model_s(x)        

        x1 = self.dn4(self.d4(self.dn3(self.d3(self.dn2(self.d2(self.dn1(self.d1(x))))))))
        for i in range(self.iter_num):
            x1 = self.model[i](x1, x)

        x1 = self.cn4(self.c4(self.cn3(self.c3(self.cn2(self.c2(self.cn1(self.c1(x1))))))))

        if self.use_kernel:
            x1 = self.kernel_model_e(x1)

        return x1

class KMDBlock4D(nn.Module):
    def __init__(self, frame_num=24, inner_num=32, use_ConvTranspose2d=0):
        super(KMDBlock4D, self).__init__()
        if use_ConvTranspose2d:
            self.d1 = nn.ConvTranspose3d(inner_num, frame_num, kernel_size=3, stride=2, padding=1, output_padding=1)
        else:
            self.d1 = nn.Conv3d(inner_num, frame_num, kernel_size=3, padding=1, bias=False)
        self.dn1 = nn.BatchNorm3d(frame_num)
        self.d2 = nn.Conv3d(frame_num, frame_num, kernel_size=3, padding=1, bias=False)
        self.dn2 = nn.BatchNorm3d(frame_num)
        self.d3 = nn.Conv3d(frame_num, frame_num, kernel_size=3, padding=1, bias=False)
        self.dn3 = nn.BatchNorm3d(frame_num)
        self.d4 = nn.Conv3d(frame_num, frame_num, kernel_size=3, padding=1, bias=False)
        self.dn4 = nn.BatchNorm3d(frame_num)

        if use_ConvTranspose2d:
            self.c1 = nn.Conv3d(frame_num, inner_num, kernel_size=3, padding=1, stride=2, bias=False)
        else:
            self.c1 = nn.Conv3d(frame_num, inner_num, kernel_size=3, padding=1, stride=1, bias=False)
        self.cn1 = nn.BatchNorm3d(inner_num)
        self.c2 = nn.Conv3d(inner_num, inner_num, kernel_size=3, padding=1, bias=False)
        self.cn2 = nn.BatchNorm3d(inner_num)
        self.c3 = nn.Conv3d(inner_num, inner_num, kernel_size=3, padding=1, bias=False)
        self.cn3 = nn.BatchNorm3d(inner_num)
        self.c4 = nn.Conv3d(inner_num, inner_num, kernel_size=3, padding=1, bias=False)
        self.cn4 = nn.BatchNorm3d(inner_num)

        self.soft_threshold = soft_threshold
        self.lmbda = nn.Parameter(torch.zeros(1, inner_num, 1, 1, 1))
        nn.init.constant_(self.lmbda, 0.02)
    
    def forward(self, x, g):
        x1 = self.dn4(self.d4(self.dn3(self.d3(self.dn2(self.d2(self.dn1(self.d1(x))))))))
        x1 = g - x1
        x1 = x + self.cn4(self.c4(self.cn3(self.c3(self.cn2(self.c2(self.cn1(self.c1(x1))))))))
        
        x1 = self.soft_threshold(x1, self.lmbda)

        return x1


class CBRBlock3D(nn.Module):
    def __init__(self, in_num=24, our_num=24):
        super(CBRBlock3D, self).__init__()
        self.c = nn.Conv3d(in_num, our_num, kernel_size=3, padding=1, bias=False)
        self.b = nn.BatchNorm3d(our_num)
        self.r = nn.ReLU()

    def forward(self, x):
        return self.b(self.c(x))