import torch.nn as nn
import torch


class DCFNetFeature(nn.Module):
    def __init__(self):
        super(DCFNetFeature, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
        )

    def forward(self, x):
        return self.feature(x)
    

class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature()
        # wf: the fourier transformation of correlation kernel w. You will need to calculate the best wf in update method.
        self.wf = None
        # xf: the fourier transformation of target patch x.
        self.xf = None
        self.config = config

    def forward(self, z):
        """
        :param z: the multiscale searching patch. Shape (num_scale, 3, crop_sz, crop_sz)
        :return response: the response of cross correlation. Shape (num_scale, 1, crop_sz, crop_sz)

        You are required to calculate response using self.wf to do cross correlation on the searching patch z
        """
        # obtain feature of z and add hanning window
        z = self.feature(z) * self.config.cos_window
        # TODO: You are required to calculate response using self.wf to do cross correlation on the searching patch z
        # put your code here
        
        #calculate fourier transform
        z = torch.rfft(z, signal_ndim = 2)
        
        ### complex multiplication (x + yi)(u + vi) = (xu - yv) + (xv + yu)i
        ### conjugate multiplication (y1 - y2i)(p1 + p2i) = (y1p1 + y2p2) + (y1p2 - y2p1)i
        
        real_phiw = self.wf[..., 0] * z[..., 0] + self.wf[..., 1] * z[..., 1]
        imag_phiw = self.wf[..., 0] * z[..., 1] - z[..., 0] * self.wf[..., 1]
        phi_w = torch.stack((real_phiw, imag_phiw), -1)
        
        #n, c, h, w, v = phi_w.shape
        #response = torch.irfft(phi_w.reshape(n, 1, c, h, w, v).sum(2), signal_ndim = 2)
        response = torch.irfft(torch.sum(phi_w, dim = 1, keepdim=True), signal_ndim = 2)
        
        return response

    def update(self, x, lr=1.0):
        """
        this is the to get the fourier transformation of  optimal correlation kernel w
        :param x: the input target patch (1, 3, h ,w)
        :param lr: the learning rate to update self.xf and self.wf

        The other arguments concealed in self.config that will be used here:
        -- self.config.cos_window: the hanning window applied to the x feature. Shape (crop_sz, crop_sz),
                                   where crop_sz is 125 in default.
        -- self.config.yf: the fourier transform of idea gaussian response. Shape (1, 1, crop_sz, crop_sz//2+1, 2)
        -- self.config.lambda0: the coefficient of the normalize term.

        things you need to calculate:
        -- self.xf: the fourier transformation of x. Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        -- self.wf: the fourier transformation of optimal correlation filter w, calculated by the formula, Shape (1, channel, crop_sz, crop_sz//2+1, 2)
        """
        # x: feature of patch x with hanning window. Shape (1, 32, crop_sz, crop_sz)
        x = self.feature(x) * self.config.cos_window
        
        # TODO: calculate self.xf and self.wf
        # put your code here
        
        xf = torch.rfft(x, signal_ndim=2)
        
        #if x.shape[-1] != 2:
            #xf = torch.rfft(x, signal_ndim=2)
        #else:
            #xf = torch.fft(x, signal_ndim=2)
        
        ### (x1 + x2i)(y1 - y2i) = (x1y1 + x2y2)+(y1x2 - x1y2)i
        
        real_phiy = self.config.yf[..., 0] * xf[..., 0] + self.config.yf[..., 1] * xf[..., 1]
        imag_phiy = self.config.yf[..., 0] * xf[..., 1] - self.config.yf[..., 1] * xf[..., 0]
        
        phi_y = torch.stack((real_phiy, imag_phiy), -1)
        
        ### (p1 + p2i)(p1 - p2i) = (p1^2 + p2^2) + (p1p2 - p1p2)i = (p1^2 + p2^2)
        
        xxf = torch.sum(torch.sum(xf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        
        #xxf = xf[..., 0]**2 + xf[..., 1]**2
        #n, c, h, w, v = xxf.shape
        #xxf = xxf.reshape(n, 1, c, h, w, v).sum(2)
        
        wf = phi_y/(xxf + self.config.lambda0)
        
        if (self.wf == None) and (self.xf == None):   
            self.wf = lr * wf.data
            self.xf = lr * xf.data
            
        else:
            self.wf = (1 - lr) * self.wf.data + lr * wf.data
            self.xf = (1 - lr) * self.xf.data + lr * xf.data
    

    def load_param(self, path='param.pth'):
        checkpoint = torch.load(path)
        if 'state_dict' in checkpoint.keys():  # from training result
            state_dict = checkpoint['state_dict']
            if 'module' in state_dict.keys()[0]:  # train with nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)
            else:
                self.load_state_dict(state_dict)
        else:
            self.feature.load_state_dict(checkpoint)

