import torch
import torch.nn as nn
import torch.nn.functional as Func

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

def space_to_batch(lr, patch_shape):
    # lr is a single low resolution frame of shape (N,C,H*r,W*r)
    # patch_shape is the (H,W) info of a single patch
    # output = tensor of shape (N*r*r,C,H,W)
    ph, pw = patch_shape[0], patch_shape[1]
    output = torch.randn(0,lr.shape[1],ph,pw).to(device)
    for n in range(lr.shape[0]):
        temp = lr[n].unfold(0, 3, 3).unfold(1, ph, pw).unfold(2, ph, pw)
        temp = torch.reshape(temp, (-1,lr.shape[1],ph,pw))
        output = torch.cat((output, temp))
    return output

def batch_to_space(patches, patch_num):
    # patches is a tensor of shape (N*r*r,C,H,W)
    # patch_num is the number of row/column of patches from the original frame
    # output = tensor of shape (N,C,H*r,W*r)
    ps = patches.shape
    patch_n, patch_c, patch_h, patch_w = ps[0], ps[1], ps[2], ps[3]
    pn_h, pn_w = patch_num[0], patch_num[1]
    batch = patch_n // (pn_h * pn_w)
    output = torch.randn(batch, patch_c, patch_h*pn_h, patch_w*pn_w).to(device)
    cnt = 0
    for n in range(batch):
        for h in range(pn_h):
            for w in range(pn_w):
                output[n, :, (patch_h*h):(patch_h*(h+1)), (patch_w*w):(patch_w*(w+1))] = patches[cnt]
                cnt += 1
    return output

def depth_to_space(x, scale):
    return Func.pixel_shuffle(x, scale)

class AdaptiveConv(nn.Module):
    def __init__(self, F):
        super(AdaptiveConv, self).__init__()
        self.conv_feature = nn.Conv2d(3, 256, kernel_size=(3,3), stride=(1,1))
        self.conv_kernel = nn.Conv2d(256, 27 * F, kernel_size=(3,3), stride=(1,1))
        self.conv_bias = nn.Conv2d(256, F, kernel_size=(3,3), stride=(1,1))
        self.F = F

    def forward(self, x):
        # This operation assumes that the input always has 3 channels
        # and size of computed kernel is always 3 * 3 * 3
        feature = Func.relu(self.conv_feature(x))
        batch, channel, x_h, x_w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        x_unfold = torch.permute(Func.unfold(x, 3, padding=1), (0,2,1))
        kernel = self.conv_kernel(feature)

        kernel = torch.reshape(kernel, (batch, self.F, 27))
        kernel = torch.permute(kernel, (0,2,1))

        output = torch.matmul(x_unfold, kernel)
        output = torch.permute(output, (0,2,1))
        output = torch.reshape(output, (batch,self.F,x_h,x_w))

        bias = self.conv_bias(feature)
        output = Func.relu(output + bias)
        return output

class RegularConv(nn.Module):
    def __init__(self, F, scale):
        super(RegularConv, self).__init__()
        self.conv1 = nn.Conv2d(F, 128, kernel_size=(5,5), stride=(1,1), padding='same')
        self.conv2 = nn.Conv2d(128, 32, kernel_size=(3,3), stride=(1,1), padding='same')
        self.conv3 = nn.Conv2d(32, 3 * scale * scale, kernel_size=(3,3), stride=(1,1), padding='same')
    
    def forward(self, x):
        x = Func.relu(self.conv1(x))
        x = Func.relu(self.conv2(x))
        output = self.conv3(x)
        return output

class SR_Model(nn.Module):
    def __init__(self, F, scale, patch_shape):
        super(SR_Model, self).__init__()
        self.adaptive = AdaptiveConv(F)
        self.regular = RegularConv(F, scale)
        self.patch_shape = patch_shape
        self.scale = scale

    def forward(self, x):
        # x is the single low resolution frame
        patch_num = (x.shape[-2] // self.patch_shape[0], x.shape[-1] // self.patch_shape[1])
        out = space_to_batch(x, self.patch_shape)
        out = self.adaptive(out)
        out = batch_to_space(out, patch_num)
        out = self.regular(out)
        out = depth_to_space(out, self.scale)
        return out