from torch import nn

#Code source: https://github.com/wgchang/DSBN/blob/master/model/dsbn.py

class _DomainSpecificBatchNorm(nn.Module):

    def __init__(self, num_features, num_domains, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()

        self.bns = nn.ModuleList(
            [nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_domains)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class DomainSpecificBatchNorm3d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))


# from torchvision import models
# import torch
# from torch.nn import functional

# class SingleConv(nn.Module):
#     ''' {Conv2d, BN, ReLU} '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.single_conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        
#     def forward(self, x):
#         return self.single_conv(x)

# class SimpleConv(nn.Module):
#     ''' {Conv2d, BN, ReLU} '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.simple_conv = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True))
        
#     def forward(self, x):
#         return self.simple_conv(x)
    
# class DoubleConv(nn.Module):
#     ''' {Conv2d, BN, ReLU}x2 '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
#             DomainSpecificBatchNorm2d(out_chan, 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             DomainSpecificBatchNorm2d(out_chan, 2),
#             nn.ReLU(inplace=True))
        
#     def forward(self, x):
#         return self.double_conv(x)
    
# class TripleConv(nn.Module):
#     ''' {Conv2d, BN, ReLU}x3 '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.triple_conv = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True))          

#     def forward(self, x):
#         return self.triple_conv(x)
    
# class QuadripleConv(nn.Module):
#     ''' {Conv2d, BN, ReLU}x4 '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.quadriple_conv = nn.Sequential(
#             nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_chan),
#             nn.ReLU(inplace=True))

#     def forward(self, x):
#         return self.quadriple_conv(x)
    
# class DoubleDown(nn.Module):
#     ''' maxPool2d + {Conv2d, BN, ReLU}x2 '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.double_down = nn.Sequential(nn.MaxPool2d(2, 2), DoubleConv(in_chan, out_chan))

#     def forward(self, x):
#         return self.double_down(x)
    
# class DoubleUp(nn.Module):
#     ''' ConvTranspose2d + {Conv2d, BN, ReLU}x2 '''
    
#     def __init__(self, in_chan, out_chan, mid_chan=None):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
#         if mid_chan == None:
#             mid_chan = in_chan
#         self.conv = DoubleConv(mid_chan, out_chan)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x1, x2], dim=1)
#         return self.conv(x)
    
# class TripleUp(nn.Module):
#     ''' ConvTranspose2d + {Conv2d, BN, ReLU}x3 '''
    
#     def __init__(self, in_chan, out_chan, mid_chan=None):
#         super().__init__()
#         self.up = nn.ConvTranspose2d(in_chan, out_chan, kernel_size=2, stride=2)
#         if mid_chan == None:
#             mid_chan = in_chan
#         self.conv = TripleConv(mid_chan, out_chan)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x1, x2], dim=1)
#         return self.conv(x)
    
# class QuadripleUp(nn.Module):
#     ''' ConvTranspose2d + {Conv2d, BN, ReLU}x4 '''
    
#     def __init__(self, in_chan, out_chan, mid_chan=None):
#         super().__init__()

#         self.up = nn.ConvTranspose2d(in_chan , out_chan, kernel_size=2, stride=2)
#         if mid_chan == None:
#             mid_chan = in_chan
#         self.conv = QuadripleConv(mid_chan, out_chan)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x1, x2], dim=1)
#         return self.conv(x)
    
# class OutConv(nn.Module):
#     ''' Conv2d '''
    
#     def __init__(self, in_chan, out_chan):
#         super().__init__()
#         self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
    
# def up_sample2d(x, t, mode="bilinear"):
#     ''' 2D up-sampling '''
    
#     return functional.interpolate(x, t.size()[2:], mode=mode, align_corners=False)

# class uNet(nn.Module):
    
#     def __init__(self, n_channels, n_classes):
        
#         super(uNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.enblock1 = DoubleConv(n_channels, 32)
#         self.enblock2 = DoubleDown(32, 64)
#         self.enblock3 = DoubleDown(64, 128)
#         self.enblock4 = DoubleDown(128, 256)
#         self.center = DoubleDown(256, 256)
#         self.deblock1 = DoubleUp(256, 256, 512)
#         self.deblock2 = DoubleUp(256, 128)
#         self.deblock3 = DoubleUp(128, 64)
#         self.deblock4 = DoubleUp(64, 32)
#         self.outc = OutConv(32, n_classes)

#     def forward(self, x):
#         x1 = self.enblock1(x)
#         x2 = self.enblock2(x1)
#         x3 = self.enblock3(x2)
#         x4 = self.enblock4(x3)
#         x5 = self.center(x4)
#         x = self.deblock1(x5,x4)
#         x = self.deblock2(x,x3)
#         x = self.deblock3(x,x2)
#         x = self.deblock4(x,x1)
#         logits = self.outc(x)
#         return logits

# if __name__ == "__main__":
#     from ipdb import set_trace
#     from torchsummary import summary
#     net = uNet(1,1)
#     set_trace()
