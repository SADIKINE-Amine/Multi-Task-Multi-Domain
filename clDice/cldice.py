import  torch
import  torch.nn            as nn
import  torch.nn.functional as F
from    .soft_skeleton      import soft_skel
from    ipdb                import set_trace

class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, iters)
        skel_true = soft_skel(y_true, iters)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,1:,...])+smooth)/(torch.sum(skel_pred[:,1:,...])+smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,1:,...])+smooth)/(torch.sum(skel_true[:,1:,...])+smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=10, alpha=0.4, smooth = 1., sigmoid=False):
        super(soft_dice_cldice, self).__init__()
        self.iter       = iter_
        self.smooth     = smooth
        self.alpha      = alpha
        self.sigmoid    = sigmoid
        self.sig        = nn.Sigmoid()

    def forward(self, y_pred, y_true):
        if self.sigmoid:
            y_pred = self.sig(y_pred)
        dice        = soft_dice(y_true, y_pred)
        skel_pred   = soft_skel(y_pred, self.iter)
        skel_true   = soft_skel(y_true, self.iter)
        tprec       = (torch.sum(torch.mul(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens       = (torch.sum(torch.mul(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice     = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice
