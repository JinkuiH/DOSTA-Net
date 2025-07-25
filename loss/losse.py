import torch
from loss.dice import SoftDiceLoss#, MemoryEfficientSoftDiceLoss,SoftDiceLossWeight
from loss.robust_ce_loss import RobustCrossEntropyLoss,RobustCrossEntropyLossWeight
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

class CharbonnierLoss(nn.Module):
        """Charbonnier Loss (L1)"""
        def __init__(self, eps=1e-3):
            super(CharbonnierLoss, self).__init__()
            self.eps = eps

        def forward(self, x, y):
            diff = x - y
            # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
            loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
            return loss
        
class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        #s#uper().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def softmax_helper_dim1(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 1)


# class DC_and_CE_lossWeight(nn.Module):
#     # def __init__(self, alpha=0.5, beta=0.1, weight_ce=1.0, weight_dice=1.0):
#     def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
#                  dice_class=SoftDiceLossWeight):
#         """
#         Combined Dice and Cross-Entropy Loss with sample weighting.
#         """
#         super(DC_and_CE_lossWeight, self).__init__()
#         if ignore_label is not None:
#             ce_kwargs['ignore_index'] = ignore_label
#             ce_kwargs['reduction'] = 'none'

#         self.weight_ce = weight_ce
#         self.weight_dice = weight_dice
#         self.dice_loss = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
#         self.ce_loss = nn.CrossEntropyLoss(reduction='none')

#     def forward(self, pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor):
#         """
#         pred: predicted logits, shape (n, c, x, y, z)
#         target: ground truth, shape (n, 1, x, y, z)
#         state: calcium state, shape (n, 1)
#         epoch: current epoch, scalar
#         """
#         # Compute sample-wise weights (w)
        
#         # w = torch.softmax(w,dim=0)
#         # Compute Dice and Cross-Entropy Loss
#         dice_loss = self.dice_loss(pred, target, w)
#         ce_loss = self.ce_loss(pred, target.squeeze().long())
#         ce_loss = torch.mean(ce_loss, dim=(1,2,3))
#         final_ce_loss = (w * ce_loss).sum()/w.sum()
#         # Combine losses
#         total_loss = self.weight_dice * dice_loss + self.weight_ce * final_ce_loss
#         return total_loss
    
class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        # print(net_output.shape, target_dice.shape)
        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result