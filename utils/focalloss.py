import os, sys

lib_path = os.path.abspath(os.path.join('..', 'datasets'))
sys.path.append(lib_path)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#from utils import one_hot_embedding

# focal loss
def neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pred = pred.unsqueeze(1).float()
  gt = gt.unsqueeze(1).float()

  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()
  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
  neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def focalloss(pred_mask, pred_regr, mask, regr, weight=0.4, size_average=True, num_class=20):
    # Binary mask loss
    pred_mask = torch.sigmoid(pred_mask) # class masks
    mask_loss = neg_loss(pred_mask, mask)
    
    sum=np.sum(mask.cpu().numpy()==1)
    
    # Regression L1 loss
    #print(pred_regr.size())
    #print(regr.size())
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask.sum(1)).sum(1).sum(1) / mask.sum(1).sum(1).sum(1)
    regr_loss = regr_loss.mean(0)
  
    # Sum
    loss = mask_loss +regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss ,mask_loss , regr_loss