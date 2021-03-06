'''
This file is from
https://github.com/zhirongw/lemniscate.pytorch/blob/master/lib/NCECriterion.py
'''
import pdb

import torch
from torch import nn

eps = 1e-7

class NCECriterion(nn.Module):

    def __init__(self, nLem):
        super(NCECriterion, self).__init__()
        self.nLem = nLem

    def forward(self, x, targets):

        ##########################
        # for_kl = x.clone()
        # for_kl = mytensor
        # gaussian = torch.rand_like(for_kl).to('cuda:0')
        # # kl_loss = torch.distributions.kl.kl_divergence(for_kl, gaussian)
        # # kl_loss = torch.nn.functional.kl_div(for_kl.log(), gaussian)
        # kl_loss = torch.nn.KLDivLoss(reduction='batchmean')(for_kl.log(), gaussian)
        kl_loss = 0
        # print('kl in nce', kl_loss)
        ##########################

        batchSize = x.size(0)
        K = x.size(1)-1
        Pnt = 1 / float(self.nLem)
        Pns = 1 / float(self.nLem)
        
        # eq 5.1 : P(origin=model) = Pmt / (Pmt + k*Pnt) 
        Pmt = x.select(1,0)
        Pmt_div = Pmt.add(K * Pnt + eps)
        lnPmt = torch.div(Pmt, Pmt_div)
        
        # eq 5.2 : P(origin=noise) = k*Pns / (Pms + k*Pns)
        Pon_div = x.narrow(1,1,K).add(K * Pns + eps)
        Pon = Pon_div.clone().fill_(K * Pns)
        lnPon = torch.div(Pon, Pon_div)
     
        # equation 6 in ref. A
        lnPmt.log_()
        lnPon.log_()
        
        lnPmtsum = lnPmt.sum(0)
        lnPonsum = lnPon.view(-1, 1).sum(0)
        
        loss = - (lnPmtsum + lnPonsum) / batchSize
        
        return loss, kl_loss

def create_model(opt):
    ndata = opt['ndata']
    return NCECriterion(ndata)
