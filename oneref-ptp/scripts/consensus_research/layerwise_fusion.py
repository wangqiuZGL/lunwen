"""Frozen multi-layer features; distinguish semantic and geometric mixing."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class LayerwiseHead(nn.Module):
    def __init__(self,model,kind,split=577):
        super().__init__();self.kind=kind;self.split=split
        for n in ('vision_head','language_head','visu_proj','bbox_embed'):setattr(self,n,copy.deepcopy(getattr(model,n)).float().requires_grad_(True))
        self.gate_norm=nn.LayerNorm(768)
        if kind=='shared_wide':self.gate=nn.Sequential(nn.Linear(768,6),nn.GELU(),nn.Linear(6,3))
        elif kind=='separate':self.gate=nn.Linear(768,6)
        else:self.gate=nn.Linear(768,3)
        self.strength=nn.Parameter(torch.zeros(2))
        if kind.startswith('layer'):
            self.gate.requires_grad_(False);self.gate_norm.requires_grad_(False);self.strength.requires_grad_(False)
        elif kind=='equal':self.gate.requires_grad_(False);self.gate_norm.requires_grad_(False)
        self.eval()
    def mix(self,features):
        if self.kind.startswith('layer'):
            index={'layer6':0,'layer9':1,'layer12':2}[self.kind]
            return features[:,index],features[:,index],None
        anchor=features[:,2]
        if self.kind=='equal':
            w=features.new_full((features.shape[0],3),1/3);ws=wg=w
        else:
            query=self.gate_norm(anchor[:,self.split])
            logits=self.gate(query)
            if self.kind=='separate':ws,wg=logits[:,:3].softmax(-1),logits[:,3:].softmax(-1)
            else:ws=wg=logits.softmax(-1)
        ds=(features-anchor[:,None])
        semantic=anchor+self.strength[0].tanh()*(ds*ws[:,:,None,None]).sum(1)
        geometric=anchor+self.strength[1].tanh()*(ds*wg[:,:,None,None]).sum(1)
        return semantic,geometric,torch.stack((ws,wg),1)
    def forward(self,features):
        semantic,geometric,weights=self.mix(features)
        vision=F.normalize(self.vision_head(semantic[:,:self.split]),dim=-1)[:,1:]
        language=F.normalize(self.language_head(semantic[:,self.split:]),dim=-1)[:,0]
        scores=(vision*language[:,None]).sum(-1)
        visual=self.visu_proj(geometric[:,:self.split])[:,1:]
        pooled=(scores.softmax(-1)[:,:,None]*visual).sum(1)
        return self.bbox_embed(pooled).sigmoid(),scores,weights
