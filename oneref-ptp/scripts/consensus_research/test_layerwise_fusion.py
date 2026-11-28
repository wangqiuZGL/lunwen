import unittest
import torch
from torch import nn
from layerwise_fusion import LayerwiseHead


class TestFusion(unittest.TestCase):
    def test_anchor_and_gate_learning(self):
        torch.manual_seed(1)
        model=nn.Module();model.vision_head=nn.Linear(768,32);model.language_head=nn.Linear(768,32)
        model.visu_proj=nn.Linear(768,16);model.bbox_embed=nn.Linear(16,4);model.requires_grad_(False)
        x=torch.randn(2,3,7,768)
        base=LayerwiseHead(model,'layer12',split=4)(x)[0]
        for name in ('equal','shared','shared_wide','separate'):
            head=LayerwiseHead(model,name,split=4);pred,_,_=head(x)
            torch.testing.assert_close(pred,base,atol=0,rtol=0)
            self.assertTrue(head.bbox_embed.weight.requires_grad)
            pred.sum().backward();self.assertGreater(float(head.strength.grad.abs().sum()),0.)
            if name!='equal':
                head.zero_grad();head.strength.data.fill_(.2);head(x)[0].sum().backward()
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in head.gate.parameters()),0.)
    def test_layer_mapping(self):
        model=nn.Module();model.vision_head=nn.Identity();model.language_head=nn.Identity()
        model.visu_proj=nn.Identity();model.bbox_embed=nn.Linear(768,4)
        x=torch.randn(1,3,7,768)
        for name,i in [('layer6',0),('layer9',1),('layer12',2)]:
            s,g,w=LayerwiseHead(model,name,split=4).mix(x)
            self.assertTrue(torch.equal(s,x[:,i]));self.assertTrue(torch.equal(g,x[:,i]))


if __name__=='__main__':unittest.main()
