import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.Bert import Bert
from model.crf import CRF,AutoCRF

class BertCRF(nn.Module):
    def __init__(
            self,
            tagset_size = 19,
            bert_pretrained_ckpt = None,
            device='cpu',
            fix_bert = False,
            autoCRF = True,
            MAX_LEN = 512
    ):
        super(BertCRF,self).__init__()
        print(f'building BertCRF...')
        self.device = device

        self.bert = Bert(bert_pretrained_ckpt,tagset_size,device,fix_bert)
        
        if autoCRF:
            self.crf = AutoCRF(tagset_size,device,MAX_LEN)
        else:
            self.crf = CRF(tagset_size,device)

    # 计算标注序列
    def forward(self,sentence,sentence_mask):
        outs = self.bert(sentence,sentence_mask)
        tag_seq = self.crf._viterbi_decode(outs,sentence_mask)
        return tag_seq
    
    # 计算负对数似然损失
    def neg_log_likelihood_loss(self,sentence,sentence_mask,gt_tags):
        outs = self.bert(sentence,sentence_mask)
         
        total_loss = self.crf.neg_log_likelihood_loss(outs, sentence_mask, gt_tags)
        tag_seq = self.crf._viterbi_decode(outs, sentence_mask)
        return total_loss, tag_seq
    
