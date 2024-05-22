import torch
from transformers import BertModel
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup

# pretrained_model = '../weight/bert_hub/sbert-base-chinese-nli'

# 中文Bert模型
class Bert(nn.Module):
    def __init__(
            self,
            pretrained_ckpt = '../weight/bert_hub/sbert-base-chinese-nli',
            D_out = 17 + 2, # 17个标签 + 1个开始符 + 1个终止符
            device = 'cpu',
            fix_bert = False
    ):
        super(Bert,self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_ckpt)
        # 将Bert输出的768维隐状态映射到
        self.proj = nn.Sequential(
            nn.Linear(768,100),
            nn.ReLU(),
            nn.Linear(100,D_out)
        )

        if fix_bert == True:
            print(f'Fixed bert backbone...')
            for param in self.bert.parameters():
                param.requires_grad = False
            
        # if fix_backbone:
        #     for param in self.face_embedder.parameters():
        #         param.requires_grad = False
    
    def forward(self,sentence,sentence_mask):
        output_sentences = self.bert(input_ids=sentence,
                            attention_mask=sentence_mask)
        
        output_tags = self.proj(output_sentences[0])
        # print(f'shape of output_tags is: {output_tags.shape}')

        # print(f'shape of outputs_sentences[0]: {output_sentences[0].shape}')

        return output_tags


