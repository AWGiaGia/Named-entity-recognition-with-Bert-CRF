import time,os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import torch.nn as nn
from data.dataset import preprocessing
from model.BertCRF import BertCRF
import random
from utils.metric import Metrics
from sklearn.metrics import f1_score,accuracy_score,recall_score
import yaml
import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings("ignore")

def seed_anything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True


def test(model,test_dataloader,device,save_path,tagset_path):
    tagset = []
    with open(tagset_path, 'r', encoding='utf-8') as f:
        tagset = json.load(f)
        
    pred_list = None
    label_list = None
    mask_list = None
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            sentences,masks,tags = tuple(t.to(device) for t in batch)
            tag_seq = model.forward(sentences,masks) #(b,s)
            
            if pred_list == None:
                pred_list = tag_seq
                label_list = tags
                mask_list = masks
            else:
                pred_list = torch.concat((pred_list,tag_seq),0)
                label_list = torch.concat((label_list,tags),0)
                mask_list = torch.concat((mask_list,masks),0)
            
    pred_list = pred_list.cpu().flatten().tolist()
    label_list = label_list.cpu().flatten().tolist()
    mask_list = mask_list.cpu().flatten().tolist()
    
    eff_preds = []
    eff_labels = []


    for i in range(len(mask_list)):
        if mask_list[i] == True:
            eff_preds.append(pred_list[i])
            eff_labels.append(label_list[i])


    Metrics(eff_labels,eff_preds,tagset)
    # acc = accuracy_score(eff_labels,eff_preds)
    # recall = recall_score(eff_labels,eff_preds,average='macro')
    # f1score = f1_score(eff_labels,eff_preds,average='macro')
    # print(f'acc is: {acc}')
    # print(f'recall is: {recall}')
    # print(f'f1-socre is: {f1score}')
    









if __name__ == '__main__':
    config_path = './config/test.yaml'
    with open(config_path,'r',encoding='utf-8') as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)
    
    print(f'EXP Settings: ')
    for k,v in configs.items():
        print(f'{k}: {v}')
    print(f'*'*30)

    seed_anything(configs['seed'])

    # 创建结果保存路径
    if not os.path.exists(os.path.join('./output','test',configs['name'])):
        os.mkdir(os.path.join('./output','test',configs['name']))

    # 将配置文件保存到结果保存路径
    with open(os.path.join('./output','test',configs['name'],'test.yaml'),'w') as f:
        f.write(yaml.dump(configs,allow_unicode=True))

    # 读取测试数据
    test_sentences, test_masks, test_tag_lists = preprocessing(configs['test_path'],configs['MAX_LEN'],configs['tokenizer_path'],configs['tag2idx_path'])

    # 转化为tensor类型
    test_sentences = torch.tensor(test_sentences)
    test_masks = torch.tensor(test_masks) > 0.5 # 转为bool
    test_tag_lists = torch.tensor(test_tag_lists)

    # 创建DataLoader
    test_dataset = TensorDataset(test_sentences,test_masks,test_tag_lists)
    test_sampler = RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,sampler=test_sampler,batch_size=configs['batch_size'],num_workers=configs['num_workers'])

    # 加载模型
    model = BertCRF(configs['types_of_tags'],configs['pretrained_bert'],configs['device'],configs['autoCRF'],configs['MAX_LEN'])
    model.to(configs['device'])
    state_dict = torch.load(configs['pretrained_ckpt'])
    model.load_state_dict(state_dict)
    
    print("Start testing:\n")
    test(model,test_dataloader,configs['device'],os.path.join('./weight',configs['name']),configs['idx2tag_path'])

