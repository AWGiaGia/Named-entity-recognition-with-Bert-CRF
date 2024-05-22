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
# from utils.metric import Metrics
from sklearn.metrics import f1_score,accuracy_score,recall_score
import yaml
import matplotlib.pyplot as plt

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

def initialize_model(epochs,device,pretrained_bert,types_of_tags,fix_bert,autoCRF,MAX_LEN):
    # 定义模型
    model = BertCRF(types_of_tags,pretrained_bert,device,fix_bert,autoCRF,MAX_LEN)
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=5e-5,  # 默认学习率
                      eps=1e-8  # 默认精度
                      )
    
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    return model, optimizer, scheduler

def train(model,train_dataloader,val_dataloader,optimizer,scheduler,epochs,val_gap,device,save_path):
    best_f1 = 0.
    train_loss_curve = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for _,batch in enumerate(train_dataloader):
            sentences,masks,tags = tuple(t.to(device) for t in batch)
            
            model.zero_grad()
            optimizer.zero_grad()
            loss, tag_seq = model.neg_log_likelihood_loss(sentences,masks,tags)

            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        train_loss_curve.append(total_loss)
        print(f"Epoch {epoch + 1}/{configs['epochs']}, Loss: {total_loss / len(train_dataloader)}")

        if epoch % val_gap == 0 or epoch >= epochs:
            model.eval()
            pred_list = None
            label_list = None
            mask_list = None
            with torch.no_grad():
                for _,batch in enumerate(val_dataloader):
                    sentences,masks,tags = tuple(t.to(device) for t in batch)
                    tag_seq = model.forward(sentences,masks) #(b,s)

                    # print(f'Here shape of tag_seq is: {tag_seq.shape}')

                    if pred_list == None:
                        pred_list = tag_seq
                        label_list = tags
                        mask_list = masks
                    else:
                        pred_list = torch.concat((pred_list,tag_seq),0)
                        label_list = torch.concat((label_list,tags),0)
                        mask_list = torch.concat((mask_list,masks),0)

                    # pred_list += tag_seq.cpu().float().tolist()
                    # label_list += tags.cpu().float().tolist()


                pred_list = pred_list.cpu().flatten().tolist()
                label_list = label_list.cpu().flatten().tolist()
                mask_list = mask_list.cpu().flatten().tolist()
                
                eff_preds = []
                eff_labels = []


                for i in range(len(mask_list)):
                    if mask_list[i] == True:
                        # # 去掉'O'
                        # if label_list[i] == 16:
                        #     continue
                        eff_preds.append(pred_list[i])
                        eff_labels.append(label_list[i])



                acc = accuracy_score(eff_labels,eff_preds)
                recall = recall_score(eff_labels,eff_preds,average='macro')
                f1score = f1_score(eff_labels,eff_preds,average='macro')
                print(f'acc is: {acc}')
                print(f'recall is: {recall}')
                print(f'f1-socre is: {f1score}')

                if best_f1 < f1score:
                    best_f1 = f1score
                    torch.save(model.state_dict(),os.path.join(save_path,'best.pth'))

                torch.save(model.state_dict(),os.path.join(save_path,f'ep{epoch}.pth'))

    plt.plot(train_loss_curve)
    plt.savefig(os.path.join(save_path,f'loss_curve.png'))
    plt.clf()










if __name__ == '__main__':
    config_path = './config/train.yaml'
    with open(config_path,'r',encoding='utf-8') as f:
        configs = yaml.load(f,Loader=yaml.FullLoader)
    
    print(f'EXP Settings: ')
    for k,v in configs.items():
        print(f'{k}: {v}')
    print(f'*'*30)

    seed_anything(configs['seed'])

    # 创建结果保存路径
    if not os.path.exists(os.path.join('./weight',configs['name'])):
        os.mkdir(os.path.join('./weight',configs['name']))
    if not os.path.exists(os.path.join('./output','train',configs['name'])):
        os.mkdir(os.path.join('./output','train',configs['name']))

    # 将配置文件保存到模型保存路径
    with open(os.path.join('./weight',configs['name'],'train.yaml'),'w') as f:
        f.write(yaml.dump(configs,allow_unicode=True))

    # 读取训练集和测试集数据
    train_sentences, train_masks, train_tag_lists = preprocessing(configs['train_path'],configs['MAX_LEN'],configs['tokenizer_path'],configs['tag2idx_path'])
    val_sentences, val_masks, val_tag_lists = preprocessing(configs['val_path'],configs['MAX_LEN'],configs['tokenizer_path'],configs['tag2idx_path'])

    # 转化为tensor类型
    train_sentences = torch.tensor(train_sentences)
    train_masks = torch.tensor(train_masks) > 0.5 # 转为bool
    train_tag_lists = torch.tensor(train_tag_lists)
    val_sentences = torch.tensor(val_sentences)
    val_masks = torch.tensor(val_masks) > 0.5 # 转为bool
    val_tag_lists = torch.tensor(val_tag_lists)

    # 创建DataLoader
    train_dataset = TensorDataset(train_sentences,train_masks,train_tag_lists)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=configs['batch_size'],num_workers=configs['num_workers'])

    val_dataset = TensorDataset(val_sentences,val_masks,val_tag_lists)
    val_sampler = RandomSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,sampler=val_sampler,batch_size=configs['batch_size'],num_workers=configs['num_workers'])

    # 加载模型
    model, optimizer, scheduler = initialize_model(epochs=configs['epochs'],
                                                   device = configs['device'],
                                                   pretrained_bert=configs['pretrained_bert'],
                                                   types_of_tags = configs['types_of_tags'],
                                                   fix_bert = configs['fix_bert'],
                                                   autoCRF = configs['autoCRF'],
                                                   MAX_LEN = configs['MAX_LEN'])

    print("Start training and validating:\n")
    train(model,train_dataloader,val_dataloader,optimizer,scheduler,configs['epochs'],configs['val_gap'],configs['device'],os.path.join('./weight',configs['name']))

