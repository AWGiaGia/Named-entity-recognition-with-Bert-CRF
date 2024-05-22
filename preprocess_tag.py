import json
import os
from os.path import join,exists
from os import listdir,mkdir


data_root = './dataset/train.char'

if __name__ == '__main__':
    tag_set = {}
    tag_list = []
    idx = 0
    with open(data_root,'r',encoding='utf-8') as f:
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                if tag in tag_set:
                    continue
                else:
                    tag_set[tag] = idx
                    tag_list.append(tag)
                    idx +=1
    with open(join('./config','tag2idx.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tag_set,ensure_ascii=False))
    
    
    with open(join('./config','idx2tag.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tag_list,ensure_ascii=False))
    

    # with open("save.json","w", encoding='utf-8') as f: ## 设置'utf-8'编码
    #     f.write(    json.dumps(   dict1  ,ensure_ascii=False     )     ) 
        