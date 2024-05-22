import torch
import torch.utils.data.dataset as Dataset
import json
from transformers import BertTokenizer

# idx2tag = []
# tag2idx = {}
# with open('F:\\dasan_shiyan\\MachineLearning\\Exp1\Code\\config\\idx2tag.json', 'r', encoding='utf-8') as f:
#     idx2tag = json.load(f)
# with open('F:\\dasan_shiyan\\MachineLearning\\Exp1\Code\\config\\tag2idx.json', 'r', encoding='utf-8') as f:
#     tag2idx = json.load(f)

# print(tag2idx)

# idx2tag = json.load('F:\\dasan_shiyan\\MachineLearning\\Exp1\Code\\config\\idx2tag.json')
# print(idx2tag)

def preprocess_single_sentence(sentence,tokenizer_path,MAX_LEN):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    encoded_sent = tokenizer.encode_plus(
        text=sentence,  # 预处理语句
        add_special_tokens=True,  # 加 [CLS] 和 [SEP]
        max_length=MAX_LEN,  # 截断或者填充的最大长度
        padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
        return_attention_mask=True  # 返回 attention mask
    )
    # attention_mask = encoded_sent.get('attention_mask')
    # print(f'Here attention_mask is: {attention_mask}')
    # print(f'Here shape of attention_mask: {attention_mask.shape}')
    # raise ValueError('out')
    
    return encoded_sent.get('input_ids'),encoded_sent.get('attention_mask')
    

def preprocessing(data_root,MAX_LEN,tokenizer_path,tag2idx_path):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
    tag2idx = {}
    with open(tag2idx_path, 'r', encoding='utf-8') as f:
        tag2idx = json.load(f)


    input_ids = []
    input_tags = []
    attention_masks = []

    word_lists = []
    tag_lists = []
    with open(data_root,'r',encoding='utf-8') as f:
        word_str = ''
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_str += word[0]
                tag_list.append(tag2idx[tag])
            else:
                word_lists.append(word_str)
                tag_lists.append(tag_list)
                word_str = ''
                tag_list = []
    
    for sent in word_lists:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # 预处理语句
            add_special_tokens=True,  # 加 [CLS] 和 [SEP]
            max_length=MAX_LEN,  # 截断或者填充的最大长度
            padding='max_length',  # 填充为最大长度，这里的padding在之间可以直接用pad_to_max但是版本更新之后弃用了，老版本什么都没有，可以尝试用extend方法
            return_attention_mask=True  # 返回 attention mask
        )

        # 把输出加到列表里面
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    
    for tag in tag_lists:
        padded_tag = []
        for i in range(MAX_LEN):
            if i < len(tag):
                padded_tag.append(tag[i])
            else:
                padded_tag.append(16)

        input_tags.append(padded_tag)

    # print(f'attention_masks is: {attention_masks}')
    # raise ValueError('out')

    # print(f'Here input_ids is: {input_tags}')
    # raise ValueError('out')


    return input_ids,attention_masks,input_tags







if __name__ == '__main__':
    input_ids, attention_masks,tag_lists = preprocessing('F:\\dasan_shiyan\\MachineLearning\\Exp1\\Code\\NER_DATASET\\demo.train.char',
                                                    512,
                                                    'F:\\dasan_shiyan\\MachineLearning\\Exp1\\Code\\weight\\bert_hub\\sbert-base-chinese-nli',
                                                    'F:\\dasan_shiyan\\MachineLearning\\Exp1\Code\\config\\tag2idx.json')
    # tag_types = {}
    # for i in range(len(tag_lists)):
    #     for j in range(len(tag_lists[i])):
    #         tag_types[tag_lists[i][j]] = 1

    # idx = 0
    # for i in tag_types:
    #     print(f'No.{idx}: {i}')
    #     idx += 1

    # print(tag_lists)

    # print(input_ids)
    # print('*'*50)
    # print(attention_masks)
    # print('*'*50)
    # print(tag_lists)

    # with open('F:\\dasan_shiyan\\MachineLearning\\Exp1\\Code\\NER_DATASET\\demo.test.char','r',encoding='utf-8') as f:
    #     print(f.read())