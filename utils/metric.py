from sklearn.metrics import f1_score,accuracy_score,recall_score
import numpy as np

def Metrics(labels,preds,tagset):
    labels = np.array(labels)
    preds = np.array(preds)
    tagset_size = len(tagset)
    
    # print(f'Here preds: {preds}')
    # print(f'Here labels: {labels}')
    # raise ValueError('out')
    
    for i in range(tagset_size):
        pred_set = (preds == i)
        label_set = (labels == i)
        acc = accuracy_score(label_set,pred_set)
        recall = recall_score(label_set,pred_set)
        f1score = f1_score(label_set,pred_set)
        
        print(f'{tagset[i]}: {acc}(acc)\t {recall}(recall)\t {f1score}(f1score)')
        
    
    acc = accuracy_score(labels,preds)
    recall = recall_score(labels,preds,average='macro')
    f1score = f1_score(labels,preds,average='macro')
    print(f'total: {acc}(acc)\t {recall}(recall)\t {f1score}(f1score)')
        
        