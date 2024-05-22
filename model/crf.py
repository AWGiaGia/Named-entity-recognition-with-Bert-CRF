# 条件随机场（CRF）的代码实现

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from TorchCRF import CRF as PytorchCRF





START_TAG = -2
STOP_TAG = -1

# 返回值由两部分构成：
# 1）max_score表示对于当前的每一个状态，上一个状态最有可能的得分
# 2）log_sum_exp_dif表示对于当前的每一个状态，上一个状态中各个状态与最有可能的状态的得分之间的对数指数差值
# 这两部分直接相加
def log_sum_exp(vec,m_size):
    '''
    args:
        vec(bat_size, from_target, to_target) (b,t,t)
        m_size: (to_target) (t)
    return:
        (bat_size, from_target) (b,t)
    '''
    # 计算出对于当前的每一个状态，其最有可能的上一个状态是什么 (b,t)
    _, idx = torch.max(vec, 1)  # (b,t)

    # idx.view(-1, 1, m_size)的形状为(b,1,t)
    # 表示选中对于当前的每一个状态，它最有可能的上一个状态的得分 (b,1,t)
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)

    # 将max_score扩展为(b,t,t)
    # vec的传入值是cur_values (b,t,t)，表示从上一个词到当前词的转移分数
    # 得到形状仍为(b,t,t)
    # 这里理解为：vec(i,j) - \max_{k} vec(k,j), for i,k in from_target and j in to_target
    # 即对于当前的每一个状态，计算每一个上一状态得分与最大的上一状态得分之差
    dif = vec - max_score.expand_as(vec)
    # 取指数
    exp_dif = torch.exp(dif)
    # 将其按行求和
    # 这里理解为：sum_dif_j = \sum_{i} dif(i,j), for i in from_target and j in to_target
    # 即将所有的指数分差加起来，得到(b,t)
    sum_exp_dif = torch.sum(exp_dif, 1)
    # 取对数
    log_sum_exp_dif = torch.log(sum_exp_dif)
    # 返回值由两个部分构成：
    # 1）max_score表示对于当前的每一个状态，上一个状态最有可能的得分
    # 2）log_sum_exp_dif表示对于当前的每一个状态，上一个状态中各个状态与最有可能的状态的得分之间的对数指数差值
    return max_score.view(-1, m_size) + log_sum_exp_dif


class CRF(nn.Module):
    def __init__(self,tagset_size,device):
        super(CRF,self).__init__()
        print(f'building batched crf...')
        self.device = device
        self.average_batch = False
        self.tagset_size = tagset_size

        init_transitions = torch.zeros(self.tagset_size, self.tagset_size)
        init_transitions.to(device)
        self.transitions = nn.Parameter(init_transitions) #转移概率矩阵
        # 两个tag_size是为了表示标签到标签的转移概率
        # 其中第一个tag_size表示前一个标签的数量，记作from_target_size
        # 第二个tag_size表示当前标签的数量，记作to_target_size

        # print(f'init_transitions is of: {init_transitions.shape}')

    # 执行维特比解码
    def forward(self, feats):
        path_score, best_path = self._viterbi_decode(feats)
        return path_score, best_path
    
    # 通过维特比算法，计算出概率最大的路径
    def _viterbi_decode(self,feats,mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)  (b,s,t)
                mask: (batch, seq_len) (b,s)
            output:
                decode_idx: (batch, seq_len) decoded sequence (b,s)
        """
        batch_size = feats.shape[0]
        seq_len = feats.shape[1]
        tag_size = feats.shape[2]
        assert(tag_size == self.tagset_size)
        # 对每个句子计算长度，返回维度为(b,1)，其中length_mask(i)表示一个batch中，第i个句子的长度
        length_mask = torch.sum(mask.long(),dim=1).view(batch_size,1).long()
        # print(f'Here length_mask is: {length_mask}')
        # print(f'Here shape of length_mask is: {length_mask.shape}')
        # raise ValueError('out')

        # mask to (s,b)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size #序列长与句子数的乘积(s*b)

        # (b,s,t)->(s,b,t)->(s*b,1,t)->(s*b,t,t)
        # 计算发射分数，即tag_size个标签各自对应的概率
        # 将(s*b,1,t)的发射分数扩张为(s*b,t,t)，是为了便于后续与转移分数self.transitions相加
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)

        # 将feats表示的发射分数与self.transitions对应的转移分数进行相加
        # 发射分数的来源：上游网络输出
        # 转移分数的来源：CRF模型参数
        # （这就是CRF的公式吗？）
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        # (s*b,t,t)->(s,b,t,t)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)


        seq_iter = enumerate(scores)

        back_points = list()
        # 存储每个时间步每个句子中，达到当前标签最大分数对应的最佳路径。
        # 在后续的动态规划过程中，使用这些最佳路径来回溯找到整个序列的最优路径

        partition_history = list()
        # 存储每个时间步每个句子中，达到当前标签的最大分数
        # 具体而言，即P(x_t) = \max P(x_t|x_{t-1})
        # x_t表示当前状态，x_{t-1}表示上一个状态

        # reverse mask
        mask = (1 - mask.long()).byte()
        # raise ValueError(f'shape of mask: {mask.shape}')

        is_first = True
        for idx,cur_values in seq_iter:
            # cur_values的维度为(b,t,t)，含义为：bat_size * from_tagset_size * to_tagset_size
            if is_first: #第一个字符
                # 第一个字符的状态，只能由START_TAG状态转换而来，因此其上一步的最佳标注即为START_TAG，当前标签的最大可能性得分为START_TAG对应的得分
                partition = cur_values[:, START_TAG, :].clone().view(batch_size, tag_size, 1) # (b,t,1)
                partition_history.append(partition)
                is_first = False
                continue #第一个字符的处理结束
            
            # cur_values是当前字符的分数矩阵(b,t,t)，表示从上一个标签到当前标签的分数
            # partition存储了到上一个时间步为止的每个标签的最大分数，(b,t,1)，表示上一个状态的可能性
            # 在维特比算法中，需要考虑上一个时间步到当前时间步的转移，并选择在上一个时间步中得分最高的标签
            # 因此，为计算当前时间步的最大分数，需要将上一个时间步的最大分数加上当前时间步的发射分数，得到
            # 当前时间步各个标签的累积分数
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

            # partition的维度为(b,t)，其中partition(b,i)的含义是当前标签的最大可能性得分（在上一步标签中取最大值）
            # cur_bp的维度是(b,t)，其中cur_bp(b,i)的含义是若当前状态为i，则上一步最有可能的状态是cur_bp(b,i)
            partition, cur_bp = torch.max(cur_values, 1) # partition为cur_values在from_target_size维度上的最大值，即(b,t)，cur_bp为其对应的索引

            # print(f'Here shape of partition is: {partition.shape}')
            # print(f'Here cur_bp is: {cur_bp}')
            # print(f'Here shape of cur_bp is: {cur_bp.shape}')
            # raise ValueError('out')

            # (b,t) -> (b,t,1)
            partition = partition.view(batch_size, tag_size, 1) 
            partition_history.append(partition)
        
        
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size).bool(), 0) 


            back_points.append(cur_bp)

        # 将一系列(b,t,1)拼接为(b,t,s)，然后转换为(s,b,t)，最后转换为(b,s,t)，和模型输入的feat的维度一致
        partition_history = torch.cat(partition_history,0).view(seq_len, batch_size,-1).transpose(1,0).contiguous() ## (batch_size, seq_len. tag_size)

        # 记录了每个句子的最后一个位置（即length-1），维度为(b,1,t)
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1

        # torch.gather(input, dim, index)
        # input: 目标变量，输入
        # dim: 需要沿着取值的坐标轴
        # index: 需要取值的索引矩阵
        # 参考解释：https://blog.csdn.net/weixin_42200930/article/details/108995776
        # 在这里直接理解为：选择每个句子最后一个词的标注情况，初始维度和last_position一致，即(b,1,t)
        # 然后再将其转置为(b,t,1)
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        
        # 将最后一个词的标注分数与转移矩阵分数进行逐元素相加，以考虑从每个标签到终止状态的转移情况
        # 维特比算法中每个句子的总得分由两部分组成：
        # 1）每个句子在最后一个词的标注情况，即last_partition张量中记录的分数
        # 2）从每个标签到终止状态的转移情况，即转移分数矩阵self.transitions
        # 也就是说，对于最后一个词，不仅需要考虑它的最终得分情况，还要考虑它转移到终止状态的得分
        # 因此上面需要为last_partition进行转置
        # 得到维度为(b,t,t)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)

        # 维度为(b,t)，同理，last_bp(b,i)的含义是若当前状态是i，则上一个最有可能的状态时last_bp(b,i)
        # 实际上，由于结束状态总是STOP_TAG，所以这里唯一有用的是last_bp(b,STOP_TAG)
        _, last_bp = torch.max(last_values, 1)

        # 维度为(b,t)的全零张量
        pad_zero = autograd.Variable(torch.zeros(batch_size, tag_size)).long()
        pad_zero = pad_zero.to(self.device)
        # 原本的back_points中包含s-1个cur_bp。其中，第j个cur_bp(b,i)，表示在第j个状态为i的情况下，第j-1个状态最有可能的值
        # 至于最后一个状态s-1（下标从零开始），其标签由last_bp(b,STOP_TAG)给出
        # 这样就会导致，当j=s-1（下标从零开始）时，第s-1个cur_bp(b,i)没有对应的值
        # 为了模型能够正确处理标签，需要对第s-1个cur_bp进行填充
        # 填充标签通常赋予较低的分数（这里是0）
        back_points.append(pad_zero)
        back_points = torch.cat(back_points).view(seq_len, batch_size, tag_size)

        # 选择最后一个词的状态，维度为(b,)
        pointer = last_bp[:, STOP_TAG]

        # 将pointer维度转换为(b,1,t)
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        # (s,b,t)-->(b,s,t)
        back_points = back_points.transpose(1,0).contiguous()

        # scatter(dim,index,src)
        # dim: 沿着哪个维度进行索引
        # index: 用来scatter的元素索引
        # src: 用来scatter的源元素，可以是一个标量也可以是一个张量
        # 相当于gather的逆操作。
        # 其中src对应gather的输出out、index对应gather的index表、输出对应gather的输入向量input
        # 从gather的角度考虑，就是根据gather的out，和index表，还原出gather的输入input
        # index和src的维度始终是要一致的
        # 输出的维度由back_points自身的维度指定
        # 这里的作用是：将终止标签的索引pointer，插入到back_points张量中的每个句子的最后一个时间步的位置上
        back_points.scatter_(1, last_position, insert_last)

        # (b,s,t) --> (s,b,t)
        back_points = back_points.transpose(1,0).contiguous()
        # (s,b)
        decode_idx = autograd.Variable(torch.LongTensor(seq_len, batch_size))
        decode_idx = decode_idx.to(self.device)

        # pointer.data的维度是(b)
        # 因此这里理解为，得到最后一个词的标记
        decode_idx[-1] = pointer.data
        
        # 从倒数第二个时间步开始，回溯找到整个句子的最佳标签序列
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data.squeeze(-1)


        # (s,b) --> (b,s)，方便处理
        decode_idx = decode_idx.transpose(1,0)


        return decode_idx

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        assert(tag_size == self.tagset_size)
        # (b,s) --> (s,b)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)

        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        seq_iter = enumerate(scores)

        is_first = True
        partition = None
        for idx, cur_values in seq_iter:
            if is_first:
                partition = cur_values[:, START_TAG, :].clone().view(batch_size, tag_size, 1)
                is_first = False
                continue


            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            # ----------------------------------------------------上面的部分和维特比算法相同-----------------------
            
            # (b,t,t)在第一个t的维度（from_target）求和，得到(b,t)
            cur_partition = log_sum_exp(cur_values, tag_size)

            # mask的当前维度为(s,b)
            # 取出在当前字符位置的掩码mask(i,b)，即维度为mask_idx(b)
            # 将其扩展为mask_idx(b,t)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            # masked_select用法参考https://blog.csdn.net/geiyes/article/details/116988016
            # 这里理解为：取出mask_idx为True部分的cur_partition，得到的数组为(1,x)，其中x与mask_idx为True的元素个数有关
            # x的最大值为b*t
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            # (b,t)-->(b,t,1)
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)
            # masked_scatter_用法参考https://blog.csdn.net/weixin_46398647/article/details/126857664
            # 作用是将mask_idx为True的位置，将partition对应位置替换为masked_cur_partition对应的值
            # 对于mask_idx为False的位置，partition对应位置仍保持原值
            # masked_cur_partition会被自动由(b*t)扩展为(b,t,1)，这样就可以完成该运算了
            partition.masked_scatter_(mask_idx, masked_cur_partition)  

        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        # 计算最后一个词的分数，维度为(b,t,t)
        # 计算方式为转移矩阵分数 + partition分数
        cur_values = self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)

        # 新的分数由两个部分构成：
        # 1）max_score表示对于当前的每一个状态，上一个状态最有可能的得分
        # 2）log_sum_exp_dif表示对于当前的每一个状态，上一个状态中各个状态与最有可能的状态的得分之间的对数指数差值
        # 最终新分数的表示为max_score + log_sum_exp_dif
        # 维度为(b,t)
        cur_partition = log_sum_exp(cur_values, tag_size)
        # 由于最后一个词的状态为STOP_TAG，因此只取STOP_TAG标签对应的分数
        final_partition = cur_partition[:, STOP_TAG]

        # final_partition.sum()表示将一个batch中所有的分数进行求和
        # scores返回的是上文中计算出来的状态分数矩阵，具体含义见维特比算法
        return final_partition.sum(), scores

    # 计算gold score
    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size) (s,b,t,t)
                mask: (batch, seq_len) (b,s)
                tags: tensor  (batch, seq_len) (b,s) 应该就是ground_truth
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        
        ## convert tag value into a new format, recorded label bigram information to index  
        new_tags = autograd.Variable(torch.LongTensor(batch_size, seq_len)) # (b,s)
        new_tags = new_tags.to(self.device)
        # 将每一个标签表示为前一个标签和当前标签的组合，以便更好地捕捉标签之间的转移情况
        # 对于第一个时间步，由于没有前一个标签，所以将当前标签直接映射为一个新的标签
        # 对于后续的时间步，将前一个标签和当前标签的组合映射为一个新的标签
        # 这种转换将标签序列转换为一个新的序列，其中每个元素都表示了相邻两个标签的组合情况
        # 从而更有效地表示了标签之间的关系
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:,0] =  (tag_size - 2)*tag_size + tags[:,0]

            else:
                new_tags[:,idx] =  tags[:,idx-1]*tag_size + tags[:,idx]

        # 只取出转移矩阵中当前状态为STOP_TAG的列，得到维度为(t)
        # 将其广播为(b,t)
        end_transition = self.transitions[:,STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        # 对每个句子计算长度，返回维度为(b,1)，其中length_mask(i)表示一个batch中，第i个句子的长度
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        # 取出每个句子最后一个词的gt标注，维度为(b,1)
        end_ids = torch.gather(tags, 1, length_mask - 1)
        # 得到转移矩阵中，最后一个词的gt标注所对应的分数，维度为(b,1)
        end_energy = torch.gather(end_transition, 1, end_ids)
        # (b,s) --> (s,b) --> (s,b,1)
        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1)

        # 将scores由(s,b,t,t)转化为(s,b,t*t)
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)

        tg_energy = tg_energy.masked_select(mask.transpose(1,0))

        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    # 计算负对数似然损失
    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)  (b,s,t)
                mask: (batch, seq_len) (b,s)
                tags: tensor  (batch, seq_len) (b,s) 
            output:
                loss value
        """
        # nonegative log likelihood
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        if self.average_batch:
            return (forward_score - gold_score)/batch_size
        else:
             return forward_score - gold_score

# import torch
# from torchcrf import CRF
# num_tags = 5  # NER数据集中
# crf = CRF(num_tags=num_tags,
# 			batch_first=True)
class AutoCRF(nn.Module):
    def __init__(self,tagset_size,device,MAX_LEN):
        super(AutoCRF,self).__init__()
        self.crf = PytorchCRF(num_labels = tagset_size)
        self.crf.to(device)
        self.MAX_LEN = MAX_LEN
    
    def _viterbi_decode(self,feats,mask):
        out = self.crf.viterbi_decode(feats,mask)
        
        # 将每个输出都填充到定长MAX_LEN
        padded = []
        for vector in out:
            padded.append(vector + [-1] * (self.MAX_LEN - len(vector)))
        
        return torch.tensor(padded)

    
    def neg_log_likelihood_loss(self,feats,masks,tags):
        loss = -1 * self.crf(feats,tags,masks)
        return loss.sum()




if __name__ == '__main__':
    # torch.Size([2,4,6])
    vec = torch.tensor(
        (
        [
            [
                1.,2.,3.,4.,5.,6.
            ],

            [
                7.,8.,9.,10.,11.,12.
            ],

            [
                6.,5.,4.,3.,2.,1.
            ],

            [
                3.,2.,1.,5.,6.,4.
            ]
        ],
        [
            [
                1.,2.,3.,4.,5.,6.
            ],

            [
                7.,8.,9.,10.,11.,12.
            ],

            [
                6.,5.,4.,3.,2.,1.
            ],

            [
                3.,2.,1.,5.,6.,4.
            ]
        ]
        )
    )
    # torch.Size([2,6,6]) (b,t,t)
    feat = torch.tensor(
        (
            [
                [1.,2.,3.,2.,1.,0.],
                [1.,5.,4.,3.,2.,0.],
                [1.,2.,3.,4.,3.,2.],
                [1.,2.,3.,4.,5.,6.],
                [6.,5.,4.,3.,2.,1.],
                [1.,2.,3.,4.,5.,0.]
            ],
            [
                [1.,2.,3.,2.,1.,0.],
                [1.,5.,4.,3.,2.,0.],
                [1.,2.,3.,4.,3.,2.],
                [1.,2.,3.,4.,5.,6.],
                [6.,5.,4.,3.,2.,1.],
                [1.,2.,3.,4.,5.,0.]
            ]
        )
    )
   # (s,b,t,t)
    scores = feat.expand(4,2,6,6)

    # (b,s)
    tags = torch.tensor(
        (
            [  
                0,1,2,3
            ],
            [
                1,2,3,0
            ]

        )
    )


    crf = CRF(4,'cpu')
    mask = torch.ones(vec.shape)[:,:,0] > 0.5 #vec是(b,s,t)，mask是(b,s)
    # print(crf._viterbi_decode(vec,mask))
    # final_partition, scores = crf._calculate_PZ(vec,mask)
    # print(f'final is: {final_partition}')
    # print(f'scores is: {scores}')
    # print(log_sum_exp(feat,6))
    print(crf.neg_log_likelihood_loss(vec, mask, tags))