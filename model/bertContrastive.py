"""
@author: Zhouhong Gu
@idea:  利用SiameseNet做对比学习
"""

import torch, tqdm
import torch.nn as nn
import torch.optim as optim

from pytorch_transformers import BertModel, BertTokenizer
from tools import gzh

import matplotlib.pyplot as plt
import random


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(gzh.bert_model)
        self.bert = BertModel.from_pretrained(gzh.bert_model)
        self.fn = nn.Linear(768, 768)
        self.fn.to('cuda')
        self.bert.to('cuda')

    def forward(self, value_list, max_len=5):
        tokens = [self.tokenizer.convert_tokens_to_ids(i)[:max_len] + [0] * (max_len - len(i)) for i in
                  [self.tokenizer.tokenize(j) for j in value_list]]
        output1, output2 = self.bert(torch.tensor(tokens, device='cuda'))[1]
        output1 = self.fn(output1)
        output2 = self.fn(output2)
        cosine = torch.cosine_similarity(output1, output2, 0)
        return cosine

    def getEmbed(self, value_list, max_len=5):
        tokens = [self.tokenizer.convert_tokens_to_ids(i)[:max_len] + [0] * (max_len - len(i)) for i in
                  [self.tokenizer.tokenize(j) for j in value_list]]
        output = self.bert(torch.tensor(tokens, device='cuda'))[1]
        output = self.fn(output)
        return output.detach().cpu().numpy()


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, Ew, y, m=0.3):
        l_1 = 0.25 * (1.0 - Ew) * (1.0 - Ew)
        l_0 = torch.where(Ew < m * torch.ones_like(Ew), torch.full_like(Ew, 0), Ew) * torch.where(
            Ew < m * torch.ones_like(Ew), torch.full_like(Ew, 0), Ew)

        loss = y * 1.0 * l_1 + (1 - y) * 1.0 * l_0
        return loss.sum()


def train(texts, epoch, model, criterion, optimizer):
    '''

    :param data: one,two,label
    :param model:
    :param criterion:
    :param optimizer:
    :return:
    '''
    losses = []
    for _ in tqdm.tqdm(range(epoch)):
        epoch_loss = 0
        for text in texts:
            label = text[2]
            text = text[:2]
            sim = model(text)
            loss = criterion(sim.reshape(1), torch.FloatTensor([label]).to('cuda'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
    return losses


def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)  # 字符串转list
    str_list.insert(pos, str_add)  # 在指定位置插入字符串
    str_out = ''.join(str_list)  # 空字符连接
    return str_out


def addNoisy(word, noisy, times=5):
    no = ''.join(noisy)
    for _ in range(times):
        num = random.random() * len(no)
        no = random.sample(no, int(num))
    num = random.random() * len(no)
    no = str_insert(no, int(num), word)
    return no


def getAugmentedData(key1, key2, noisy, size=5):
    one = [key1]
    zero = [key2]
    for i in noisy:
        key1_no = addNoisy(key1, noisy)
        key2_no = addNoisy(key2, noisy)
        one.append(key1_no)
        zero.append(key2_no)
    for i in noisy:
        i = random.sample(noisy,1)[0]
        j = random.sample(noisy,1)[0]
        temp1 = i + key1 + j
        temp2 = i + key2 + j
        one.append(temp1)
        zero.append(temp2)
    texts = []
    for i1, j1 in zip(one, zero):
        for i2, j2 in zip(one, zero):
            texts.append([i1, i2, 1])
            texts.append([j1, j2, 1])
            texts.append([i1, j1, 0])
            texts.append([i2, j2, 0])
    texts = random.sample(texts, 4 * size)
    return texts


if __name__ == '__main__':
    lr = 1e-2
    criterion = ContrastiveLoss()
    encoder = Encoder()
    optimizer = optim.SGD(encoder.parameters(), lr=lr)

    noisy = ['.', '>', '。', ',', '<', '，']
    key1 = '男'
    key2 = '女'

    texts = getAugmentedData(key1, key2, noisy)
    losses = train(texts, 20, encoder, criterion, optimizer)
    plt.plot(losses)
    plt.show()
    text = ['男', '女']
    embed1, embed2 = encoder.getEmbed(text)
    print(torch.cosine_similarity(torch.tensor(embed1), torch.tensor(embed2), dim=0))

    torch.save(encoder, './Contrastive.json')
