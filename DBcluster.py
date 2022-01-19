"""
@author: Zhouhong Gu

@idea:
    1. 简单来讲，就是通过大部分去预测少部分
        a. 文件读取
            i. 如果不需要迭代，就直接新建一个文件夹，并选择所有属性值进行下一轮的预测
            ii. 如果需要迭代，查找该属性下最新更新的下标，并选择其中的正例进行下一轮的预测
        b. 针对每次预测，要重新训练分类器
            i. 每次使用2/3个属性值去预测1/3个属性值
            ii. 预测完之后将频率前95%的属性值修改为'真'
        c. 将结果输出到文件
"""

import torch, os
from tools import gzh
import torch.nn as nn
from transformers import AlbertModel, AlbertTokenizer

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import config
import warnings
warnings.filterwarnings("ignore")

font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=15)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class embed(nn.Module):
    def __init__(self, method):
        super(embed, self).__init__()
        self.method = method
        assert method in ['random', 'bert_encoder', 'bert_word']
        if method == 'random':
            self.embed = nn.Embedding(27012, 768)
        elif method == 'bert_encoder':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif method == 'bert_word':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')

    def forward(self, word, char2id, fre):
        '''
        获得word的向量
        :param word:  str
        :param fre: 字频
        :return:
        '''
        if self.method == 'random':
            vector = torch.zeros(1, 768)
            num = 0
            for c in word:
                num += fre[c]
                vector += self.embed(torch.tensor(char2id[c])) * fre[c]
            vector /= num
            return vector
        elif self.method == 'bert_encoder':
            word = word.tolist()
            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to('cuda')
            outputs = self.model(**inputs)[1]
            return outputs
        elif self.method == 'bert_word':
            vector = torch.zeros(1, 768)
            word = word.tolist()
            inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True)
            for key in inputs:
                inputs[key] = inputs[key].to('cuda')
            outputs = self.model(**inputs)[1]
            num = 0
            for w, output in zip(word, outputs):
                num += fre[w]
                vector += output * fre[w]
            vector /= num
            return vector


output = config.clusterO_output
chooseOPath = config.chooseO_output

par = config.clusterOPar()
eps = par.eps
threhold = 0.0001
# embed = nn.Embedding(27012, 2)
Embed = embed(method='random')

po = gzh.readJson(gzh.cndbpedia_json)

# 遍历chooseO输出的内容
for file in os.listdir(chooseOPath):
    if file.endswith('xls'):
        continue
    pp = file.split('_')[0]
    num = int(file.strip('.txt').split('_')[-1])
    # 寻找有没有更大的下标
    mark = False
    for file_ in os.listdir(chooseOPath):
        if pp == file_.split('_')[0] and num < int(file_.strip('.txt').split('_')[-1]):
            mark = True
            break
    if mark:
        continue
    try:
        f = open(os.path.join(chooseOPath, file), encoding='utf-8')
    except:
        print('not find %s' % file)
        continue
    # 读取文件，获得对和错的属性值
    right = []
    false = []
    for line in iter(f):
        name, one, two = line.strip().split('\t')
        if float(one) - float(two) > threhold:
            false.append(name)
        else:
            right.append(name)
    print('属性值:%s，正确个数:%d，错误个数:%d' % (pp, len(right), len(false)))
    # 删除false，节省内存
    del (false)

    # 计算词频
    charF = {}
    for value in right:
        for c in value:
            charF[c] = charF.get(c, 0) + po[pp].get(value, 1)
            if c in pp:
                charF[c] = 1
            # charF[c] = charF.get(c, 0) + 1

    # 获得字符id，以及计算embedding
    value2embed = {}
    char2id = {}
    for key in right:
        for c in key:
            char2id[c] = char2id.get(c, len(char2id))
        emb = Embed(key, char2id, charF)
        value2embed[key] = emb.detach().numpy().reshape(-1)

    # 降维，可视化
    embeds = []
    values = []
    for key in value2embed:
        values.append(key)
        embeds.append(value2embed[key])
    # 先聚类
    y_pred = DBSCAN(eps=eps).fit_predict(embeds)
    # 然后降维
    dim = PCA(n_components=2).fit_transform(embeds)[:, :2]
    # 获得类别总数
    classes = set([i for i in y_pred])

    # 获得聚类信息
    cluster = {}
    for i, j in zip(y_pred, values):
        if i != -1:
            li = cluster.get(i, [])
            li.append(j)
            cluster[i] = li

    # 画图以及保存
    plt.clf()
    plt.figure(figsize=(9, 9))
    plt.title('%s密度聚类效果' % file.strip('.txt'))
    plt.scatter(dim[:, 0], dim[:, 1], c=y_pred)
    for w, (x, y) in zip(values, dim):
        plt.text(x + 0.005, y + 0.005, w)
    plt.savefig(os.path.join(output, '%s.png' % file.strip('.txt')))

    # 保存聚类信息
    new_cluster = {}
    for key in cluster:
        for index, i in enumerate(cluster[key]):
            cluster[key][index] = str(cluster[key][index])
        new_cluster[str(key)] = cluster[key]
    gzh.toJson(new_cluster, os.path.join(output, '%s.json' % file.strip('.txt')))
