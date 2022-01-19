"""
@author: Zhouhong Gu
@idea:  1. 利用聚类的分布扰动去增强样本
            a. 计算所有embedding之间的差，然后取平均
            b. 计算所有embedding到最高频属性值之间的差，然后取平均
"""
import numpy as np
from tools import gzh
import random


class AdversrialAugmentation:
    def __init__(self, augmentedPath='../DataSet/augmentedData.json'):

        self.augmentedData = gzh.readJson(augmentedPath)
        self.bert_name = 'bert'
        self.encoder = gzh.bert_encoder(self.bert_name)

    def getAttackEmbedding(self, num, scale, po, function):
        '''
        获得攻击样本和embedding
        :param num:
        :param scale:
        :param po:
        :param function:
        :return:
        '''
        keys = random.sample(self.augmentedData.keys(), min(len(self.augmentedData), num))
        # 攻击embedding
        attackEmbedding = np.zeros(768)
        num = 0
        for key in keys:
            pp = key.split('_')[0]
            if function == 0:
                # 找到类内最高频的属性值
                core_value = ''
                temp = 0
                for value in self.augmentedData[key]:
                    if po[pp][value] > temp:
                        core_value = value
                        temp = po[pp][value]
                del (temp)
                # 得到最高频属性值的embedding
                core_embedding = self.encoder.forward(core_value)[0]
                # 获得所有属性值的embedding
                a = self.augmentedData[key].copy()
                a.remove(core_value)
                embeddings = self.encoder.forward(a)
                # 计算扰动
                attackEmbedding += np.sum(np.tile(core_embedding, [len(a), 1]) - embeddings, axis=0)
                num += len(embeddings)
            elif function == 1:
                embeddings = self.encoder.forward(self.augmentedData[key])
                for index, embedding in enumerate(embeddings):
                    # 计算扰动
                    attackEmbedding += np.sum(
                        np.tile(embedding, [len(embeddings) - index - 1, 1]) - embeddings[index + 1:],
                        axis=0)
                    num += len(embeddings) - index - 1
        return attackEmbedding * scale / num

    def getAttackPattern(self, num, po):
        keys = random.sample(self.augmentedData.keys(), min(len(self.augmentedData), num))
        patterns = []
        for key in keys:
            pp = key.split('_')[0]
            oo = self.augmentedData[key]
            # 统计字频
            # char_fre = gzh.getCharFre(po, pp, oo)
            # 找到类内最高频的
            # allNum = sum([po[pp][o] for o in oo])
            head = sorted(oo, key=lambda x: po[pp][x], reverse=True)[0]
            for o in oo:
                if head in o:
                    pat = o.replace(head, '[HEAD]')
                    for c in pp:
                        pat = pat.replace(c, '')
                    if pat.replace('[HEAD]', ''):
                        patterns.append(pat)
        print('原始patterns长度:%d' % len(patterns))
        print('取set后patterns长度:%d' % len(set(patterns)))
        return set(patterns)


if __name__ == '__main__':
    po = gzh.readJson(gzh.cndbpedia_json)
    aa = AdversrialAugmentation()
    patterns = aa.getAttackPattern(2, po)
    '''
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    template = lambda x, y: '%d是%d。' % (x, y)
    key = random.sample(self.augmentedData.keys(), 1)[0]
    pp = key.split('_')[0]
    values = self.augmentedData[key]
    embeddings = self.encoder.forward(values)

    xs = embeddings[:, 1]
    ys = embeddings[:, 0]
    plt.scatter(xs, ys)
    for w, x, y in zip(values, xs, ys):
        plt.text(x + 0.005, y + 0.005, w)
    plt.savefig('./DataSet/img/%s.png' % pp)
    plt.show()
    '''

    '''
    # 超参数
    num = 2
    scale = 1
    function = 1
    po = gzh.readJson(gzh.cndbpedia_json)
    output = getAttackEmbedding(num, scale, po, function)
    '''
