"""
@author: Zhouhong Gu
@idea:  1. 尝试用聚类结果做数据增强
            a. 根据聚类结果建一张词表
            b. 搜索训练集，如果找到相同的单词，就可以用类中的其他单词来增强
                i. 要建一个索引树，快速查找一句话中有无待匹配单词
                ii. 若有待匹配单词，则调用字典进行替换
"""
from fastHan import FastHan

from tools import gzh
import os
import random
import tqdm
from tools.adversrialAugmentation import AdversrialAugmentation


# ----------------------------------------------------------
def buildUpAugmentedData(rateSumThre, path='../clusterO', output='../DataSet/augmentedData.json'):
    '''
    建立词表
    :param rateSumThre:
    :return:
    '''
    po = gzh.readJson(gzh.cndbpedia_json)
    path = path
    output = output

    normalization = {}
    classes = 0
    for file1 in os.listdir(path):
        if not file1.endswith('.json'):
            continue
        name1 = file1.split('_')[0]
        num1 = int(file1.split('_')[1].strip('.json'))
        mark = False
        for file2 in os.listdir(path):
            if not file2.endswith('.json'):
                continue
            name2 = file2.split('_')[0]
            num2 = int(file2.split('_')[1].strip('.json'))
            if num1 < num2 and name2 == name1:
                mark = True
                break
        if mark:
            continue
        print(file1)
        data = gzh.readJson(os.path.join(path, file1))
        classes += len(data)
        # 频率之和高于某个阈值的类才会做归一化
        allNum = sum([i[1] for i in po[name1].items()])
        for key in data:
            if sum(po[name1][o] / allNum for o in data[key]) > rateSumThre:
                normalization[name1 + '_%d' % len(normalization)] = data[key].copy()
    gzh.toJson(normalization, output)
    print('共有%d个类' % classes)
    print('生成了%d种增强数据' % len(normalization))
    print('生成了%d种增强方案' % sum([len(normalization[i]) for i in normalization]))


aa = AdversrialAugmentation(augmentedPath='./DataSet/augmentedData.json')


# ----------------------------------------------------------
class augmentation:
    def __init__(self, augmentedDataPath='../DataSet/augmentedData.json'):
        self.trie = gzh.Trie()
        try:
            self.aug = gzh.readJson(augmentedDataPath)
            num = self.trie.add_dict(self.aug)
            print('字典树构建完成，加入%d待匹配词' % num)
        except:
            print('字典树构建失败')

    def addAugmentData(self, augmentedDataPath='../DataSet/augmentedData.json'):
        try:
            self.aug = gzh.readJson(augmentedDataPath)
            num = self.trie.add_dict(self.aug)
            print('字典树构建完成，加入%d待匹配词' % num)
        except:
            print('字典树构建失败')

    def augment(self, mission, data, po, augmentationRate, stopNum, func):
        '''
        获得增强后的数据集
        :param mission: 什么任务
        :param data: 被增强的数据
        :param po:
        :param augmentationRate: 增强多少
        :param stopNum: 增强前n个
        :param func: 哪种增强方式
        :return:
        '''
        assert func in ['pattern', 'replace']
        addedTrainData = []
        if func == 'pattern':
            '''
            模式增强
            pattern-level augmentation
            '''
            patterns = aa.getAttackPattern(1, po)
            model = FastHan(model_type='base')
            if mission == 'bq':
                for traindata in tqdm.tqdm(data[:stopNum]):
                    answers = model(traindata[:-1], 'POS')

                    one = []
                    two = []
                    for index, (dat, answer) in enumerate(zip(traindata[:-1], answers)):
                        for ans in answer:
                            if ans[1] == 'NN':
                                for pattern in patterns:
                                    pat = pattern.replace('[HEAD]', ans[0])
                                    if index == 0:
                                        one.append(dat.replace(ans[0], pat))
                                    elif index == 1:
                                        two.append(dat.replace(ans[0], pat))
                    for i in one:
                        for j in two:
                            addedTrainData.append([i, j, traindata[-1]])
            elif mission == 'tnews':
                for traindata in tqdm.tqdm(data[:stopNum]):
                    answers = model(traindata[:-1], 'POS')
                    one = []
                    for index, (dat, answer) in enumerate(zip(traindata[:-1], answers)):
                        for ans in answer:
                            if ans[1] == 'NN':
                                for pattern in patterns:
                                    pat = pattern.replace('[HEAD]', ans[0])
                                    one.append(dat.replace(ans[0], pat))
                    for i in one:
                        addedTrainData.append([i, traindata[-1]])
            elif mission == 'inews':
                data = [[i[0][:256], i[1]] for i in data]
                print(data[:10])
                for traindata in tqdm.tqdm(data[:stopNum]):
                    answers = model(traindata[:-1], 'POS')
                    one = []
                    for index, (dat, answer) in enumerate(zip(traindata[:-1], answers)):
                        for ans in answer:
                            if ans[1] == 'NN':
                                for pattern in patterns:
                                    pat = pattern.replace('[HEAD]', ans[0])
                                    one.append(dat.replace(ans[0], pat))
                    for i in one:
                        addedTrainData.append([i, traindata[-1]])
        elif func == 'replace':
            '''
            text-level augmentation
            替换增强
            '''
            noKey = set()
            count = {}
            if stopNum <= 0:
                stopNum = -1
            if mission in ['weibo','toutiao','sst']:
                for sent, label in tqdm.tqdm(data[:stopNum]):
                    # 找到一句话中涉及到了哪些语义类中的词汇
                    words = self.trie.search(sent)
                    if words:
                        for word in words:
                            # 获得语义类的标志词
                            key = self.trie.get_key(word[0])
                            # 忘了这个是干啥的了
                            if not key:
                                noKey.add(word[0])
                                continue
                            # 获得可以增强的样本个数
                            count[key] = count.get(key, 0) + 1
                            # 获得语义类
                            values = self.aug[key]
                            for value in values:
                                new_data = sent
                                new_data = new_data.replace(word[0], value)
                                addedTrainData.append([new_data, word[0], value, label])
            elif mission in ['qnli']:
                for sent1, sent2, label in tqdm.tqdm(data[:stopNum]):
                    for index, sent in enumerate([sent1, sent2]):
                        words = self.trie.search(sent)
                        if words:
                            for word in words:
                                key = self.trie.get_key(word[0])
                                if not key:
                                    noKey.add(word[0])
                                    continue
                                count[key] = count.get(key, 0) + 1
                                values = self.aug[key]
                                for value in values:
                                    new_data = sent
                                    new_data = new_data.replace(word[0], value)
                                    if index == 0:
                                        addedTrainData.append([new_data, sent2, word[0], value, label])
                                    else:
                                        addedTrainData.append([sent1, new_data, word[0], value, label])
            if noKey:
                print('未找到Key:', noKey)
            if count:
                print(count)
        f = open('addData.txt', 'w', encoding='utf-8')
        for dat in addedTrainData:
            f.write(str(dat) + '\n')
        f.close()
        if mission in ['weibo','toutiao','sst']:
            addedTrainData = [[i[0],i[1]] for i in addedTrainData]
        elif mission in ['qnli']:
            addedTrainData = [[i[0],i[1],i[2]] for i in addedTrainData]
        return random.sample(addedTrainData, min(len(addedTrainData), int(len(data) * augmentationRate)))


if __name__ == '__main__':
    # 超参数
    rateSumThre = 0.1

    path = '../resultOfKBC/clusterO'
    output = '../DataSet/augmentedData.json'

    buildUpAugmentedData(rateSumThre, path, output)
