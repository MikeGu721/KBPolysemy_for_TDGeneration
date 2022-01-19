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

import config, random, os
from tools import gzh
from model import bertClassifier
from model.bertClassifier import fn_cls
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import time


def getDummies(label, size=2):
    answer = [0] * size
    answer[label] = 1
    return answer


def getOneSample(pp, po, need_iter, threhold=0):
    '''
    正采样
    :param pp: 属性
    :param po: 知识图谱的字典
    :param need_iter: 是否需要迭代
    :return:
    '''
    # 迭代采样
    if need_iter == True and pp + '.txt' in os.listdir(output_path):
        index = 0
        for file in os.listdir(output_path):
            if pp in file:
                index += 1
        if index <= 1:
            f = open(os.path.join(output_path, pp + '.txt'), encoding='utf-8')
        else:
            f = open(os.path.join(output_path, pp + '_%d.txt' % index), encoding='utf-8')
        one_sample = [i.strip().split('\t')[0] for i in f if
                      float(i.strip().split('\t')[2]) - float(i.strip().split('\t')[1]) > threhold]
    # 不迭代
    else:
        one_sample = list(set(po[pp].keys()))

    return one_sample


def getZeroSample(pp, po, length):
    '''
    负采样
    :param pp: 属性
    :param po: 知识图谱的字典
    :param length: 负例大小
    :return:
    '''
    zero_sample = set()
    while (len(zero_sample) < length):
        key_li = random.sample(list(po.keys()), length * p_range)
        for key in key_li:
            if key == pp:
                continue
            value_li = random.sample(list(po[key].keys()), min(length * o_range, len(list(po[key].keys()))))
            for value in value_li:
                zero_sample.add(value)
    zero_sample = random.sample(list(zero_sample), length)
    return zero_sample


def getWritingFile(pp, need_iter):
    '''
    获得写入文件
    :param pp: 属性
    :param need_iter: 是否需要迭代
    :return:
    '''
    if need_iter:
        index = 1
        for file in os.listdir(output_path):
            if pp == file.split('_')[0]:
                index += 1
        name = os.path.join(config.chooseO_output, '%s_%d.txt' % (pp, index))
        f = open(name, 'w', encoding='utf-8')
    else:
        index = 1
        name = os.path.join(config.chooseO_output, '%s_%d.txt' % (pp, index))
        f = open(name, 'w', encoding='utf-8')
    print('%s写入: %s' % (pp, name))
    return f


# 超参数

# 是否进行迭代

par = config.chooseOPar()
cp_par = config.choosePPar()

# need_iter = False
need_iter = True

output_path = config.chooseO_output
knowledgeGraph = config.knowledgeGraph
p_range = par.p_range
o_range = par.o_range
EPOCH = par.EPOCH
train_batch_size = par.train_batch_size
test_batch_size = par.test_batch_size
learning_rate = par.learning_rate
highest_rate = par.highest_rate
use_template = par.use_template
save_model = par.save_model
threhold = par.threhold

# 根据选择P的结果选择O
excel = pd.read_excel(cp_par.excel_name)
metric_threhold = par.metric_threhold
properties = []
for idd, property, one, two, three, four in iter(excel.values):
    if four < metric_threhold:
        properties.append(property)

po = gzh.readJson(gzh.cndbpedia_json)

start = time.time()

# 指定一些P进行选择O
# properties = ['性别', '色彩', '小说进度', '纸张']

print(properties)

for property in properties:
    property = '性别'
    # 找到计算最高的95%的属性值
    allNum = sum([i[1] for i in po[property].items()])
    bigest_o = []
    rate = highest_rate
    sorted_value = sorted(po[property].items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_value:
        bigest_o.append(name)
        rate -= value / allNum
        if rate <= 0:
            break
    print('---' * 20)
    print('%s %d %.2f' % (property, allNum, time.time() - start))
    print(bigest_o)

    # 正采样
    one_sample = getOneSample(property, po, need_iter,threhold)

    # 负采样
    zero_sample = getZeroSample(property, po, len(one_sample))

    # 获得文件读写地址
    f = getWritingFile(property, need_iter)

    num = int(len(one_sample) / 3)

    one_sample = one_sample + one_sample
    zero_sample = zero_sample + zero_sample

    template_length = len(par.getTemplate(p='', o='', label=0))

    for epoch in range(3):
        epoch_start = time.time()
        start = epoch * num
        end = (epoch + 2) * num
        train = [
            [par.getTemplate(p=property, o=i, label=getDummies(1))] if index < len(one_sample[start:end]) else [
                par.getTemplate(p=property, o=i, label=getDummies(0))] for
            index, i in
            enumerate(one_sample[start:end] + zero_sample[start:end])]

        temp = []
        for i in train:
            for j in i:
                temp.extend(j)
        train = temp
        del (temp)

        start = (epoch + 2) * num
        end = (epoch + 3) * num
        test_ori = [i for i in one_sample[start:end]]
        test = [par.getTemplate(property, i, getDummies(0)) for i in one_sample[start:end]]

        temp = []
        for i in test:
            temp.extend(i)
        test = temp
        del (temp)
        model = fn_cls()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        # 训练过程
        epoch_loss = bertClassifier.trainClassifier(train, model, criterion, optimizer, train_batch_size,
                                                    save_model)
        print('%d Loss: %.4f spend Time: %.4f' % (epoch + 1, epoch_loss, time.time() - epoch_start))

        answers, _, _, _, _ = bertClassifier.predictClassifier(test, model, test_batch_size)
        scores = []
        for index, ((sentence, label), output) in enumerate(zip(test, answers)):
            scores.append(output)
            # f.write(sentence + '\t' + str(output[0]) + '\t' + str(output[1]) + '\n')
            if len(scores) == template_length:
                zero_score = max([float(i[0]) for i in scores])
                one_score = sum([float(i[1]) for i in scores]) / len(scores)
                oo = test_ori[int(index / template_length)]
                if oo in bigest_o:
                    zero_score = 0
                    one_score = 1
                f.write(
                    str(test_ori[int(index / template_length)]) + '\t' + str(zero_score) + '\t' + str(one_score) + '\n')
                # f.write('---' * 20 + '\n')
                scores = []
    f.close()
    break
