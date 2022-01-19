"""
@author: Zhouhong Gu
@idea:     利用数据增强模式去尝试着增强一些内容，对比增强之后的准确率
"""
import os, sys
import config
from tools.dataAugmentation import augmentation
from tools import gzh
import time, tqdm
import pandas as pd
from model.getModel import getModel, trainModel, testModel

# 读取记录
print('正在读取历史记录')
resultPath = './DataSet/result/result.xls'
try:
    excel = pd.read_excel(resultPath)
    excel = excel[excel.columns[1:]]
except:
    excel = pd.DataFrame(
        columns=['Mission', 'func', 'Time', 'augment rate', 'epoch', 'lr', 'batch size', 'acc未加强', 'pre未加强',
                 'recall未加强',
                 'f1未加强', 'acc加强后', 'pre加强后', 'recall加强后',
                 'f1加强后'])
    excel.to_excel(resultPath)
checkpoint = {}
t = ''
for i in time.localtime()[:-3]:
    t += str(i)
checkpoint['Time'] = t

po = gzh.readJson(gzh.cndbpedia_json)

par = config.augmentTestPar()

# ----------------------------------------------------------------------
# 超参数
print('正在读取超参数')
lr = par.lr
train_batch_size = par.train_batch_size
test_batch_size = par.test_batch_size
EPOCH = par.EPOCH
mission = par.mission
func = par.func
augmentationRate = par.augmentationRate
stopNum = par.stopNum
checkpoint['epoch'] = EPOCH
checkpoint['lr'] = lr
checkpoint['batch size'] = train_batch_size
checkpoint['func'] = func
checkpoint['augment rate'] = augmentationRate


# ----------------------------------------------------------------------


# 数据集

def generateDataset(dataset_path, mission):
    if mission in ['toutiao', 'weibo', 'sst']:
        train = gzh.readJson(os.path.join(dataset_path,'train.json'))
        label_size = len(set([i['label'] for i in train]))
        trainData = [[i['sentence'], gzh.getDummies(i['label'],size=label_size)] for i in train]
        del (train)

        dev = gzh.readJson(os.path.join(dataset_path,'dev.json'))
        testData = [[i['sentence'], gzh.getDummies(i['label'],size=label_size)] for i in dev]
        del (dev)
    if mission in ['qnli']:
        train = gzh.readJson(os.path.join(dataset_path,'train.json'))
        label_size = len(set([i['label'] for i in train]))
        trainData = [[i['sentence'], i['question'], gzh.getDummies(i['label'],size=label_size)] for i in train]
        del (train)

        dev = gzh.readJson(os.path.join(dataset_path,'dev.json'))
        testData = [[i['sentence'], i['question'], gzh.getDummies(i['label'],size=label_size)] for i in dev]
        del (dev)
    return trainData, testData, label_size

print('正在读取数据集')
dataSetPath = './DataSet/data/%s' % mission
trainData, testData, label_size = generateDataset(dataSetPath, mission)
print('训练集数量：%d' % (len(trainData)))
print('测试集数量：%d' % (len(testData)))
aug = augmentation(augmentedDataPath='./DataSet/augmentedData.json')
addedTrainData = aug.augment(mission, trainData, po, augmentationRate, stopNum, func)
print(addedTrainData[:5])
print('添加了%d个增强数据' % len(addedTrainData))

# ----------------------------------------------------------------------
# 不加入数据增强
print('开始训练不加入数据增强的任务')
# 模型
model, criterion, optimizer = getModel(mission, lr, size=label_size)

losses = []
# 训练
for epoch in tqdm.tqdm(range(EPOCH)):
    epoch_loss = trainModel(mission, trainData, model, criterion, optimizer, train_batch_size, False)
    losses.append(epoch_loss)

print(losses)
answers, tp, fp, tn, fn = testModel(mission, testData, model, test_batch_size)
try:
    num = (tp + tn + fp + fn)
    acc = (tp + tn) / num
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * pre * recall / (pre + recall)
except Exception as e:
    print(tp, fp, tn, fn)
    print(e)
    sys.exit()
aa = '不加入增强的数据\t准确率%.4f，f1:%.4f' % (acc, f1)
checkpoint['acc未加强'] = acc
checkpoint['pre未加强'] = pre
checkpoint['recall未加强'] = recall
checkpoint['f1未加强'] = f1
print(aa)
# ----------------------------------------------------------------------
# 加入数据增强
print('开始训练加入数据增强的任务')
# 模型
del (model)
del (criterion)
del (optimizer)
model, criterion, optimizer = getModel(mission, lr, label_size)
losses = []
# 训练
for epoch in tqdm.tqdm(range(EPOCH)):
    epoch_loss = trainModel(mission,
                            trainData + addedTrainData, model,
                            criterion, optimizer, train_batch_size, False)
    losses.append(epoch_loss)

print(losses)
answers, tp, fp, tn, fn = testModel(mission, testData, model, test_batch_size)
num = (tp + tn + fp + fn)
acc = (tp + tn) / num
pre = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * pre * recall / (pre + recall)
aa = '加入增强的数据\t准确率%.4f，f1:%.4f' % (acc, f1)
checkpoint['acc加强后'] = acc
checkpoint['pre加强后'] = pre
checkpoint['recall加强后'] = recall
checkpoint['f1加强后'] = f1
print(aa)

checkpoint['Mission'] = mission
excel = excel.append(checkpoint, ignore_index=True)
excel.to_excel(resultPath)
