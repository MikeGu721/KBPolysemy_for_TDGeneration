from model import bertContrastive
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties
from tools import clusterO

font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=15)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 获得正确的属性值
def getRightValue(key='性别', d_value=0):
    f = open('./output/%s.txt' % key, encoding='utf-8')
    right = set()
    for line in iter(f):
        value, zero, one = line.strip().split('\t')
        if float(one) - float(zero) > d_value:
            right.add(value)
    return list(right)


def getF(key, po, values, choose=0.95):
    f = {}
    al = sum([i[1] for i in po[key].items()])
    for value in values:
        f[value] = po[key][value] / al
    f = sorted(f.items(), key=lambda x: x[1], reverse=True)
    choosen_value = []
    now = 0
    for value, num in f:
        choosen_value.append(value)
        now += num
        if now > choose:
            break
    return choosen_value


def getCharF(key, po):
    charF = {}
    al = 0
    for value in po[key]:
        al += len(value) * po[key][value]
    for value in po[key]:
        for c in value:
            charF[c] = charF.get(c, 0) + po[key][value]
    for char in charF:
        charF[char] = charF[char] / al
    return charF


if __name__ == '__main__':
    from tools import gzh
    import tqdm

    print('读取知识图谱中')
    po = gzh.readJson(gzh.cndbpedia_json)
    for key in ['界', '正文语种', '政治面貌', '民族', '所属洲', '球场位置', '游戏产地', '装帧', '词性', '婚姻情况']:
        values = getRightValue(key)
        # 得到词频，取最高的95%属性值用于增强
        choosen_values = getF(key, po, values)
        print('选中了%d个属性值' % len(choosen_values))
        # 生成数据增强的噪音
        charF = getCharF(key, po)
        noisy_rate = 1e-3
        charF = sorted(charF.items(), key=lambda x: x[1], reverse=False)

        noisy = ['.', '>', '。', ',', '<', '，']
        choose_num = 0
        for char, num in charF:
            noisy.append(char)
            choose_num += num
            if choose_num > noisy_rate:
                break
        noisy = noisy[:100]
        print('选中了%d个噪音' % len(noisy))
        # 对比学习
        texts = []
        for index, key1 in tqdm.tqdm(enumerate(choosen_values)):
            for key2 in choosen_values[index:]:
                if key2 == key1:
                    continue
                new_texts = bertContrastive.getAugmentedData(key1, key2, noisy)
                for text in new_texts:
                    texts.append(text)
        print(texts[:10])
        lr = 1e-3
        criterion = bertContrastive.ContrastiveLoss()
        encoder = bertContrastive.Encoder()
        optimizer = optim.SGD(encoder.parameters(), lr=lr)

        losses = bertContrastive.train(texts, 20, encoder, criterion, optimizer)
        plt.clf()
        plt.plot(losses)
        plt.show()
        plt.clf()
        embeds = encoder.getEmbed(values)
        cluster = clusterO.getDB(key, values, embeds, name='_2')
        pairs = {}
        for value1 in choosen_values:
            pairs[value1] = {}
            for value2 in values:
                if value2 == value1:
                    continue
                sim = encoder([value1, value2]).detach().cpu().numpy()
                pairs[value1][value2] = str(sim)
        # print(losses)
        # print(value1, value2, pairs[value1][value2])
        gzh.toJson(pairs, './result/%s_pair.json' % key)
