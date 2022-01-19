import numpy as np
from tools import gzh

bert = gzh.bert_encoder()

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r'c:\windows\fonts\simsun.ttc', size=15)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 密度聚类
def getDB(key, values, embeds, name=''):
    '''
    密度聚类
    :param key: 属性名
    :param values: 属性值 list
    :param embeds: 属性值对应的embed list
    :return:
    '''
    value2embed = {}
    for value, embed in zip(values, embeds):
        value2embed[value] = embed

    # 密度聚类
    y_pred = DBSCAN(eps=0.1).fit_predict(embeds)
    dim = PCA(n_components=2).fit_transform(embeds)[:, :2]

    cluster = {}
    for i, j in zip(y_pred, values):
        if i != -1:
            li = cluster.get(i, [])
            li.append(j)
            cluster[i] = li
    clustered_num = 0
    for i in cluster:
        clustered_num += len(cluster[i])

    plt.clf()
    plt.figure(figsize=(18, 18))
    plt.title('%s密度聚类效果' % key)

    plt.scatter(dim[:, 0], dim[:, 1], c=y_pred)
    for w, (x, y) in zip(values, dim):
        # plt.text(x + 0.005, y + 0.005, w, font=font)
        plt.text(x + 0.005, y + 0.005, w)
    plt.savefig('./result/%s%s.png' % (key, name))

    new_cluster = {}
    for key in cluster:
        for index, i in enumerate(cluster[key]):
            cluster[key][index] = str(cluster[key][index])
        new_cluster[str(key)] = cluster[key]

    gzh.toJson(new_cluster, './result/%s.json' % key)
    return cluster


def getEmbedFre(query, po):
    values_lis = []
    fre_dic = {}
    char_dic = {}
    al = sum([i[1] for i in po[query].items()])
    for key in po[query]:
        values_lis.append(key)
        fre_dic[key] = po[query][key] / al
        for c in key:
            char_dic[c] = char_dic.get(c, 0) + po[query][key]
    embed_dic = {}
    for key in po[query]:
        outputs = bert(list(key))
        embed = np.zeros(outputs.shape[-1])
        for output, c in zip(outputs, key):
            embed += output * char_dic.get(c, 1)
        embed /= sum([char_dic.get(i, 1) for i in key])
        embed_dic[key] = embed
    return embed_dic, fre_dic, char_dic


def wc(embed_dic, fre_dic, line=5):
    '''
    embed_dic: key:属性值 value:属性值的embedding
    fre_dic: key:属性值 value:属性值的频率
    '''
    belong_dic = {}
    see = {}
    for key1 in embed_dic:
        max_value = [[0, 0]] * line
        max_score = [0] * line
        for key2 in embed_dic:
            if key2 == key1:
                continue
            sim = gzh.cosine(embed_dic[key1], embed_dic[key2])
            score = sim
            if score > max_score[-1]:
                max_score.append(score)
                max_value.append([key2, score])

            max_value = sorted(max_value, key=lambda x: x[1], reverse=True)[:line]
            max_score = sorted(max_score, reverse=True)[:line]

        mark = True
        see[key1] = max_value
        for value, score in max_value:
            if value == 0:
                continue
            if fre_dic[value] > fre_dic[key1]:
                belong_dic[key1] = value
                mark = False
                break
        if mark:
            belong_dic[key1] = key1
    return belong_dic, see


def cluster_o(embed_dic, fre_dic, line=10, epoch=8):
    assert epoch > 0
    final_belong = {}
    new_embed_dic = embed_dic
    new_fre_dic = fre_dic
    for i in range(epoch):
        belong_dic, see = wc(new_embed_dic, new_fre_dic, line)

        cluster = {}
        for key in belong_dic:
            li = cluster.get(belong_dic[key], [])
            li.append(key)
            cluster[belong_dic[key]] = li
        for key1 in cluster:
            for key2 in cluster[key1]:
                final_belong[key2] = key1
        temp_embed_dic = {}
        for key in cluster.keys():
            temp_embed_dic[key] = new_embed_dic[key]
        new_embed_dic = temp_embed_dic
        # print('第%d轮迭代，还剩下%d个类' % (i + 1, len(new_embed_dic.keys())))
    return final_belong, list(belong_dic.keys())
