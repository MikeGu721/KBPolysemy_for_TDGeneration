"""
@author: Zhouhong Gu
@idea: 选择p
"""

import gzh, tqdm
import numpy as np
import pandas as pd
import config

import re


# 计算只有中文的部分
def getChinese(word):
    return ''.join(re.findall(u'[\u4e00-\u9fa5]', word))


# 计算中文字符占比
def getChinRate(dic: dict):
    a = 0
    b = 0
    for value in dic:
        value_num = dic[value]
        a += len(value) * value_num
        text = getChinese(value)
        b += len(text) * value_num
    return b / a


# 计算交叉熵
def H(di: dict):
    al = sum([i[1] for i in di.items()])
    return al, sum(-np.log2(i[1] / al) * i[1] / al for i in di.items())


# 计算词频
def crossPhrase(di: dict):
    phrases = {}
    for key in di:
        for span in range(0, len(key)):
            span += 1
            for start in range(0, len(key) - span + 1):
                phrase = key[start:start + span]
                phrases[phrase] = phrases.get(phrase, 0) + di[key]
    return phrases


if __name__ == '__main__':

    # 超参数
    par = config.choosePPar()

    min_num = par.min_num
    cha_rate = 0.9
    output_path = config.chooseP_output
    knowledgeGraph = config.knowledgeGraph
    excel_name = par.excel_name

    # 获得知识图谱
    po = gzh.readJson(knowledgeGraph)

    # 计算所有属性的香农熵，以及中文字符占比
    hh = {}
    for key in tqdm.tqdm(po):
        a, b = H(po[key])
        if a > min_num:
            num = getChinRate(po[key])
            if num < cha_rate:
                continue
            hh[key] = b

    # 按照香农熵的大小进行升序排列
    s_hh = sorted(hh.items(), key=lambda x: x[1])

    # 取前100个最小香农熵的属性，计算字符切片香农熵
    hhh = {}
    for key in s_hh[:100]:
        ent = key[1]
        key = key[0]
        str_fra = crossPhrase(po[key])
        c, str_fra_ent = H(str_fra)
        str_fra_num = len(str_fra)
        # H(value) * H(string fragment) / log(number of fragment)
        hhh[key] = [ent, str_fra_ent, str_fra_num, ent * str_fra_ent * str_fra_ent * np.log(str_fra_num)]
    # 生成excel
    excel = pd.DataFrame(
        columns=['key', 'value entropy', 'string fragment entropy', 'number of string fragment', 'metric'])
    # 保存各项指标
    for key in hhh:
        dic = {}
        dic['key'] = key
        dic['value entropy'] = hhh[key][0]
        dic['string fragment entropy'] = hhh[key][1]
        dic['number of string fragment'] = hhh[key][2]
        dic['metric'] = hhh[key][3]
        excel = excel.append(dic, ignore_index=True)
    excel.to_excel(excel_name)
