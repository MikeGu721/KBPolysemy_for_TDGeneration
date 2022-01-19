cndbpedia = 'D:\guzhouhong\科研\数据集\cndbpedia.parses.txt'
cndbpedia_json = 'D:\guzhouhong\科研\数据集\cndbpedia.json'
dbpedia = 'D:\guzhouhong\科研\数据集\dbpedia.parses.txt'
dbpedia_json = 'D:\guzhouhong\科研\数据集\dbpedia.json'
wwm_bert_model = r'D:\公用数据\tfhub\chinese_roberta_wwm_ext_L-12_H-768_A-12'
bert_model = r'D:\公用数据\bert_语言模型\chinese_L-12_H-768_A-12'

from collections import defaultdict
import json
from pytorch_transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch


class bert_encoder:
    def __init__(self, bert_name='bert', device='cuda'):
        super(bert_encoder, self).__init__()
        if bert_name == 'bert':
            bert_path = bert_model
        self.bert = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert.to(device)
        self.device = device

    def forward(self, sentence, max_len=24):
        tokens = [self.tokenizer.tokenize(str(i)) for i in sentence]
        tensor = [self.tokenizer.convert_tokens_to_ids(t)[:max_len] + [0] * max(0, max_len - len(t)) for t in tokens]
        outputs = self.bert(torch.tensor(tensor).to(self.device))[1]
        return outputs.detach().cpu().numpy()


'''class bert_encoder(nn.Module):  # aaa
    def __init__(self, device='cuda'):
        super(bert_encoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        self.bert.to(device)
        self.device = device

    def forward(self, text_lis: list, max_len=10):
        token = [self.tokenizer.tokenize(text) for text in text_lis]
        token = [self.tokenizer.convert_tokens_to_ids(i)[:max_len] + [0] * max(0, max_len - len(i)) for i in token]
        output = self.bert(torch.tensor(token).to(self.device))[1]
        return output.detach().cpu().numpy()'''


# 计算各种指标
def getMetrices(tp, fp, tn, fn):
    num = (tp + tn + fp + fn)
    acc = (tp + tn) / num
    pre = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * pre * recall / (pre + recall)
    return acc, pre, recall, f1


# 写json
def readJson(path):
    f = open(path, encoding='utf-8')
    a = json.load(f)
    f.close()
    return a


# 读取json
def toJson(dic, path):
    f = open(path, 'w', encoding='utf-8')
    jsonData = json.dumps(dic, indent=4, ensure_ascii=False)
    f.write(jsonData)
    f.close()


import numpy as np


def getCharFre(po, key, values=None):
    char_fre = {}
    if not values:
        values = po[key].keys()
    allNum = sum([len(value) * int(po[key].get(value, 1)) for value in values])
    for value in values:
        for c in value:
            char_fre[c] = char_fre.get(c, 0) + int(po[key].get(value, 1)) / allNum
    return char_fre


def cosine(v_1, v_2):
    tf = np.dot(v_1, v_2)
    idf = np.sqrt(np.dot(v_1, v_1) * np.dot(v_2, v_2))
    return tf / idf


def getDummies(label, size=2):
    a = [0] * size
    a[label] = 1
    return a


# 字典树
class TrieNode(object):
    def __init__(self, value=None):
        # 值
        self.value = value
        # fail指针
        self.fail = None
        # 尾标志：标志为i表示第i个模式串串尾，默认为0
        self.tail = 0
        # 子节点，{value:TrieNode}
        self.children = {}


class Trie(object):
    def __init__(self, words=[]):
        # 根节点
        self.root = TrieNode()
        # 模式串个数
        self.count = 0
        self.words = words
        self.keys = {}
        for word in words:
            self.insert(word)
        self.ac_automation()

    def add_words(self, words):
        self.words.extend(words)
        for word in words:
            self.insert(word)
        self.ac_automation()

    def add_dict(self, dic: dict):
        words = []
        for key, values in dic.items():
            words.extend(values)
            for value in values:
                self.keys[str(value)] = key
        print(words[:10])
        self.add_words(words)
        return len(words)

    def get_key(self, word: str):
        return self.keys.get(word, None)

    def insert(self, word):
        """
        基操，插入一个字符串
        :param word: 字符串
        :return:
        """
        self.count += 1
        cur_node = self.root
        for char in word:
            if char not in cur_node.children:
                # 插入结点
                child = TrieNode(value=char)
                cur_node.children[char] = child
                cur_node = child
            else:
                cur_node = cur_node.children[char]
        cur_node.tail = self.count

    def ac_automation(self):
        """
        构建失败路径
        :return:
        """
        queue = [self.root]
        # BFS遍历字典树
        while len(queue):
            temp_node = queue[0]
            # 取出队首元素
            queue.remove(temp_node)
            for value in temp_node.children.values():
                # 根的子结点fail指向根自己
                if temp_node == self.root:
                    value.fail = self.root
                else:
                    # 转到fail指针
                    p = temp_node.fail
                    while p:
                        # 若结点值在该结点的子结点中，则将fail指向该结点的对应子结点
                        if value.value in p.children:
                            value.fail = p.children[value.value]
                            break
                        # 转到fail指针继续回溯
                        p = p.fail
                    # 若为None，表示当前结点值在之前都没出现过，则其fail指向根结点
                    if not p:
                        value.fail = self.root
                # 将当前结点的所有子结点加到队列中
                queue.append(value)

    def search(self, text):
        """
        模式匹配
        :param self:
        :param text: 长文本
        :return:
        """
        p = self.root
        # 记录匹配起始位置下标
        start_index = 0
        # 成功匹配结果集
        rst = defaultdict(list)
        for i in range(len(text)):
            single_char = text[i]
            while single_char not in p.children and p is not self.root:
                p = p.fail
            # 有一点瑕疵，原因在于匹配子串的时候，若字符串中部分字符由两个匹配词组成，此时后一个词的前缀下标不会更新
            # 这是由于KMP算法本身导致的，目前与下文循环寻找所有匹配词存在冲突
            # 但是问题不大，因为其标记的位置均为匹配成功的字符
            if single_char in p.children and p is self.root:
                start_index = i
            # 若找到匹配成功的字符结点，则指向那个结点，否则指向根结点
            if single_char in p.children:
                p = p.children[single_char]
            else:
                start_index = i
                p = self.root
            temp = p
            while temp is not self.root:
                # 尾标志为0不处理，但是tail需要-1从而与敏感词字典下标一致
                # 循环原因在于，有些词本身只是另一个词的后缀，也需要辨识出来
                if temp.tail:
                    rst[self.words[temp.tail - 1]].append((start_index, i))
                temp = temp.fail
        return rst
