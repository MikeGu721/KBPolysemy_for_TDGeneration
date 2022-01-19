from tools import gzh

import torch, random
import torch.nn as nn

from pytorch_transformers import BertModel, BertTokenizer

class fn_cls(nn.Module):
    def __init__(self, device='cuda', size=2):
        super(fn_cls, self).__init__()
        self.bert = BertModel.from_pretrained(gzh.bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(gzh.bert_model)
        self.linear = nn.Linear(768, size)

        self.bert.to(device)
        self.linear.to(device)
        self.device = device

    def forward(self, text_lis: list, max_len=10):
        token = [self.tokenizer.tokenize(text) for text in text_lis]
        token = [self.tokenizer.convert_tokens_to_ids(i)[:max_len] + [0] * max(0, max_len - len(i)) for i in token]
        output = self.bert(torch.tensor(token).to(self.device))[1]
        output = self.linear(output)
        return torch.softmax(output, dim=1)


def getContext(word1, word2, use_template):
    '''

    :param hyper: 上位词
    :param hypo: 下位词
    :param use_template: 是否用模板
    :return:
    '''
    if not use_template:
        return '%s[SEP]%s' % (word1, word2)
    else:
        return '我的%s是%s。' % (word1, word2)


def trainClassifier(train, model, criterion, optimizer, batch_size, save_model):
    '''
    训练
    :param train: x,y
    :param model:
    :param criterion:
    :param optimizer:
    :param batch_size:
    :param use_template:
    :param save_model:
    :return:
    '''
    texts = []
    targets = []
    epoch_loss = 0
    train = random.sample(train, len(train))
    for word, target in train:
        texts.append(word)
        targets.append(target)
        if len(texts) == batch_size:
            outputs = model(texts)
            loss = criterion(outputs, torch.FloatTensor(targets).to('cuda'))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            texts = []
            targets = []
    if texts:
        outputs = model(texts)
        loss = criterion(outputs, torch.FloatTensor(targets).to('cuda'))
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if save_model:
        torch.save(model.state_dict(), './model/BertClassifier.pkl')
        print('已保存模型')
    return epoch_loss


def predictClassifier(test, model, batch_size):
    '''
    预测
    :param test:
    :param model:
    :param batch_size:
    :param use_template:
    :return:
    '''
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    texts = []
    test = random.sample(test, len(test))
    answers = []
    labels = []
    for word, label in test:
        texts.append(word)
        labels.append(label)
        if len(texts) == batch_size:
            outputs = model(texts)
            answers.extend(outputs.detach().cpu().numpy())
            for label, output in zip(labels, outputs):
                if label[output.detach().cpu().numpy().argmax()] == 1:
                    tp += 1
                    tn += len(label)-1
                else:
                    fp += 1
                    fn += 1
                    tn += len(label)-2
            labels = []
            texts = []
    if texts:
        outputs = model(texts)
        answers.extend(outputs.detach().cpu().numpy())
        for label, output in zip(labels, outputs):
            if label[output.detach().cpu().numpy().argmax()] == 1:
                tp += 1
                tn += len(label)-1
            else:
                fp += 1
                fn += 1
                tn += len(label)-2
    return answers, tp, fp, tn, fn
