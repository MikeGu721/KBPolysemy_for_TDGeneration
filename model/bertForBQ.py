from tools import gzh

import torch, random, tqdm
import torch.nn as nn

from pytorch_transformers import BertModel, BertTokenizer


class fn_cls(nn.Module):
    def __init__(self, device='cuda'):
        super(fn_cls, self).__init__()
        self.bert = BertModel.from_pretrained(gzh.bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(gzh.bert_model)
        self.linear = nn.Linear(768 * 2, 2)

        self.bert.to(device)
        self.linear.to(device)
        self.device = device

    def forward(self, text_lis1, text_lis2: list, max_len=16):
        '''
        :param text_lis: batch_size * 2
        :param max_len:
        :return:
        '''
        token1 = [self.tokenizer.tokenize(text) for text in text_lis1]
        token2 = [self.tokenizer.tokenize(text) for text in text_lis2]
        token1 = [self.tokenizer.convert_tokens_to_ids(i)[:max_len] + [0] * max(0, max_len - len(i)) for i in token1]
        token2 = [self.tokenizer.convert_tokens_to_ids(i)[:max_len] + [0] * max(0, max_len - len(i)) for i in token2]
        output1 = self.bert(torch.tensor(token1).to(self.device))[1]
        output2 = self.bert(torch.tensor(token2).to(self.device))[1]
        output = self.linear(torch.cat((output1, output2), dim=1))
        return torch.softmax(output, dim=1)


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
    texts1 = []
    texts2 = []
    targets = []
    epoch_loss = 0
    train = random.sample(train, len(train))
    for word1, word2, target in train:
        texts1.append(word1)
        texts2.append(word2)
        targets.append(target)
        if len(texts1) == batch_size:
            outputs = model(texts1, texts2)
            loss = criterion(outputs, torch.FloatTensor(targets).to('cuda'))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            texts1 = []
            texts2 = []
            targets = []
    if texts1:
        outputs = model(texts1, texts2)
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
    texts1 = []
    texts2 = []
    test = random.sample(test, len(test))
    answers = []
    labels = []
    for word1, word2, label in test:
        texts1.append(word1)
        texts2.append(word2)
        labels.append(label)
        if len(texts1) == batch_size:
            outputs = model(texts1, texts2)
            outputs = outputs.detach().cpu().numpy()
            answers.extend(outputs)
            texts1 = []
            texts2 = []
            for output, label in zip(outputs, labels):
                if label == 1:
                    if output[1] > output[0]:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if output[1] > output[0]:
                        fp += 1
                    else:
                        tn += 1
            labels = []
    if texts1:
        outputs = model(texts1, texts2)
        outputs = outputs.detach().cpu().numpy()
        answers.extend(outputs)
        for output, label in zip(outputs, labels):
            if label == 1:
                if output[1] > output[0]:
                    tp += 1
                else:
                    fn += 1
            else:
                if output[1] > output[0]:
                    fp += 1
                else:
                    tn += 1
    return answers, tp, fp, tn, fn
