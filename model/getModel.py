from model import bertForBQ, bertClassifier
import torch.nn as nn
import torch.optim as optim


def getModel(mission, lr, size=2):
    if mission in ['qnli']:
        model = bertForBQ.fn_cls()
    elif mission in ['sst','toutiao','weibo']:
        model = bertClassifier.fn_cls(size=size)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    return model, criterion, optimizer


def trainModel(mission, train, model, criterion, optimizer, batch_size, save_model):
    if mission in ['qnli']:
        epoch_loss = bertForBQ.trainClassifier(train, model, criterion, optimizer, batch_size, save_model)
    elif mission in ['sst','toutiao','weibo']:
        epoch_loss = bertClassifier.trainClassifier(train, model, criterion, optimizer, batch_size, save_model)
    return epoch_loss


def testModel(mission, data, model, batch_size):
    if mission in ['qnli']:
        answers, tp, fp, tn, fn = bertForBQ.predictClassifier(data, model, batch_size)
    elif mission in ['sst','toutiao','weibo']:
        answers, tp, fp, tn, fn = bertClassifier.predictClassifier(data, model, batch_size)
    return answers, tp, fp, tn, fn
