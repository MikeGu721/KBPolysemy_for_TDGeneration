import os
from tools import gzh

knowledgeGraph = gzh.cndbpedia_json

if os.path.abspath('.').split('\\')[-1] != '属性值对齐':
    chooseP_output = '../resultOfKBC/chooseP'
    chooseO_output = '../resultOfKBC/chooseO'
    clusterO_output = '../resultOfKBC/clusterO'
else:
    chooseP_output = './resultOfKBC/chooseP'
    chooseO_output = './resultOfKBC/chooseO'
    clusterO_output = './resultOfKBC/clusterO'


class augmentTestPar:
    def __init__(self):
        # self.mission = 'weibo'
        self.mission = 'toutiao'

        self.language = 'chinese'
        # self.language = 'english'

        self.func = 'replace'
        # self.func = 'pattern'

        # learning rate
        self.lr = 1e-2
        # batch_size for training
        self.train_batch_size = 32
        # batch size for testing
        self.test_batch_size = 64
        # EPOCH for training
        self.EPOCH = 10
        # the rate of added data
        self.augmentationRate = 0.1
        # I FORGOT
        self.stopNum = 500 if self.func == 'pattern' else -1


class choosePPar():
    def __init__(self):
        '''
        min_num: 该属性值至少有min_num个三元组
        excel_name: 数据输出到哪里
        '''
        self.min_num = 5000
        self.excel_name = os.path.join(chooseP_output, 'chooseP_output.xls')


class clusterOPar():
    def __init__(self):
        '''

        '''
        self.eps = 0.2


class chooseOPar():
    def __init__(self):
        '''
        p_range: 负采样p的搜寻范围
        o_range: 负采样o的搜寻范围
        metric_threhold: 选择属性的metric小于某阈值的进行选择O的过程
        EPOCH: 分类器训练多少轮
        train_batch_size: 训练时的batch_size
        test_batch_size: 测试时的batch_size
        highest_rate: 保留最高频的属性值永远为真
        learning_rate: 学习率
        use_template: 训练时是否使用模板
        save_model: 是否需要保存模型
        need_iter: 是否需要迭代
        '''
        self.p_range = 2
        self.o_range = 2
        self.metric_threhold = 1100
        self.EPOCH = 3
        self.train_batch_size = 2
        self.test_batch_size = 16
        self.highest_rate = 0.95
        self.learning_rate = 0.001
        self.use_template = True
        self.save_model = False
        self.need_iter = True
        self.threhold = 0.2
        self.template = ['我的[P]是[O]。',
                         '这个东西的[P]是[O]。',
                         '我的[P]是[O]。',
                         '[S]的[P]是[O]。',
                         '所有[S]的[P]是[O]。',
                         '[O]是用来描述[P]的。',
                         '[O]可以用来描述[S]的[P]。',
                         '很多[S]都是[O][P]的。']

    def getTemplate(self, p, o, label, s=None):
        answer = []
        for tem in self.template:
            if s:
                if '[S]' in tem:
                    tem = tem.replace('[S]', s)
            else:
                if '[S]' in tem:
                    continue
            tem = tem.replace('[P]', p)
            tem = tem.replace('[O]', o)
            answer.append([tem, label])
        return answer


if __name__ == '__main__':
    import torch
    embedding = torch.nn.Embedding(5,3)
    co = chooseOPar()
    word = [['性别', '男', [0, 1]],
            ['性别', '女', [0, 1]],
            ['性别', '未知', [1, 0]]]
    a = [[co.getTemplate(i, j, k)] for i, j, k in word]
    temp = []
    for i in a:
        for j in i:
            temp.extend(j)
    a = temp
    for i in a:
        print(i)
        print('---' * 20)
