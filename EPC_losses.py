import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
import os
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def variable_to_numpy(x):
    ans = x.cpu().data.numpy()
    if torch.numel(x) == 1:
        return float(np.sum(ans))
    return ans

def inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=1000):
    return initial_lr * ((1 + gamma * min(1.0, step / float(max_iter))) ** (- power))



def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class OptimWithSheduler:
    def __init__(self, optimizer, scheduler_func):
        self.optimizer = optimizer
        self.scheduler_func = scheduler_func
        self.global_step = 0.0
        for g in self.optimizer.param_groups:
            g['initial_lr'] = g['lr']

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        for g in self.optimizer.param_groups:
            g['lr'] = self.scheduler_func(step=self.global_step, initial_lr=g['initial_lr'])
        self.optimizer.step()
        self.global_step += 1


class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  # if isinstance(optims, Iterable) else [optims]

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True

def setGPU(i):
    global os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % (i)
    gpus = [x.strip() for x in (str(i)).split(',')]
    NGPU = len(gpus)
    print('gpu(s) to be used: %s' % str(gpus))
    return NGPU

def CrossEntropyLoss(label, predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()
    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'
    ce = -label * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * ce * class_level_weight) / float(N)


def BCELossForMultiClassification(label, predict_prob, class_level_weight=None, instance_level_weight=None,
                                  epsilon=1e-12):
    N, C = label.size()
    N_, C_ = predict_prob.size()

    assert N == N_ and C == C_, 'fatal error: dimension mismatch!'

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    bce = -label * torch.log(predict_prob + epsilon) - (1.0 - label) * torch.log(1.0 - predict_prob + epsilon)
    return torch.sum(instance_level_weight * bce * class_level_weight) / float(N)


def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind


def test(model1,model2,mini_batches_test,both = True):
    with torch.no_grad():

        correct = 0
        total_num = 0

        correct1 = 0
        total_num1 = 0

        model1.train(False)
        model2.train(False)
        model1.eval()
        model2.eval()

        acc1 = 0
        if both == True:
            for (i, (im, label)) in enumerate(mini_batches_test):
                fs = Variable(torch.from_numpy(im)).cuda()
                bzs, d = fs.size()[0], fs.size()[1]
                # fs = fs.view(fs.size()[0], 1, fs.size()[1])

                label = Variable(torch.from_numpy(label)).cuda()

                recover_fea_s, z = model1(fs)
                # recover_fea_s = recover_fea_s.view(bzs, d)
                logit_s, prob_source = model2.forward(recover_fea_s)

                predict_prob, label1 = [variable_to_numpy(x) for x in (prob_source, label)]
                label1 = np.argmax(label1, axis=-1).reshape(-1, 1)
                predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
                correct += float(np.sum(label1.flatten() == predict_index.flatten()))
                total_num += label1.flatten().shape[0]

            acc1 = correct / total_num

        for (i, (im, label)) in enumerate(mini_batches_test):
            fs = Variable(torch.from_numpy(im)).cuda()

            label = Variable(torch.from_numpy(label)).cuda()

            logit_s, prob_source = model2.forward(fs)

            predict_prob, label1 = [variable_to_numpy(x) for x in (prob_source, label)]
            label1 = np.argmax(label1, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            correct1 += float(np.sum(label1.flatten() == predict_index.flatten()))
            total_num1 += label1.flatten().shape[0]

        acc2 = correct1 / total_num1

        return acc1,acc2


def test_z(model1,model2,mini_batches_test,both = True):
    with torch.no_grad():

        correct = 0
        total_num = 0

        correct1 = 0
        total_num1 = 0

        model1.train(False)
        model2.train(False)
        model1.eval()
        model2.eval()

        acc1 = 0
        if both == True:
            for (i, (im, label)) in enumerate(mini_batches_test):
                fs = Variable(torch.from_numpy(im)).cuda()
                bzs, d = fs.size()[0], fs.size()[1]
                # fs = fs.view(fs.size()[0], 1, fs.size()[1])

                label = Variable(torch.from_numpy(label)).cuda()

                recover_fea_s, z = model1(fs)
                # recover_fea_s = recover_fea_s.view(bzs, d)
                logit_s, prob_source = model2.forward(recover_fea_s)

                predict_prob, label1 = [variable_to_numpy(x) for x in (prob_source, label)]
                label1 = np.argmax(label1, axis=-1).reshape(-1, 1)
                predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
                correct += float(np.sum(label1.flatten() == predict_index.flatten()))
                total_num += label1.flatten().shape[0]

            acc1 = correct / total_num

        for (i, (im, label)) in enumerate(mini_batches_test):
            fs = Variable(torch.from_numpy(im)).cuda()
            recover_fea_s, z = model1(fs)
            label = Variable(torch.from_numpy(label)).cuda()

            logit_s, prob_source = model2.forward(z)

            predict_prob, label1 = [variable_to_numpy(x) for x in (prob_source, label)]
            label1 = np.argmax(label1, axis=-1).reshape(-1, 1)
            predict_index = np.argmax(predict_prob, axis=-1).reshape(-1, 1)
            correct1 += float(np.sum(label1.flatten() == predict_index.flatten()))
            total_num1 += label1.flatten().shape[0]

        acc2 = correct1 / total_num1

        return acc1,acc2

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


class Regularization(torch.nn.Module):
    def __init__(self, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.weight_decay = weight_decay
        self.p = p

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

