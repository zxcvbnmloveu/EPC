import itertools
import time
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import *
from collections import Counter
import matplotlib.pyplot as plt

from TCM_util import read_data, GaussianNoise, get_mini_batches
from TCM_Model import *
from TCM_losses import *
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def output1(log,mes):
    print(mes)
    log.write(mes+'\n')


def train(source_num,target_num,conf = None, dataset = 'office'):

    target_test, target_test_label0, source_test, source_test_label0, class_num,s_name,t_name\
        = read_data(dataset, source_num, target_num, Is_Norm=conf['Is_Norm'])

    target_test_label = np.zeros((len(target_test_label0), class_num)).astype('float32')
    for i in range(len(target_test_label0)):
        target_test_label[i] = one_hot(class_num, target_test_label0[i])
    print('data_t',target_test.shape)

    source_test_label = np.zeros((len(source_test_label0), class_num)).astype('float32')
    for i in range(len(source_test_label0)):
        source_test_label[i] = one_hot(class_num, source_test_label0[i])
    print('data_s',source_test.shape)

    def getdata():
        # load feature representations
        source_train = source_test
        source_label = source_test_label
        source_label0 = source_test_label0
        target_train = target_test
        target_label = target_test_label
        target_label0 = target_test_label0

        return source_train, source_label, target_train, target_label, source_label0, target_label0


    now = time.localtime(time.time())
    path = './data/acc/' + str(now.tm_mon)  + '_' + str(now.tm_mday) + '/'
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    log = open(path + s_name + '_'+ t_name + '_'
               + str(now.tm_hour) + '_' + str(now.tm_min)+ '_' + str(now.tm_sec) + '.log', 'w')

    result = []
    # for l,u in [(0.4,0.88),(0.4,0.9),(0.4,0.92)]:
    for _ in range(1):
        loss_list = []
        p = conf['p']
        bz = conf['bz']
        u = 0.95
        l = 0.455
        max_k = conf['m_iter']
        isDrop = False
        label_smoothing = 0
        Isls = False
        drop_n = conf['drop_n']
        sigma = 0

        for rangei in range(conf['run_times']):
            print()
            print(conf)
            # create networksb
            in_dim = conf['dim']
            Isbasic = False
            en = En(in_dim, class_num, Drop=isDrop, p=p, Isbasic = Isbasic).cuda()
            de1 = De(in_dim, class_num).cuda()
            de2 = De(in_dim, class_num).cuda()

            scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75,
                                                                      max_iter=3000)

            opt_en = OptimWithSheduler(optim.Adam(en.parameters(), weight_decay=5e-4, lr=conf['lr']),
                                       scheduler)
            opt_de1 = OptimWithSheduler(optim.Adam(de1.parameters(), weight_decay=5e-4, lr=conf['lr']),
                                        scheduler)
            opt_de2 = OptimWithSheduler(optim.Adam(de2.parameters(), weight_decay=5e-4, lr=conf['lr']),
                                        scheduler)

            mini_batches_test_t = get_mini_batches(target_test, target_test_label, 32)
            mini_batches_test_s = get_mini_batches(source_test, source_test_label, 32)

            ftest_t = Variable(torch.from_numpy(target_test)).cuda()
            ftest_s = Variable(torch.from_numpy(source_test)).cuda()

            cnt1 = 0
            cnt2 = 0

            output1(log, '\n')
            output1(log,
                    '########### {:d} training {:s} to {:s}  ############'.format(rangei + 1, s_name[0], t_name[0]))

            opt_en.zero_grad()
            opt_de1.zero_grad()
            opt_de2.zero_grad()

            Lambda = 0
            b_acc1 = 0
            b_acc2 = 0
            print('normal 2de')

            for k in range(max_k):

                if u < l: break
                if k == 0 or k % len_s == 0:
                    source_train, source_label, _, _, _, _ = \
                        getdata()

                    mini_batches_source = get_mini_batches(source_train, source_label, bz, test=False
                                                           ,is_shuffle=True)
                    len_s = len(mini_batches_source)
                    iter_s = iter(mini_batches_source)

                    # cnt1 += 1

                if k == 0 or k % len_t == 0:
                    # 不打乱效果就明显变差
                    _, _, target_train, target_label, _, _ = \
                        getdata()

                    mini_batches_target = get_mini_batches(target_train, target_label, bz, test=False,
                                                           is_shuffle=True)

                    len_t = len(mini_batches_target)
                    iter_t = iter(mini_batches_target)
                    # cnt2 += 1

                im_source, label_source = iter_s.__next__()
                im_target, label_target = iter_t.__next__()

                fs = Variable(torch.from_numpy(im_source)).cuda()
                ls = Variable(torch.from_numpy(label_source)).cuda()
                ft = Variable(torch.from_numpy(im_target)).cuda()
                lt = Variable(torch.from_numpy(label_target)).cuda()

                fsl = fs
                ftl = ft

                de1.train(True)
                de2.train(True)
                en.train(True)

                loss_t = 0
                loss_ce = 0
                mse_s = 0
                mse_t = 0

                for drop_cnt in range(drop_n):
                    if conf['Is_aug']:
                        tmp_fs = GaussianNoise(sigma=0.01).forward(fs)
                        tmp_fs = F.dropout(tmp_fs, p)
                        tmp_ft = GaussianNoise(sigma=0.01).forward(ft)
                        tmp_ft = F.dropout(tmp_ft, p)
                    else:
                        tmp_fs = fs
                        tmp_ft = ft
                    zs, logit_zs = en(tmp_fs.detach())
                    zt, logit_zt = en(tmp_ft.detach())
                    rs = de1(zs)
                    rt = de2(zt)
                    temp = 2
                    lfs_norm = F.normalize(nn.Softmax(1)(logit_zs / temp), p=2, dim=1)
                    tmp_lft_norm = F.normalize(nn.Softmax(1)(logit_zt / temp), p=2, dim=1)
                    if drop_cnt == 0:
                        lft_norm = tmp_lft_norm
                    else:
                        lft_norm = torch.cat((lft_norm,tmp_lft_norm),0)

                    if Isls:
                        ls = (1.0 - label_smoothing) * ls + label_smoothing / class_num
                    loss_ce += CrossEntropyLoss(ls, zs)
                    mse_s += nn.MSELoss()(fsl, rs)
                    mse_t += nn.MSELoss()(ftl, rt)
                # print(lft_norm.size())
                It = lft_norm.mm(lft_norm.t())

                if conf['weight']:
                    loss_t += -torch.mean(
                        (1 + It / 100).detach() * (It.detach() > u).float() * torch.log(torch.clamp(It, 1e-10, 1)) +
                        (1.1 - It / 100).detach() * (It.detach() < l).float() * torch.log(
                            torch.clamp(1 - It, 1e-10, 1)))
                else:
                    loss_t += -torch.mean((It.detach() > u).float() * torch.log(torch.clamp(It, 1e-10, 1)) + (
                            It.detach() < l).float() * torch.log(torch.clamp(1 - It, 1e-10, 1)))

                loss_t /= drop_n
                loss_ce /= drop_n
                mse_s /= drop_n
                mse_t /= drop_n

                with OptimizerManager([opt_en, opt_de1, opt_de2]):
                    loss = 0
                    loss += loss_ce
                    if conf['lt']:
                        loss += loss_t
                    if conf['mse']:
                        loss += 0.01*mse_s
                        loss += mse_t
                    loss.backward()

                if k % 50 == 0:
                    # print(zs.max(1)[0][:9])
                    # print(zt.max(1)[0][:30])
                    with torch.no_grad():
                        # lf_cls.train(False)
                        de1.train(True)
                        de2.train(True)
                        en.train(True)
                        test_zs, _ = en(ftest_s)
                        y_pred_s = test_zs.argmax(1).data.cpu().numpy()
                        ground_truth_s = source_test_label0

                        acc1, ind = cluster_acc(ground_truth_s, y_pred_s)

                        test_zt, _ = en(ftest_t)
                        y_pred_t = test_zt.argmax(1).data.cpu().numpy()
                        ground_truth_t = target_test_label0
                        acc2, ind = cluster_acc(ground_truth_t, y_pred_t)

                        b_acc1 = max(b_acc1, acc1)
                        b_acc2 = max(b_acc2, acc2)
                        print('range {:d} lfcls  sacc:{:.3f} tacc:{:.4f} bsacc:{:.3f} btacc:{:.3f}\n'
                              .format(k, acc1, acc2, b_acc1, b_acc2))
            output1(log, 'range {:d} sacc:{:.3f} tacc:{:.3f}'
                    .format(rangei, b_acc1, b_acc2))
            result.append('range {:d} sacc:{:.3f} tacc:{:.3f}'
                          .format(rangei, b_acc1, b_acc2))

            # for visda mean acc

                # if k % 50 == 0:
                #     # print(zs.max(1)[0][:9])
                #     # print(zt.max(1)[0][:30])
                #     with torch.no_grad():
                #         # lf_cls.train(False)
                #         de1.train(True)
                #         de2.train(True)
                #         en.train(True)
                #         # test_zs, _ = en(ftest_s)
                #         # y_pred_s = test_zs.argmax(1).data.cpu().numpy()
                #         # ground_truth_s = source_test_label0
                #         #
                #         # acc1, ind = cluster_acc(ground_truth_s, y_pred_s)

            #             test_zt, _ = en(ftest_t)
            #             y_pred_t = test_zt.argmax(1).data.cpu().numpy()
            #             ground_truth_t = target_test_label0
            #             acc2, ind = cluster_acc(ground_truth_t, y_pred_t)
            #
            #             st = ['pl','bc','bu','ca','ho','kn','my','pe','pl','sk','tr','tru']
            #             tmp_str = []
            #             acc1 = 0
            #             for i1 in range(class_num):
            #                 sum_i1 = sum(ground_truth_t==i1)
            #                 correct_i1 = sum((y_pred_t==i1)&(y_pred_t==ground_truth_t))
            #                 tmp_acc = correct_i1/sum_i1
            #                 # print(sum_i1,correct_i1)
            #                 acc1 += tmp_acc
            #                 tmp_str.append(st[i1] + ' ' + str(round(tmp_acc, 3)))
            #             acc1 /= class_num
            #             print(' '.join(tmp_str))
            #             b_acc1 = max(b_acc1, acc1)
            #             b_acc2 = max(b_acc2, acc2)
            #             print('range {:d} lfcls  sacc:{:.3f} tacc:{:.4f} bsacc:{:.3f} btacc:{:.3f}\n'
            #                   .format(k, acc1, acc2, b_acc1, b_acc2))
            # output1(log, 'range {:d} sacc:{:.3f} tacc:{:.3f}'
            #         .format(rangei, b_acc1, b_acc2))
            # result.append('range {:d} sacc:{:.3f} tacc:{:.3f}'
            #               .format(rangei, b_acc1, b_acc2))

    output1(log, '\n')
    output1(log, '########### final result {:s} to {:s}  ############'.format(s_name[0], t_name[0]))
    output1(log, '\n'.join(result))


#导入包
parser = argparse.ArgumentParser(description='My Network')
#定义，声明
parser.add_argument('-s', type=int, default=0)
parser.add_argument('-t', type=int, default=1)
parser.add_argument('-gpu', type=str, default='3')
parser.add_argument('-m_iter', type=int, default=5000)
parser.add_argument('-bz', type=int, default=128)


args = parser.parse_args()
#获取一个字典
source_num,target_num = args.s,args.t
conf = {}

# train(source_num,target_num,conf = conf)
def train_of31():
    # office
    # task_office = ['amazon', 'webcam', 'dslr']
    conf['p'] = 0.7
    conf['m_iter'] = 3000
    conf['bz'] = 128
    conf['drop_n'] = 2
    conf['run_times'] = 10
    conf['Is_Norm'] = False
    conf['lr'] = 1e-3
    conf['weight'] = False
    # train(0,1,conf = conf,dataset='office')
    # train(0,2,conf = conf,dataset='office')
    train(1,0,conf = conf,dataset='office')
    # train(1,2,conf = conf,dataset='office')
    train(2,0,conf = conf,dataset='office')
    # train(2,1,conf = conf,dataset='office')

def train_oh():
    # elif dataset == 'office-home':
    # class_num = 65
    # task_office = ['Clipart', 'Product', 'RealWorld', 'Art']
    conf['p'] = 0.3
    conf['m_iter'] = 3000
    conf['bz'] = 256
    conf['drop_n'] = 1
    conf['run_times'] = 10
    conf['Is_Norm'] = False
    conf['lr'] = 1e-3
    conf['weight'] = False

    train(0,1,conf = conf,dataset='office-home')
    train(0,2,conf = conf,dataset='office-home')
    train(0,3,conf = conf,dataset='office-home')
    train(1,0,conf = conf,dataset='office-home')
    train(1,2,conf = conf,dataset='office-home')
    train(1,3,conf = conf,dataset='office-home')
    train(2,0,conf = conf,dataset='office-home')
    train(2,1,conf = conf,dataset='office-home')
    train(2,3,conf = conf,dataset='office-home')
    train(3,0,conf = conf,dataset='office-home')
    train(3,1,conf = conf,dataset='office-home')
    train(3,2,conf = conf,dataset='office-home')

def train_clef():
    # dataset == 'clef':
    # task_office = ['c', 'i', 'p']
    # class_num = 12
    train(0,1,conf = conf,dataset='clef')
    train(0,2,conf = conf,dataset='clef')
    train(1,0,conf = conf,dataset='clef')
    train(1,2,conf = conf,dataset='clef')
    train(2,0,conf = conf,dataset='clef')
    train(2,1,conf = conf,dataset='clef')

def train_visda():
    # dataset == 'visda':
    # class_num = 12
    # task_office = ['train', 'validation']
    train(0,1,conf = conf,dataset='visda')

def train_decaf():
    # office
    # task_office = ['c', 'a', 'w', 'd']
    conf['p'] = 0.5
    conf['bz'] = 128
    conf['m_iter'] = 1500
    conf['drop_n'] = 2
    conf['dim'] = 4096
    conf['run_times'] = 5
    conf['Is_Norm'] = True
    conf['lr'] = 1e-2

    train(0,1,conf = conf,dataset='decaf')
    train(0,2,conf = conf,dataset='decaf')
    train(0,3,conf = conf,dataset='decaf')

    # train(1,0,conf = conf,dataset='decaf')
    train(1,2,conf = conf,dataset='decaf')
    train(1,3,conf = conf,dataset='decaf')

    train(2,0,conf = conf,dataset='decaf')
    train(2,1,conf = conf,dataset='decaf')
    train(2,3,conf = conf,dataset='decaf')

    train(3,0,conf = conf,dataset='decaf')
    train(3,1,conf = conf,dataset='decaf')
    train(3,2,conf = conf,dataset='decaf')


def ablation():
    conf['weight'] = True
    conf['Is_aug'] = True
    conf['lt'] = True
    conf['mse'] = True

    conf['dim'] = 2048
    conf['Is_Norm'] = False
    conf['p'] = 0.5
    conf['bz'] = 128
    conf['drop_n'] = 2
    conf['run_times'] = 3
    conf['lr'] = 1e-3

    conf['m_iter'] = 2000
    # task_office = ['amazon', 'webcam', 'dslr']
    train(0,1,conf = conf,dataset='office')
    train(0,2,conf = conf,dataset='office')
    train(1,0,conf = conf,dataset='office')
    train(2,0,conf = conf,dataset='office')

    # task_office = ['c', 'i', 'p']

    conf['m_iter'] = 3000
    conf['drop_n'] = 2

    # train(0,1,conf = conf,dataset='visda')
    # # task_office = ['Clipart', 'Product', 'RealWorld', 'Art']
    conf['p'] = 0.1
    conf['m_iter'] = 2000
    train(0, 1, conf=conf, dataset='office-home')
    train(3, 1, conf=conf, dataset='office-home')

ablation()

print('CATCM')