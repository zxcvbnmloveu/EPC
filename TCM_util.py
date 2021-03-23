import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from TCM_losses import BCELossForMultiClassification, CrossEntropyLoss
import numpy as np
import h5py


class GaussianNoise(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0.0).cuda()

    def forward(self, x):
        if self.training:
            sampled_noise = self.noise.repeat(*x.size()).normal_(mean=0, std=self.sigma)
            x = x + sampled_noise
        return x

def get_model_params(net, name):
    """Get parameters of models by name."""
    for n, p in net.named_parameters():
        if n == name:
            return p

def calc_similiar_penalty(F_1, F_2):
    """Calculate similiar penalty |W_1^T W_2|."""
    F_1_params = F_1.parameters()
    F_2_params = F_2.parameters()
    similiar_penalty = [torch.sum(layer1 *layer2) / 2
                        for layer1,layer2 in zip(F_1_params,F_2_params)]
    mean = lambda x: 0.05 * torch.mean(torch.stack(x))

    return mean(similiar_penalty)

def read_data(dataset, s, t, Is_Norm = False):
    if dataset == 'office':
        class_num = 31
        task_office = ['amazon', 'webcam', 'dslr']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name
        target_name = t_name

        data_t = scipy.io.loadmat('./data/Office31/' + target_name + '.mat')
        data_s = scipy.io.loadmat('./data/Office31/' + source_name + '.mat')

        target_test = data_t['resnet50_features']
        target_test = np.reshape(target_test, (target_test.shape[0], 2048)).astype('float32')
        target_test_label0 = data_t['labels']
        target_test_label0 = np.reshape(target_test_label0, -1).astype('int')

        source_test = data_s['resnet50_features']
        source_test = np.reshape(source_test, (source_test.shape[0], 2048)).astype('float32')
        source_test_label0 = data_s['labels']
        source_test_label0 = np.reshape(source_test_label0, -1).astype('int')

    elif dataset == 'office-home':
        class_num = 65
        task_office = ['Clipart', 'Product', 'RealWorld', 'Art']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name
        target_name = t_name

        data_t = scipy.io.loadmat('./data/OfficeHome/' + target_name + '.mat')
        data_s = scipy.io.loadmat('./data/OfficeHome/' + source_name + '.mat')

        target_test = data_t['resnet50_features']
        target_test = np.reshape(target_test, (target_test.shape[0], 2048)).astype('float32')
        target_test_label0 = data_t['labels']
        target_test_label0 = np.reshape(target_test_label0, -1).astype('int')

        source_test = data_s['resnet50_features']
        source_test = np.reshape(source_test, (source_test.shape[0], 2048)).astype('float32')
        source_test_label0 = data_s['labels']
        source_test_label0 = np.reshape(source_test_label0, -1).astype('int')

    elif dataset == 'clef':
        class_num = 12
        task_office = ['c', 'i', 'p']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name
        target_name = t_name
        data_t = scipy.io.loadmat('./data/ImageCLEF/' + target_name + '.mat')
        data_s = scipy.io.loadmat('./data/ImageCLEF/' + source_name + '.mat')

        target_test = data_t['resnet50_features']
        target_test = np.reshape(target_test, (target_test.shape[0], 2048)).astype('float32')
        target_test_label0 = data_t['labels']
        target_test_label0 = np.reshape(target_test_label0, -1).astype('int')

        source_test = data_s['resnet50_features']
        source_test = np.reshape(source_test, (source_test.shape[0], 2048)).astype('float32')
        source_test_label0 = data_s['labels']
        source_test_label0 = np.reshape(source_test_label0, -1).astype('int')

    elif dataset == 'visda':
        task_office = ['train', 'validation']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name
        target_name = t_name

        class_num = 12
        # data_s = h5py.File('./data/VisDA_resnet50/train.mat', 'r')
        # data_t = h5py.File('./data/VisDA_resnet50/validation.mat', 'r')
        # source_test = data_s['fts']
        # source_test = np.transpose(source_test, (1, 0)).astype('float32')
        # source_test_label0 = data_s['labels']
        # source_test_label0 = np.reshape(source_test_label0, -1).astype('int')
        # target_test = data_t['fts']
        # target_test = np.transpose(target_test, (1, 0)).astype('float32')
        # target_test_label0 = data_t['labels']
        # target_test_label0 = np.reshape(target_test_label0, -1).astype('int')

        target_test = np.load('./data/visda/'+target_name+'_test_fea.npy').astype('float32')
        target_test_label0 = np.load('./data/visda/'+target_name+'_test_label.npy').astype('int')
        source_test = np.load('./data/visda/'+source_name+'_test_fea.npy').astype('float32')
        source_test_label0 = np.load('./data/visda/'+source_name+'_test_label.npy').astype('int')

    elif dataset == 'visda101':
        task_office = ['train', 'validation']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name + '_' + s_name
        target_name = s_name + '_' + t_name

        class_num = 12
        target_test = np.load('./data/visda/validation_test_fea101.npy').astype('float32')
        target_test_label0 = np.load('./data/visda/validation_test_label101.npy').astype('int')
        source_test = np.load('./data/visda/train_test_fea101.npy').astype('float32')
        source_test_label0 = np.load('./data/visda/train_test_label101.npy').astype('int')

    elif dataset == 'decaf':
        task_office = ['c','a','w','d']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name
        target_name = t_name

        class_num = 10
        data_s = scipy.io.loadmat('./data/ptj_decaf/' + source_name + '.mat')
        data_t = scipy.io.loadmat('./data/ptj_decaf/' + target_name + '.mat')
        source_test = data_s['feas'].astype('float32')
        source_test_label0 = data_s['labels']
        source_test_label0 = np.reshape(source_test_label0, -1).astype('int')-1
        target_test = data_t['feas'].astype('float32')
        target_test_label0 = data_t['labels']
        target_test_label0 = np.reshape(target_test_label0, -1).astype('int')-1

    elif dataset == 'pie':
        task_office = ['PIE05','PIE07','PIE09','PIE27','PIE29']
        s_name = task_office[s]
        t_name = task_office[t]
        source_name = s_name
        target_name = t_name

        class_num = 68
        data_s = scipy.io.loadmat('./data/PIE/' + source_name + '.mat')
        data_t = scipy.io.loadmat('./data/PIE/' + target_name + '.mat')
        source_test = data_s['fea'].astype('float32')
        source_test_label0 = data_s['gnd']
        source_test_label0 = np.reshape(source_test_label0, -1).astype('int')-1
        target_test = data_t['fea'].astype('float32')
        target_test_label0 = data_t['gnd']
        target_test_label0 = np.reshape(target_test_label0, -1).astype('int')-1
        mark = ['1','2','3','4','5']
        s_name = mark[s]
        t_name = mark[t]

    elif dataset == 'PACS':
        task_office = ['art_painting','cartoon','photo','sketch']
        # task_office = ['a','c','p','s']

        s_name = task_office[s]
        t_name = task_office[t]
        target_name = 'target_'+t_name

        class_num = 7
        target_test = np.load('./data/PACS/' + target_name + '_test_fea.npy').astype('float32')
        target_test_label0 = np.load('./data/PACS/' + target_name + '_test_label.npy').astype('int')
        start = True

        # source_name = task_office[s]
        # source_test = np.load('./data/PACS/' + source_name + '_test_fea.npy').astype('float32')
        # source_test_label0 = np.load('./data/PACS/' + source_name + '_test_label.npy').astype('int')
        #
        for i in range(4):
            if i == t: continue
            source_name = task_office[i]+'_'+t_name
            if start:
                start = False
                source_test = np.load('./data/PACS/' + source_name + '_test_fea.npy').astype('float32')
                source_test_label0 = np.load('./data/PACS/' + source_name + '_test_label.npy').astype('int')

            else:
                tmp_source_test = np.load('./data/PACS/' + source_name + '_test_fea.npy').astype('float32')
                tmp_source_test_label0 = np.load('./data/PACS/' + source_name + '_test_label.npy').astype('int')

                source_test = np.concatenate([source_test,tmp_source_test],0)
                source_test_label0 = np.concatenate([source_test_label0,tmp_source_test_label0],0)

    if Is_Norm:
        target_test,source_test = norm(target_test,source_test)

    return target_test,target_test_label0,source_test,source_test_label0,class_num,s_name,t_name

def norm(a,b):
    ab = np.concatenate((a, b), axis=0)
    x_norm = np.linalg.norm(ab, ord=2, axis=0, keepdims=True)
    a = a / x_norm
    b = b / x_norm
    # Divide x by its norm.
    print(x_norm.shape)
    scaler = StandardScaler()
    fit_data = scaler.fit_transform(ab)
    a = fit_data[:a.shape[0]]
    b = fit_data[a.shape[0]:]

    return a, b

def shuffle(X, Y):

    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffle = X[permutation]
    Y_shuffle = Y[permutation]
    shuffles = {"X_shuffle": X_shuffle, "Y_shuffle": Y_shuffle}
    return shuffles

def get_mini_batches(X, Y, mini_batch_size,test = True, is_shuffle = True):
    if is_shuffle:
        shuffles = shuffle(X, Y)
        # print(shuffles["X_shuffle"].shape)
    else:
        shuffles = {"X_shuffle": X, "Y_shuffle": Y}
        # print(shuffles["X_shuffle"].shape)
    num_examples = shuffles["X_shuffle"].shape[0]
    num_complete =  num_examples // mini_batch_size
    mini_batches = []
    for i in range(num_complete):
        mini_batches.append([shuffles["X_shuffle"][i * mini_batch_size:(i + 1) * mini_batch_size], shuffles["Y_shuffle"][i * mini_batch_size:(i + 1) * mini_batch_size]])

    if test:
        if 0 == num_examples % mini_batch_size:
            pass
        else:
            mini_batches.append([shuffles["X_shuffle"][num_complete * mini_batch_size:], shuffles["Y_shuffle"][num_complete * mini_batch_size:]])
    return mini_batches