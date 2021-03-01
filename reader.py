from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import math,random
# import cPickle

def prepare_data(seqs , labels , maxlen):
    seqs_new  = []
    for s in seqs:
        length = len(s)
        if length < maxlen:
            zero_list = [0] * (maxlen - length)
            s = zero_list + s
            seqs_new.append(s)
        else:
            s = s[:maxlen]
            seqs_new.append(s)
    labels = np.array(labels)
    return [np.array(seqs_new) , labels ]


'''
def prepare_data_wc(seqs , words , labels , s_maxlen , w_maxlen):
    seqs_new  = []
    for s in seqs:
        length = len(s)
        if length < maxlen:
            zero_list = [0] * (maxlen - length)
            s = zero_list + s
            seqs_new.append(s)
        else:
            s = s[:maxlen]
            seqs_new.append(s)
    labels = np.array(labels)
    return [np.array(seqs_new) , labels ]

'''
def prepare_data_wc(sens , words , labels , s_maxlen , w_maxlen):
    sens_new , words_new  = [] , []
    assert len(sens) == len(words)
    length = len(sens)
    s , w = [] , []
    for i in range(length) :
        s_len = len(sens[i])
        assert s_len == len(words[i])
        # pad sen
        if s_len < s_maxlen :
            zero_list = [0] * (s_maxlen - s_len)
            s = zero_list + sens[i]
            zeros_list_char = [ [0] * w_maxlen ] * (s_maxlen - s_len)
            w = zeros_list_char + words[i]
            sens_new.append(s)
        else : 
            s = sens[i][:s_maxlen]
            w = words[i][:s_maxlen]
            sens_new.append(s)
        # pad word
        w_tmp = []
        for word in w:
            w_len = len(word)
            if w_len < w_maxlen :
                zero_list = [0] * (w_maxlen - w_len)
                w_pad = zero_list + word
                w_tmp.append(w_pad)
            else :
                w_pad = word[:w_maxlen]
                w_tmp.append(w_pad)
        words_new.append(w_tmp)

    labels = np.array(labels)
    sens_new = np.array(sens_new)
    words_new = np.array(words_new)
    print(sens_new.shape , words_new.shape , labels.shape)
    return [sens_new , words_new , labels ]



def next_batch_data(data, label, batch_size , maxLen=300):
    while 1:
        epoch_size = int(math.ceil(data.shape[0] / batch_size))
        for i in range(epoch_size):
            if i < epoch_size - 1:
                data_batch = data[i * batch_size : (i+1) * batch_size]
                label_batch = label[i * batch_size : (i+1) * batch_size]
            else:
                data_batch = data[i * batch_size:]
                label_batch = label[i * batch_size:]
            yield data_batch, label_batch

def shuffle_data(data, label , shuffle = True):
    data_temp=[]
    label_temp=[]
    len_data = len(data)
    li=range(len_data)
    if shuffle :
        random.shuffle(li)
        for i in li:
            data_temp.append(data[i])
            label_temp.append(label[i])
    else :
        for i in li:
            data_temp.append(data[i])
            label_temp.append(label[i])
    # print(len(data_temp) , len(label_temp))
    return data_temp , label_temp

def shuffle_data_wc(data , word , label , shuffle = True):
    data_temp = []
    word_temp = []
    label_temp = []
    len_data = len(data)
    li=list(range(len_data))
    if shuffle :
        random.shuffle(li)
        for i in li:
            data_temp.append(data[i])
            word_temp.append(word[i])
            label_temp.append(label[i])
    else :
        for i in li:
            data_temp.append(data[i])
            word_temp.append(word[i])
            label_temp.append(label[i])
    # print(len(data_temp) , len(label_temp))
    return data_temp , word_temp , label_temp



def remove_unk(x , n_words):
    return [[1 if w>=n_words else w for w in sen] for sen in x]


def load_data(path , n_words = None , shuffle = False):
    with open(path , 'rb') as f:
        dataset_x , dataset_label = np.load(f)
        # print(len(dataset_x) , len(dataset_label))
    
    x , y  = shuffle_data(dataset_x , dataset_label , shuffle)
    return [x , y] 

def load_data_version_yjy(data_path , label_path , shuffle = False) :
    data = np.load(data_path)['sentences']
    label = np.load(label_path)['labels']

    x , y  = shuffle_data(data , label , shuffle)
    return [x,y]

def load_data_wc(data_path , shuffle = False):
    data = np.load(data_path,allow_pickle=True)
    sentences = data['sentence']
    words = data['word']
    labels = data['label']
    sentences , words , labels = shuffle_data_wc(sentences , words , labels , shuffle)
    return [sentences , words , labels ] 



def fold_split(fold_num, sample_num , shuffle = True):
    sample_num_one_fold = int(math.ceil(sample_num / fold_num))
    indexs = range(sample_num)
    if shuffle :
        random.shuffle(indexs)
    indexs_folds = []
    for i in range(fold_num):
        indexs_fold = {}
        if i < fold_num - 1:
            indexs_test = indexs[i * sample_num_one_fold : (i+1) * sample_num_one_fold]
            indexs_train = indexs[0 : i * sample_num_one_fold] + indexs[(i+1) * sample_num_one_fold:]
        else:
            indexs_test = indexs[i * sample_num_one_fold:]
            indexs_train = indexs[0 : i * sample_num_one_fold]
        indexs_fold['train'] = indexs_train
        indexs_fold['test'] = indexs_test
        indexs_folds.append(indexs_fold)
    return indexs_folds


def data_one_fold(indexs_folds, n, data, label):
    indexs_fold = indexs_folds[n]

    data_train = np.zeros((len(indexs_fold['train']), data.shape[1]))
    label_train = np.zeros((len(indexs_fold['train'])))
    for i in range(len(indexs_fold['train'])):
        data_train[i] = data[indexs_fold['train'][i]]
        label_train[i] = label[indexs_fold['train'][i]]

    data_test = np.zeros((len(indexs_fold['test']), data.shape[1]))
    label_test = np.zeros((len(indexs_fold['test'])))
    for i in range(len(indexs_fold['test'])):
        data_test[i] = data[indexs_fold['test'][i]]
        label_test[i] = label[indexs_fold['test'][i]]

    return data_train , label_train, data_test , label_test


def data_one_fold_wc(indexs_folds, n, data, word , label):
    indexs_fold = indexs_folds[n]

    data_train = np.zeros((len(indexs_fold['train']), data.shape[1]))
    word_train = np.zeros((len(indexs_fold['train']), word.shape[1], word.shape[2]))
    label_train = np.zeros((len(indexs_fold['train'])))
    for i in range(len(indexs_fold['train'])):
        data_train[i] = data[indexs_fold['train'][i]]
        word_train[i] = word[indexs_fold['train'][i]]
        label_train[i] = label[indexs_fold['train'][i]]

    data_test = np.zeros((len(indexs_fold['test']), data.shape[1]))
    word_test = np.zeros((len(indexs_fold['test']), word.shape[1], word.shape[2]))
    label_test = np.zeros((len(indexs_fold['test'])))
    for i in range(len(indexs_fold['test'])):
        data_test[i] = data[indexs_fold['test'][i]]
        word_test[i] = word[indexs_fold['test'][i]]
        label_test[i] = label[indexs_fold['test'][i]]

    return data_train, word_train , label_train, data_test, word_test , label_test







