#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import math,random
import cPickle as cp
# import pickle as cp
import sys , os
import collections
import re
from io import open
import json
import codecs
import operator
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk import data


english_punctuations = ['`','>','<','=','``','"','–','—','‘','-','…',',', '.',
                        ':', ';', '?', '(', ')', '[', ']',
                        '&', '!', '*', '@', '#', '$', '%', '\'', '\"']

np.random.seed(1234)
special_chars = re.compile(r'[^A-Za-z_\d,.?!;:$\- \'\"]', re.IGNORECASE)

# special tokens
PAD = u'__PAD__'
UNK = u'__UNK__'


def clean_text(text):
    """Remove special tokens and clean up text"""
    text = text.replace("``", '"').replace("''", '"').replace("`", "'")  # convert quote symbols
    text = text.replace("n 't", "n't").replace("can not", "cannot")
    text = special_chars.sub(' ', text)
    text = re.sub(' +', ' ', text)
    return text


def load_data(filename, clean=True, encoding='utf-8'):
    """Read data from file into list of tuples"""
    dataset = []
    labels = set()
    with codecs.open(filename, 'r', encoding=encoding) as f:
        for line in f:
            if encoding is not 'utf-8':
                line = line.encode('utf-8').decode(encoding)  # convert string to utf-8 version

            line = clean_text(line)
            line = nltk.word_tokenize(line.lower())  # all the tokens and labels are split by __BLANKSPACE__
            if clean:
                sentence = clean_text(' '.join(line[1:])).split(' ')  # clean text and convert to tokens again
                sentence = [word for word in sentence if not word in english_punctuations]
            else:
                sentence = line[1:]
            label = int(line[0])
            labels.add(label)
            dataset.append((sentence, label))
    print(dataset[:5])
    return dataset, len(labels)

def statistics_info(dataset) :
    '''
    the format of dataset : list , each list is (sentence , label)
    '''
    data_size = len(dataset)
    sen_maxlen , sen_minlen , sen_avelen = 0, 100000 , 0
    word_maxlen , word_minlen , word_avelen = 0, 100000 , 0
    sen_lens , word_lens = [] , []
    for d in dataset :
        sentence , _ = d
        s_len = len(sentence)
        sen_maxlen = max(sen_maxlen , s_len)
        sen_minlen = min(sen_minlen , s_len)
        sen_lens.append(s_len)
        for word in sentence :
            w_len = len(word)
            word_maxlen = max(word_maxlen , w_len)
            word_minlen = min(word_minlen , w_len)
            word_lens.append(w_len)
    sen_avelen = np.average(sen_lens)
    word_avelen = np.average(word_lens)
    sen_info = [sen_maxlen , sen_minlen , sen_avelen]
    word_info = [word_maxlen , word_minlen , word_avelen]
    return data_size , sen_info , word_info



def build_vocab(datasets, threshold=0):
    """Build word and char vocabularies"""
    word_count = dict()
    char_vocab = set()
    for dataset in datasets:
        for words, _ in dataset:
            for word in words:
                char_vocab.update(word)  # update char vocabulary , add character not word
                word_count[word] = word_count.get(word, 0) + 1  # update word count in word dict
    word_count = reversed(sorted(word_count.items(), key=operator.itemgetter(1)))
    word_vocab = set([w[0] for w in word_count if w[1] >= threshold])
    char_vocab = char_vocab
    return word_vocab, char_vocab


def load_glove_vocab(filename):
    """Read word vocabulary from embeddings"""
    with open(filename, 'r', encoding='utf-8') as f:
        vocab = {line.strip().split()[0] for line in tqdm(f, desc='Loading GloVe vocabularies')}  # {} is set
    print('\t -- totally {} tokens in GloVe embeddings.\n'.format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """write vocabulary to file"""
    sys.stdout.write('Writing vocab to {}...'.format(filename))
    with open(filename, 'w' , encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            f.write(u'{}\n'.format(word)) if i < len(vocab) - 1 else f.write(word)
    sys.stdout.write(' done. Totally {} tokens.\n'.format(len(vocab)))


def load_vocab(filename):
    """read vocabulary from file into dict"""
    word_idx = dict()
    idx_word = dict()
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, word in enumerate(f):
            word = word.strip()
            word_idx[word] = idx
            idx_word[idx] = word
    return word_idx, idx_word

def save_filtered_vectors(vocab, glove_path, save_path, word_dim):
    """Prepare pre-trained word embeddings for dataset"""
    embeddings = np.zeros([len(vocab), word_dim])  # embeddings[0] for PAD
    scale = np.sqrt(3.0 / word_dim)
    embeddings[1] = np.random.uniform(-scale, scale, [1, word_dim])  # for UNK
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Filtering GloVe embeddings'):
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                idx = vocab[word]
                embeddings[idx] = np.asarray(embedding)
    sys.stdout.write('Saving filtered embeddings...')
    np.savez_compressed(save_path, embeddings=embeddings)
    sys.stdout.write(' done.\n')

def fit_word_to_id(word, word_vocab, char_vocab):
    """Convert word str to word index and char indices"""
    char_ids = []
    for char in word:
        char_ids += [char_vocab[char]] if char in char_vocab else [char_vocab[UNK]]
    word = word_vocab[word] if word in word_vocab else word_vocab[UNK]
    return word, char_ids

def dump_to_json(dataset, filename):
    """Save built dataset into json"""
    if dataset is not None:
        cp.dump(dataset , open(filename , 'wb'))
    sys.stdout.write('dump dataset to {}.\n'.format(filename))


def build_dataset(raw_dataset, filename, word_vocab, char_vocab, num_labels, one_hot = False):
    """Convert dataset into word/char index, make labels to be one hot vectors and dump to json file"""
    dataset = {}
    dataset['sentence'] = []
    dataset['label'] = []
    dataset['word'] = []
    for sentence, label in raw_dataset:
        words = []
        chars = []
        for word in sentence:
            word , char_ids  = fit_word_to_id(word, word_vocab, char_vocab)
            words.append(word)
            chars.append(char_ids)
        if one_hot:
            label = [1 if i == label else 0 for i in range(num_labels)]
        dataset['sentence'].append(words)
        dataset['word'].append(chars)
        dataset['label'].append(label)
    dump_to_json(dataset, filename=filename)


def build_sst(trainset , devset , testset , num_labels, data_folder, glove_vocab, glove_path):
    """Performs to build vocabularies and processed dataset"""
    # build vocabularies
    word_vocab, char_vocab = build_vocab([trainset , devset , testset])  
    if glove_vocab is None:
        glove_vocab = load_glove_vocab(glove_path)
    word_vocab = [PAD, UNK] + list(word_vocab & glove_vocab)  # distinct vocab and add PAD and UNK tokens
    write_vocab(word_vocab, filename=os.path.join(data_folder, 'words.vocab'))
    char_vocab = [PAD, UNK] + list(char_vocab)  # add PAD and UNK tokens
    write_vocab(char_vocab, filename=os.path.join(data_folder, 'chars.vocab'))
    # build embeddings
    word_vocab, _ = load_vocab(os.path.join(data_folder, 'words.vocab'))
    save_filtered_vectors(word_vocab, glove_path, os.path.join(data_folder, 'glove.filtered.npz'), word_dim=300)
    # build dataset
    char_vocab, _ = load_vocab(os.path.join(data_folder, 'chars.vocab'))
    build_dataset(trainset, os.path.join(data_folder, 'train_word2idx.json'), word_vocab, char_vocab, num_labels=num_labels,
                  one_hot=False)
    build_dataset(devset, os.path.join(data_folder, 'dev_word2idx.json'), word_vocab, char_vocab, num_labels=num_labels,
                  one_hot=False)
    build_dataset(testset , os.path.join(data_folder, 'test_word2idx.json'), word_vocab, char_vocab, num_labels=num_labels,
                  one_hot=False)


def write_statistics(data , name ,  filename) :
    '''
    data :　datasize , sen_info , word_info
    '''
    sys.stdout.write('the statistics info of {}'.format(name))
    print('the dataset size is {}'.format(data[0]))
    print('sentence info :')
    print('max length is {} , min length is {} , average length is {}'.format(data[1][0],data[1][1],data[1][2]))
    print('word info :')
    print('max length is {} , min length is {} , average length is {}'.format(data[2][0],data[2][1],data[2][2]))

    with open(filename, 'w' , encoding='utf-8') as f:
        f.write(u'the statistics info of {}\n'.format(name))
        f.write(u'the dataset size is {}\n'.format(data[0]))
        f.write(u'sentence info :\n')
        f.write(u'max length is {} , min length is {} , average length is {}\n'.format(data[1][0],data[1][1],data[1][2]))
        f.write(u'word info :')
        f.write(u'max length is {} , min length is {} , average length is {}\n'.format(data[2][0],data[2][1],data[2][2]))


def prepro_sst(source_dir , target_dir , glove_path, \
                    glove_vocab=None, mode=1, if_contain_phrase = False):
    print('Process sst{} dataset...'.format(mode))
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # load dataset
    name = 'fine' if mode == 1 else 'binary'
    if not if_contain_phrase:
        train_set, num_labels = load_data(os.path.join(source_dir, 'stsa.{}.train'.format(name)))
    else:
        train_set, num_labels = load_data(os.path.join(source_dir, 'stsa.all.train'.format(name)))
    dev_set, _ = load_data(os.path.join(source_dir,  'stsa.{}.dev'.format(name)))
    test_set, _ = load_data(os.path.join(source_dir, 'stsa.{}.test'.format(name)))
    # print dataset base information
    datasize , sen_info , word_info = statistics_info(train_set + dev_set + test_set)
    write_statistics([datasize , sen_info , word_info] , name , filename=os.path.join(target_dir, 'stat.info'))
    print('the statistics info of {}'.format(name))
    print('the dataset size is {}'.format(datasize))
    print('sentence info :')
    print('max length is {} , min length is {} , average length is {}'.format(sen_info[0],sen_info[1],sen_info[2]))
    print('word info :')
    print('max length is {} , min length is {} , average length is {}'.format(word_info[0],word_info[1],word_info[2]))
    # build general
    build_sst(train_set, dev_set, test_set, num_labels, target_dir, glove_vocab, glove_path)
    print()


if __name__ == '__main__':

    folder_name = 'sst5'
    dataset_name = 'sst5'
    # data_dir = '/home/janady/retrieval_model/model/mode-lstm/global_cnn-master/data'
    source_dir = 'data/sst5'
    target_dir = 'data/sst5/embeddings'
    glove_path = os.path.join(source_dir, 'glove.840B.300d.txt')
    glove_vocab = load_glove_vocab(glove_path)
    # process sst5 dataset
    prepro_sst(source_dir ,target_dir , glove_path, glove_vocab, mode=1, if_contain_phrase=True)