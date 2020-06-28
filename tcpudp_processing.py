
import json, random
import math
import os
import torch
from Nixae import Constants
import numpy as np

class DataManager(object):

    def __init__(self, train_src, grained, max_word_seq_len, keep_case, fold_num, min_word_count, save_data):
        self.train_src = train_src
        self.max_word_seq_len = max_word_seq_len
        self.fold_num = fold_num
        self.keep_case = keep_case
        self.min_word_count = min_word_count
        self.save_data = save_data

        datas, labels = self.read_instances_from_file(train_src, grained, self.max_word_seq_len, self.keep_case)
        nums_every_fold = int(len(labels) * 0.2)
        if fold_num not in [x for x in range(5)]:
            print('[Error] fold_num is not correct.')
            exit()

        all_datas = []
        all_labels = []
        for i in range(5):
            all_datas.append(datas[i*nums_every_fold:(i+1)*nums_every_fold])
            all_labels.append(labels[i*nums_every_fold:(i+1)*nums_every_fold])

        self.test_datas = all_datas[fold_num]
        self.test_labels = all_labels[fold_num]

        self.valid_datas = all_datas[fold_num - 1]
        self.valid_labels = all_labels[fold_num - 1]

        self.train_datas = all_datas[fold_num - 2] + all_datas[fold_num - 3] + all_datas[fold_num - 4]
        self.train_labels = all_labels[fold_num - 2] + all_labels[fold_num - 3] + all_labels[fold_num - 4]

    def read_instances_from_file(self,inst_file, grained, max_sent_len, keep_case):
        ''' Convert file into word seq lists and vocab '''
        alldata_insts = []
        labels = []
        window_size = 1

        trimmed_sent_count = 0
        lable_list = [x for x in range(grained)]
        with open(inst_file) as f:
            for sent in f:
                word_inst = sent.strip().split(' ')
                if len(word_inst) > max_sent_len:
                    trimmed_sent_count += 1

                if ((len(word_inst)!=0) and (int(word_inst[-1]) in lable_list)):
                    data_list = []
                    labels += [word_inst[-1]]
                    data = word_inst[0:-1]
                    if len(data) >= max_sent_len:
                        data_list = data[0:max_sent_len]
                    else:
                        data_list = data[0:max_sent_len] + [0] * (max_sent_len - len(data))

                    alldata_insts += [data_list]

                else:
                    alldata_insts += [None]
                    print("NONE")
                    exit()

        print('[Info] Get {} instances from {}'.format(len(alldata_insts), inst_file))


        if trimmed_sent_count > 0:
            print('[Warning] {} instances are trimmed to the max sentence length {}.'
                  .format(trimmed_sent_count, max_sent_len))

        cc = list(zip(alldata_insts, labels))
        random.seed(1)
        random.shuffle(cc)
        alldata_insts[:], labels[:] = zip(*cc)

        return alldata_insts, labels

    def convert_instance_to_idx_seq(self, word_insts):
        ''' Mapping words to idx sequence. '''
        return [[int(w) for w in s] for s in word_insts]


    def getdata(self):
        print('[Info] Convert source word instances into sequences of word index.')

        self.train_src_insts = self.convert_instance_to_idx_seq(self.train_datas)
        self.valid_src_insts = self.convert_instance_to_idx_seq(self.valid_datas)
        self.test_src_insts = self.convert_instance_to_idx_seq(self.test_datas)

        value_data = [self.train_src_insts, self.valid_src_insts, self.test_src_insts,
                    self.train_labels, self.valid_labels, self.test_labels]


        for data in value_data:
            print('data num:', len(data))


        data = {
            'settings': [self.fold_num, self.max_word_seq_len, self.min_word_count],
            'train': {
                'src': self.train_src_insts,
                'lbl': self.train_labels},
            'valid': {
                'src': self.valid_src_insts,
                'lbl': self.valid_labels},
            'test': {
                'src': self.test_src_insts,
                'lbl': self.test_labels},
        }

        ## save
        if (self.save_data):
            save_data = self.save_data + str(self.fold_num)
            print('[Info] Dumping the processed data to pickle file', save_data)
            torch.save(data, save_data)
            print('[Info] Finish.', self.fold_num)

        return data


if __name__ == '__main__':
    train_src = 'data/udp_any_data_8.txt'
    maxlenth = 32
    keep_case = False
    fold_num = 0
    min_word_count = 0
    save_data = ''
    grained = 8
    datamanager = DataManager(train_src, grained, maxlenth, keep_case, fold_num, min_word_count, save_data)
    datamanager.getdata()