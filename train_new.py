'''
This script handling the training process.
'''

import argparse
import math
import time
import os



from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import Nixae.Constants as Constants
from dataset import TranslationDataset, paired_collate_fn
from Nixae.Models import Nixae
from torch.optim import lr_scheduler
import numpy as np
from tcpudp_processing import DataManager
from sklearn.metrics import balanced_accuracy_score


def train_epoch(model, training_data, optimizer, device):
    ''' Epoch operation in training phase'''
    model.train()

    total_loss = 0
    n_total = 0
    n_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        src_seq, src_lbl = map(lambda x: x.to(device), batch)

        # forward
        optimizer.zero_grad()
        pred = model(src_seq)

        sample = (torch.max(pred, 1)[1] == src_lbl).sum().item()
        n_correct += sample
        n_total += src_lbl.shape[0]


        # backward
        loss = F.cross_entropy(pred, src_lbl)
        loss.backward()

        # update parameters
        optimizer.step()
        # note keeping
        total_loss += loss.item()

    train_acc = 100. * n_correct / n_total
    print("training acc is: ", train_acc, "training loss is: ", total_loss)
    return total_loss, train_acc

def eval_epoch(model, validation_data, scheduler, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            # prepare data
            src_seq, src_lbl = [x.to(device) for x in batch]

            # forward
            pred = model(src_seq)

            y_predict = torch.cat([y_predict, torch.max(pred, 1)[1]], 0)
            y_true = torch.cat([y_true, src_lbl], 0)

            loss = F.cross_entropy(pred, src_lbl)
            total_loss += loss.item()
    scheduler.step(total_loss)

    y_true = y_true.cpu().numpy().tolist()
    y_predict = y_predict.cpu().numpy().tolist()

    y_true_trans = np.array(y_true)
    y_predict_trans = np.array(y_predict)


    acc = balanced_accuracy_score(y_true_trans, y_predict_trans)

    valid_acc = 100. * acc
    print("validation acc is: ", valid_acc, "validation loss is: ", total_loss)
    return total_loss, valid_acc


def test_epoch(model, testing_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    with torch.no_grad():
        for batch in tqdm(
                testing_data, mininterval=2,
                desc='  - (Testing) ', leave=False):

            # prepare data
            src_seq, src_lbl = [x.to(device) for x in batch]

            # forward
            pred = model(src_seq)

            y_predict = torch.cat([y_predict, torch.max(pred, 1)[1]], 0)
            y_true = torch.cat([y_true, src_lbl], 0)

            loss = F.cross_entropy(pred, src_lbl)
            total_loss += loss.item()

    y_true = y_true.cpu().numpy().tolist()
    y_predict = y_predict.cpu().numpy().tolist()

    y_true_trans = np.array(y_true)
    y_predict_trans = np.array(y_predict)


    acc = balanced_accuracy_score(y_true_trans, y_predict_trans)

    test_acc = 100. * acc
    print("Testing acc is: ", test_acc, "testing loss is: ", total_loss)
    return total_loss, test_acc




def train(model, training_data, validation_data, testing_data, optimizer, device, opt, scheduler):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None
    log_test_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'
        log_test_file = opt.log + '.test.log'


        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file, log_test_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf, open(log_test_file, 'w') as log_tef:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')
            log_tef.write('epoch,loss,ppl,accuracy\n')

    valid_accus = []
    test_accus=[]

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=train_accu,
                  elapse=(time.time()-start)/60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(
        	model, validation_data, scheduler, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(valid_loss, 100)), accu=valid_accu,
                    elapse=(time.time()-start)/60))
        valid_accus += [valid_accu]


        start = time.time()
        test_loss, test_accu = test_epoch(
        	model, testing_data, device)
        print('  - (Testing) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(
                    ppl=math.exp(min(test_loss, 100)), accu=test_accu,
                    elapse=(time.time()-start)/60))
        test_accus += [test_accu]



        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i,
            'valid_accu':valid_accu,
            'optimizer':optimizer}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + 'emb_' + str(opt.d_model) + '_pl_'+ str(opt.max_word_seq_len) + '_brn_' + str(opt.brn) + '_fold_'+ str(opt.fold_num) + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + 'emb_' + str(opt.d_model) + '_pl_'+ str(opt.max_word_seq_len) + '_brn_' + str(opt.brn) + '_fold_'+ str(opt.fold_num) + '_best.chkpt'
                valid_accus.sort(reverse = True)
                saved_accu = torch.load(model_name)['valid_accu'] if os.path.exists(model_name) else -1
                if (valid_accu >= min(valid_accus[0:1])) and (epoch_i) > 1 and valid_accu > saved_accu:
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')


        if log_train_file and log_valid_file and log_test_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf, open(log_test_file, 'a') as log_tef:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu))
                log_tef.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=test_loss,
                    ppl=math.exp(min(test_loss, 100)), accu=100*test_accu))


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str,default='')
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-d_model', type=int, default=200)
    parser.add_argument('-brn', type=int, default=3)
    parser.add_argument('-log', default=None)
    parser.add_argument('-save_model', default='model/')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true')
    parser.add_argument('--grained', type=int, default=8)  ##### class
    parser.add_argument('-train_src', default='data/udp_any_data_8.txt')
    parser.add_argument('-save_data', default='')
    parser.add_argument('-max_word_seq_len', type=int, default=128)
    parser.add_argument('-min_word_count', type=int, default=0)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-fold_num', type=int, default=0)   #############fold
    parser.add_argument('-CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('-lr', type=float, default=1e-3)


    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    #opt.cuda = opt.no_cuda

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.CUDA_VISIBLE_DEVICES

    os.makedirs(opt.save_model, exist_ok=True)

    #========= Processing Dataset =========#
    if(opt.data==''):
        datamanager = DataManager(opt.train_src,opt.grained,opt.max_word_seq_len, opt.keep_case,
                                  opt.fold_num,opt.min_word_count,opt.save_data)
        data = datamanager.getdata()

    # #========= Loading Dataset =========#
    else:
        data = torch.load(opt.data)

    print('now seq len:', opt.max_word_seq_len)
    training_data, validation_data, testing_data = prepare_dataloaders(data, opt)

    #========= Preparing Model =========#

    print(opt)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    nixae = Nixae(
        opt.max_word_seq_len,
        brn=opt.brn,
        label=opt.grained,
        d_model=opt.d_model,
        dropout=opt.dropout).to(device)

    learningrate = opt.lr

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, nixae.parameters()), lr=learningrate, weight_decay = 0.0003)   
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, verbose=True, min_lr=1e-5)

    print(nixae)
    train(nixae, training_data, validation_data, testing_data, optimizer, device, opt, scheduler)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_insts=data['train']['src'],
            src_lbls=data['train']['lbl']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)


    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_insts=data['valid']['src'],
            src_lbls=data['valid']['lbl']),

        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)


    test_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_insts=data['test']['src'],
            src_lbls=data['test']['lbl']),

        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    main()
