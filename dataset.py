import numpy as np
import torch
import torch.utils.data

from Nixae import Constants

def paired_collate_fn(insts):
    src_insts, src_lbls = list(zip(*insts))
    src_insts = collate_fn(src_insts)
    src_lbls = torch.LongTensor([int(x) for x in src_lbls])
    return src_insts, src_lbls

def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''
    max_len = max(max(len(inst) for inst in insts), Constants.MAX_LEN)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, src_insts=None, src_lbls=None):

        assert src_insts
        self._src_insts = src_insts
        self._src_lbls = src_lbls

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)


    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self._src_insts[idx], self._src_lbls[idx]
