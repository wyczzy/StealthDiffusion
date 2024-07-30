import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import dataset_folder


def get_dataset(opt):
    dset_lst = []
    if opt.genimage and opt.classes == "":
        root = opt.dataroot
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    elif opt.genimage and opt.classes != "":
        p1 = os.path.dirname(opt.dataroot[:-2])
        for cls in opt.classes:
            root = p1 + '/' + cls
            root = os.path.join(root, opt.train_split)
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    else:
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)

def get_dataset_pha(opt):
    dset_lst = []
    if opt.genimage and opt.classes[0] == "":
        root = opt.dataroot
        dset = dataset_folder(opt, root)
        dset_lst.append(dset)
    elif opt.genimage and opt.classes[0] != "":
        p1 = os.path.dirname(opt.dataroot[:-2])
        for cls in opt.classes:
            root = p1 + '/' + cls
            root = os.path.join(root, opt.train_split)
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    else:
        for cls in opt.classes:
            root = opt.dataroot + '/' + cls
            dset = dataset_folder(opt, root)
            dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst)




def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    w = torch.tensor([0.75, 0.25])
    # w = torch.tensor([0.8, 0.2])
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler


def create_dataloader(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader

def create_dataloader_pha(opt):
    shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
    dataset = get_dataset_pha(opt)
    sampler = get_bal_sampler(dataset) if opt.class_bal else None

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=int(opt.num_threads))
    return data_loader