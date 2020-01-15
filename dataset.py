import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import os
import numpy as np


class CriticDataset2(Dataset):
    
    def __init__(self, 
        x_real, 
        x_gen, 
        transform=None,
        class_len=None):

        xy_real = TensorDataset(x_real, torch.ones(len(x_real)))
        xy_gen = TensorDataset(x_gen, torch.zeros(len(x_gen)))

        self.xy = ConcatDataset([xy_real, xy_gen])
        self.transform = transform

    def __len__(self):

        return len(self.xy) 

    def __getitem__(self, idx):

        x, y = self.xy[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

class CriticDataset3(Dataset):
    
    def __init__(self, 
        x,
        transform=None,
        class_len=None):

        self.x = TensorDataset(x)
        self.transform = transform

    def __len__(self):

        return len(self.x) 

    def __getitem__(self, idx):

        x = self.x[idx][0]

        if self.transform:
            x = self.transform(x)

        return x

def get_real_data(real_datadir):

    xtrva_real = datasets.CIFAR10(root=real_datadir, download=True)
    xtr_real_ds, xva_real_ds = random_split(xtrva_real, [45000, 5000])
    xtr_real = torch.from_numpy(xtrva_real.data[xtr_real_ds.indices].transpose((0, 3, 1, 2)))
    xva_real = torch.from_numpy(xtrva_real.data[xva_real_ds.indices].transpose((0, 3, 1, 2)))
    xte_real = torch.from_numpy(datasets.CIFAR10(root=real_datadir, download=True, train=False).data.transpose((0, 3, 1, 2)))

    return xtr_real, xva_real, xte_real

def get_gen_data(gen_datadir):

    xgen = np.load(os.path.join(gen_datadir, 'gen_samples.npz'))
    xtr_gen, xva_gen, xte_gen = torch.from_numpy(xgen['train_data']), torch.from_numpy(xgen['valid_data']), torch.from_numpy(xgen['test_data'])

    return xtr_gen, xva_gen, xte_gen

def get_loaders(
        real_datadir,
        gen_datadir,
        batch_size,
        test_batch_size,
        use_feature,
        kwargs):
    """
    Returns train, validation, test loaders
    split: train/valid/test
    batch_size: for train only, returns full dataset for valid and test splits
    Each iteration of the loader will give a tuple of tensor tuple and label tensor: (u, s, v), labels.
    The first dimensions of all those tensors is the batch dimension.
    """

    xtr_real, xva_real, xte_real = get_real_data(real_datadir)
    xtr_gen, xva_gen, xte_gen = get_gen_data(gen_datadir)
    xte_gen = xte_gen[:10000]

    cr_ds = CriticDataset2
    if use_feature:
        trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    else:
        trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = cr_ds(
        xtr_real,
        xtr_gen, 
        transform=trans,
        class_len=15)

    valid_dataset = cr_ds(
        xva_real, 
        xva_gen, 
        transform=trans,
        class_len=3)

    test_dataset = cr_ds(
        xte_real, 
        xte_gen,
        transform=trans,
        class_len=2)

    train_loader = DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = DataLoader(valid_dataset,
        batch_size=test_batch_size, shuffle=True, **kwargs)

    test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader

def get_eval_loaders(
        real_datadir,
        gen_datadir,
        batch_size,
        test_batch_size,
        use_feature,
        kwargs,
        test_idx=0):
    """
    Returns train, validation, test loaders
    split: train/valid/test
    batch_size: for train only, returns full dataset for valid and test splits
    Each iteration of the loader will give a tuple of tensor tuple and label tensor: (u, s, v), labels.
    The first dimensions of all those tensors is the batch dimension.
    """

    xtr_real, xva_real, xte_real = get_real_data(real_datadir)
    xtr_gen, xva_gen, xte_gen = get_gen_data(gen_datadir)
    xte_gen = xte_gen[test_idx*10000:(test_idx+1)*10000]

    cr_ds = CriticDataset3
    if use_feature:
        trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(299),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
    else:
        trans = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = cr_ds(
        xtr_real,
        transform=trans,
        class_len=15)

    valid_dataset = cr_ds(
        xva_real, 
        transform=trans,
        class_len=3)

    test_dataset = cr_ds(
        xte_real, 
        transform=trans,
        class_len=2)

    real_train_loader = DataLoader(train_dataset,
            batch_size=batch_size, shuffle=True, **kwargs)

    real_valid_loader = DataLoader(valid_dataset,
        batch_size=test_batch_size, shuffle=True, **kwargs)

    real_test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size, shuffle=False, **kwargs)

    train_dataset = cr_ds(
        xtr_gen, 
        transform=trans,
        class_len=15)

    valid_dataset = cr_ds(
        xva_gen, 
        transform=trans,
        class_len=3)

    test_dataset = cr_ds(
        xte_gen,
        transform=trans,
        class_len=2)

    gen_train_loader = DataLoader(train_dataset,
            batch_size=batch_size, shuffle=False, **kwargs)

    gen_valid_loader = DataLoader(valid_dataset,
        batch_size=test_batch_size, shuffle=False, **kwargs)

    gen_test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size, shuffle=False, **kwargs)

    return real_train_loader, real_valid_loader, real_test_loader, gen_train_loader, gen_valid_loader, gen_test_loader


