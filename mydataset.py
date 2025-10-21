import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class MyDataset(Dataset):

    def __init__(self, data):
        self.feature1 = data[0]
        self.feature2 = data[1]
        self.labels = data[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.tensor(self.feature1[index]), torch.tensor(self.feature2[index]), self.labels[index]

def collate_fn(batch):
    data, labels = map(list, zip(*batch))
    if len(batch) % 2 != 0:
        data = data[:-1]
        labels = labels[:-1]
    # data = torch.tensor(data)
    labels = torch.tensor(labels)
    device = torch.device("cuda")
    data1_ls = []
    data2_ls = []
    label1_ls = []
    label2_ls = []
    label_ls = []
    batch_size = len(batch)
    mid = batch_size // 2
    for i in range(mid):
        data1, label1 = data[i], labels[i]
        data2, label2 = data[i + int(batch_size / 2)], labels[i + int(batch_size / 2)]
        label1_ls.append(label1.unsqueeze(0))
        label2_ls.append(label2.unsqueeze(0))
        label = (label1 ^ label2)
        data1_ls.append(data1)
        data2_ls.append(data2)
        label_ls.append(label.unsqueeze(0))
    data1 = data[:mid]
    data2 = data[mid:]
    label = torch.cat(label_ls)
    label1 = torch.cat(label1_ls)
    label2 = torch.cat(label2_ls)
    return Batch.from_data_list(data), Batch.from_data_list(data1), Batch.from_data_list(data2), label, label1, label2

def collate_fn1(batch):
    device = 'cuda'
    esm, data, labels = zip(*batch)
    if len(batch) % 2 != 0:
        esm = esm[:-1]
        data = data[:-1]
        labels = labels[:-1]

    batch_size = len(batch) // 2
    esm1_ls = esm[:batch_size]
    esm2_ls = esm[batch_size:]
    data1_ls = data[:batch_size]
    data2_ls = data[batch_size:]
    labels1_ls = labels[:batch_size]
    labels2_ls = labels[batch_size:]
    esm = torch.stack(esm).to(device)
    esm1 = torch.stack(esm1_ls).to(device)
    esm2 = torch.stack(esm2_ls).to(device)
    data = torch.stack(data).to(device)
    data1 = torch.stack(data1_ls).to(device)
    data2 = torch.stack(data2_ls).to(device)

    label_ = []
    for i in range(batch_size):
        label_.append(labels1_ls[i] ^ labels2_ls[i])
    labels = torch.tensor(labels).to(device, dtype=torch.float32)
    label1 = torch.tensor(labels1_ls).to(device, dtype=torch.float32)
    label2 = torch.tensor(labels2_ls).to(device, dtype=torch.float32)
    label_ = torch.tensor(label_).to(device, dtype=torch.float32)

    esm_dic = {'esm': esm, 'esm1': esm1, 'esm2': esm2}
    data_dic = {'data': data, 'data1': data1, 'data2': data2}
    label_dic = {'labels': labels, 'label1': label1, 'label2': label2, 'label_': label_}
    return esm_dic, data_dic, label_dic

def collate_fn2(batch):
    device = 'cuda'
    esm, data, labels = map(list, zip(*batch))
    esm = torch.stack(esm).to(device)
    data = torch.stack(data).to(device)
    labels = torch.tensor(labels).to(device, dtype=torch.float32)
    return esm, data, labels.unsqueeze(1)





