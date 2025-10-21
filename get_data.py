import numpy as np
import torch

from utils import read_fasta_lengths, pre_feature


def PAAC_embedding(sequence):
    paac = {
        'A': [0.62, -0.5, 15.0],
        'R': [-2.53, 3.0, 101.0],
        'N': [-0.78, 0.2, 58.0],
        'D': [-0.9, 3.0, 59.0],
        'C': [0.29, -1.0, 47.0],
        'Q': [-0.85, 0.2, 72.0],
        'E': [-0.74, 3.0, 73.0],
        'G': [0.48, 0.0, 1.0],
        'H': [-0.4, -0.5, 82.0],
        'I': [1.38, -1.8, 57.0],
        'L': [1.06, -1.8, 57.0],
        'K': [-1.5, 3.0, 73.0],
        'M': [0.64, -1.3, 75.0],
        'F': [1.19, -2.5, 91.0],
        'P': [0.12, 0.0, 42.0],
        'S': [-0.18, 0.3, 31.0],
        'T': [-0.05, -0.4, 45.0],
        'W': [0.81, -3.4, 130.0],
        'Y': [0.26, -2.3, 107.0],
        'V': [1.08, -1.5, 43.0],
        'X': [0.00, 0.00, 0.00]
    }

    fea = []
    for seq in sequence:
        fea.append(paac[seq])
    return np.array(fea)

def ZSCALE(sequence):
    zscale = {
        'A': [0.24, -2.32, 0.60, -0.14, 1.30],  # A
        'C': [0.84, -1.67, 3.71, 0.18, -2.65],  # C
        'D': [3.98, 0.93, 1.93, -2.46, 0.75],  # D
        'E': [3.11, 0.26, -0.11, -0.34, -0.25],  # E
        'F': [-4.22, 1.94, 1.06, 0.54, -0.62],  # F
        'G': [2.05, -4.06, 0.36, -0.82, -0.38],  # G
        'H': [2.47, 1.95, 0.26, 3.90, 0.09],  # H
        'I': [-3.89, -1.73, -1.71, -0.84, 0.26],  # I
        'K': [2.29, 0.89, -2.49, 1.49, 0.31],  # K
        'L': [-4.28, -1.30, -1.49, -0.72, 0.84],  # L
        'M': [-2.85, -0.22, 0.47, 1.94, -0.98],  # M
        'N': [3.05, 1.62, 1.04, -1.15, 1.61],  # N
        'P': [-1.66, 0.27, 1.84, 0.70, 2.00],  # P
        'Q': [1.75, 0.50, -1.44, -1.34, 0.66],  # Q
        'R': [3.52, 2.50, -3.50, 1.99, -0.17],  # R
        'S': [2.39, -1.07, 1.15, -1.39, 0.67],  # S
        'T': [0.75, -2.18, -1.12, -1.46, -0.40],  # T
        'V': [-2.59, -2.64, -1.54, -0.85, -0.02],  # V
        'W': [-4.36, 3.94, 0.59, 3.44, -1.59],  # W
        'Y': [-2.54, 2.44, 0.43, 0.04, -1.47],  # Y
        '-': [0.00, 0.00, 0.00, 0.00, 0.00],  # -
    }
    fea = []
    for seq in sequence:
        fea.append(zscale[seq])
    return np.array(fea)


def BLOSUM62(sequence):
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }
    # encodings = []
    # header = ['#']
    # for i in range(1, len(sequence) * 20 + 1):
    #     header.append('blosum62.F' + str(i))
    # encodings.append(header)
    #
    # code = []
    # for seq in sequence:
    #     code = code + blosum62[seq]
    # encodings.append(code)
    # return encodings
    fea = []
    for seq in sequence:
        fea.append(blosum62[seq])
    return np.array(fea)


def BINARY(seqence):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = ['#']
    for i in range(1, len(seqence) * 20 + 1):
        header.append('BINARY.F' + str(i))
    encodings.append(header)

    fea = []
    for aa in seqence:
        code = []
        # if aa == '-':
        #     code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #     continue
        for aa1 in AA:
            tag = 1 if aa == aa1 else 0
            code.append(tag)
        fea.append(code)
    return np.array(fea)



def data(path1, path2):
    data_result = np.load(path2, allow_pickle=True)

    data_id = data_result[:, 0]
    feature = data_result[:, 1]

    sequence, max_len = read_fasta_lengths(path1)
    esm_feature = pre_feature(feature)

    # 得到节点特征
    datas = []
    for seq in sequence:
        bin_feature = BINARY(seq)
        blo_feature = BLOSUM62(seq)
        zsl_feature = ZSCALE(seq)
        paac_feature = PAAC_embedding(seq)

        fea_matrx = np.hstack([bin_feature, blo_feature, zsl_feature])
        # 补齐到最大长度
        if fea_matrx.shape[0] <= max_len:
            zeros = np.zeros((max_len - fea_matrx.shape[0], fea_matrx.shape[1]))
            padded_data = np.vstack((fea_matrx, zeros))
        datas.append(padded_data.astype(np.float32))
    datas = np.array(datas)

    labels = data_result[:, 2]

    Test_NUM = 1669
    Train_NUM = 5842
    Valid_NUM = 835

    # Test_NUM = 1424
    # Train_NUM = 1424
    # Valid_NUM = 708

    # permuted_indices = np.arange(SPLIT_NUM, len(data_id)) # 通过设置一个数组存储索引并打乱
    train_index = np.arange(Test_NUM + Valid_NUM, Train_NUM + Test_NUM + Valid_NUM)
    valid_index = np.arange(Test_NUM, Test_NUM + Valid_NUM)
    test_index = np.arange(Test_NUM)

    train_set = [[esm_feature[i] for i in train_index], datas[train_index], labels[train_index]]
    valid_set = [[esm_feature[i] for i in valid_index], datas[valid_index], labels[valid_index]]
    test_set = [[esm_feature[i] for i in test_index], datas[test_index], labels[test_index]]

    a = train_set[2]
    b = valid_set[2]
    c = test_set[2]
    train_ones = np.sum(a == 1) + np.sum(b == 1)
    train_zero = np.sum(a == 0) + np.sum(b == 0)
    test_ones = np.sum(c == 1)
    test_zero = np.sum(c == 0)
    return train_set, valid_set, test_set
