import numpy as np
from Bio import SeqIO
from sklearn.metrics import roc_auc_score


def pre_feature(feature):
    data = []
    for i in range(len(feature)):
        feature1 = feature[i]
        data.append(feature1)
    return np.array(data)

# 读取fasta文件
def read_fasta_lengths(file_path):
    sequence = []
    # seqs = []
    max_len = 0
    for record in SeqIO.parse(file_path, "fasta"):
        seq_len = len(record.seq)
        if seq_len > max_len:
            max_len = seq_len
        sequence.append(str(record.seq))
    return np.array(sequence), max_len#, seqs


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc_list = numerator / denominator

    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    mcc = mcc_list[max_index]
    roc_auc = roc_auc_score(real_score, predict_score)
    a = [f1_score, accuracy, recall, precision, mcc, roc_auc]  # auc[0, 0], aupr[0, 0],specificity,
    res = [f"{num:.4f}" for num in a]
    return res
# 获取原始长度
def original_feature(feature, truncated_len):
    # 初始化结果列表
    feature_truncated = []
    # 遍历每个二维数组，并根据对应的大小进行截取
    for i in range(len(truncated_len)):
        feature1 = feature[i]
        feature2 = feature1.reshape(feature1.shape[1], feature1.shape[2])
        # feature_truncated.append(feature2[1:truncated_len[i] + 1, :])
        feature_truncated.append(feature2[1:-1, :])
    return np.array(feature_truncated)