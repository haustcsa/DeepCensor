import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc, f1_score, precision_score, \
    recall_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import torchvision


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# save tensors as pics.
def save_pic(tensor, save_path):
    for i in range(tensor.size(0)):
        torchvision.utils.save_image(tensor[i, :, :, :], save_path + '/{}.png'.format(i))


# 计算多分类任务的指标
def cal_metrics(y_true_all, y_pred_all, num_classes=5):
    # 初始化
    aucs = []
    tprs = []
    tprs_dict = {}
    accuracies = []

    # 对每个类进行 One-vs-Rest AUC 计算
    for i in range(num_classes):
        # 将标签转为二分类标签
        y_true_bin = (y_true_all == i).astype(int)
        y_pred_bin = y_pred_all[:, i]  # 对应类别的预测值

        # 计算 AUC
        fprs, tprs, thresholds = roc_curve(y_true_bin, y_pred_bin)
        auc_value = auc(fprs, tprs)
        aucs.append(auc_value)

        # 计算 TPR（召回率）
        eer = brentq(lambda x: 1. - x - interp1d(fprs, tprs)(x), 0., 1.)
        tprs_dict[i] = {
            'TPR': tprs,
            'EER': eer
        }

        # 计算每个类别的准确率
        # 对于每个类别，计算它的准确率
        correct_preds = (y_pred_all.argmax(axis=1) == y_true_all)  # 获取预测正确的样本
        correct_class_preds = correct_preds * (y_true_all == i)  # 仅计算该类别的正确预测
        accuracy = np.sum(correct_class_preds) / np.sum(y_true_all == i)  # 该类别的准确率
        accuracies.append(accuracy)

    # 计算总的 AUC 和 EER
    mean_auc = np.mean(aucs)

    # TPR 根据不同的 fpr 阈值进行切分
    thresholds = [1e-2, 1e-3, 1e-4]
    tpr_values = {}
    for threshold in thresholds:
        tpr_values[threshold] = []
        for i in range(num_classes):
            fprs, tprs, _ = roc_curve((y_true_all == i).astype(int), y_pred_all[:, i])
            ind = np.where(fprs > threshold)[0][0] if len(np.where(fprs > threshold)[0]) > 0 else -1
            tpr_values[threshold].append(tprs[ind - 1] if ind > 0 else 0)

    # 计算 Average Precision (AP)
    ap = average_precision_score(y_true_all, y_pred_all, average="macro")

    return ap, mean_auc, tprs_dict, tpr_values, accuracies


# plot ROC, compute AUC and EER
def plot_ROC(y_true, y_pred, num_classes=5):
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        y_true_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        # 绘制每个类别的ROC曲线
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f}, EER = {eer:.2f})')

    plt.plot([0, 1], [0, 1], 'r--')  # 对角线
    plt.title("Receiver Operating Characteristic (ROC) - Multi-Class")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.savefig('./output/roc.png', bbox_inches='tight')
    plt.show()

    return eer, roc_auc
