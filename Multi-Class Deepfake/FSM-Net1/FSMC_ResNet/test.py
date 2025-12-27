import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader


from network.data import TestDataset
from network.transform import Data_Transforms
from network.MainNet import MainNet
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize

def main():
    args = parse.parse_args()
    test_txt_path = args.test_txt_path
    batch_size = args.batch_size
    model_path = args.model_path
    num_classes = args.num_classes

    torch.backends.cudnn.benchmark = True

    # -----create test data----- #
    test_data = TestDataset(txt_path=test_txt_path, test_transform=Data_Transforms['test'])
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # -----create model----- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MainNet(num_classes).to(device)

    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    correct_test = 0.0
    total_test_samples = 0.0
    y_true = []
    y_pred = []
    predict_test_list = []

    correct_class_count = np.zeros(num_classes)
    total_class_count = np.zeros(num_classes)
    # 混淆矩阵初始化
    confusion_matrix_class = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img_rgb, labels_test = data
            img_rgb = img_rgb.cuda()
            labels_test = labels_test.cuda()

            # Feed data to model
            pre_test = model(img_rgb)

            # Prediction
            _, pred = torch.max(pre_test.data, 1)

            # Count all testing samples and correct predictions
            total_test_samples += labels_test.size(0)
            correct_test += (pred == labels_test).squeeze().sum().cpu().numpy()

            # Append predictions and labels for evaluation metrics
            y_true.extend(labels_test.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            # For AUC computation
            pre_test_probs = torch.nn.functional.softmax(pre_test, dim=1)
            predict_test_list.extend(pre_test_probs.cpu().numpy())
            # 更新每个类别的正确预测数和总样本数
            for label in range(num_classes):
                total_class_count[label] += torch.sum(labels_test == label).cpu().numpy()
                correct_class_count[label] += torch.sum((pred == label) & (labels_test == label)).cpu().numpy()

            # 更新混淆矩阵
            cm = confusion_matrix(labels_test.cpu().numpy(), pred.cpu().numpy(), labels=np.arange(num_classes))
            confusion_matrix_class += cm

        # Calculate accuracy
        acc = correct_test / total_test_samples
        print("Testing Accuracy (ACC): {:.2%}".format(acc))
        # Calculate AUC score for each class
        y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
        y_pred_probs = np.array(predict_test_list)  # 将预测概率转换为 NumPy 数组

        # 计算每个类别的 AUC
        auc_per_class = roc_auc_score(y_true_binarized, y_pred_probs, average=None, multi_class="ovr")
        for i, auc in enumerate(auc_per_class):
            print(f"AUC for class {i}: {auc:.4f}")

        # 计算宏观平均 AUC
        auc_macro = roc_auc_score(y_true_binarized, y_pred_probs, average="macro", multi_class="ovr")
        print(f"AUC (Macro): {auc_macro:.4f}")

        # 计算每个类别的准确率
        for i in range(num_classes):
            class_acc = correct_class_count[i] / total_class_count[i] if total_class_count[i] > 0 else 0
            print(f"Accuracy for class {i}: {class_acc:.2%}")

            # Calculate recall for each class
            tp = confusion_matrix_class[i, i]  # True Positive for class i
            fn = np.sum(confusion_matrix_class[i, :]) - tp  # False Negative for class i
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"Recall for class {i}: {recall:.2%}")

            # Calculate Precision for each class 你
            fp = np.sum(confusion_matrix_class[:, i]) - tp  # False Positive for class i
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            print(f"Precision for class {i}: {precision:.2%}")

            # Calculate F1 score for each class
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            print(f"F1 Score for class {i}: {f1:.4f}")
        conf_matrix = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix111:\n", conf_matrix)
        # Optional: plot ROC curve
        #plot_ROC(y_true_binarized, y_pred_probs)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--test_txt_path', '-tp', type=str, default='C:/dataset/test_updated_v4.txt')
    parse.add_argument('--model_path', '-mp', type=str, default='./output/FSMc-resnet/best.pkl')
    parse.add_argument('--num_classes', '-nc', type=int, default=5)

    label_test_list = []
    predict_test_list = []

    main()
