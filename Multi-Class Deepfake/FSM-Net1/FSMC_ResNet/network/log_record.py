
import os

def save_weights(w_rgb, w_res, epoch_num, log_dir):
    filename = open(os.path.join(log_dir, 'Weights.txt'), 'a')   # 'a': append, not overwrite;  'w': overwrite previous data
    fusion_data_save = 'Epoch_' + str(epoch_num+1) + ':' + str(w_rgb) + ' ' + str(w_res) + '\n'
    filename.write(fusion_data_save)


def save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch_num, log_dir):
    filename = open(os.path.join(log_dir, 'final_results.txt'), 'a')
    fusion_data_save = 'Epoch_' + str(epoch_num+1) + ':' + ' ' + 'acc:%.4f'%epoch_acc + ' ' + 'ap:%.4f'%ap_score + ' ' + 'auc:%.4f'%epoch_auc + ' ' + 'eer:%.4f'%epoch_eer + ' ' + 'TPR_2:%.4f'%TPR_2 + ' ' + 'TPR_3:%.4f'%TPR_3 + ' ' + 'TPR_4:%.4f'%TPR_4 + ' ' + '\n'
    filename.write(fusion_data_save)

def save_final_results(flops, params_count, time, best_acc, best_auc, log_dir):
    filename = open(os.path.join(log_dir, 'final_results.txt'), 'a')
    fusion_data_save = '-'*10 + '\n' + 'FLOPs: ' + str(flops) + '\n' + 'Params: ' + str(params_count) + '\n' + 'All time: ' + str(time) + '\n' + 'Best accuracy:%.4f'%best_acc + '\n' + 'Best AUC:%.4f'%best_auc + '\n' + '\n'
    filename.write(fusion_data_save)

# '''
# Created by: Zhiqing Guo
# Institutions: Xinjiang University
# Email: guozhiqing@xju.edu.cn
# Copyright (c) 2023
# '''
# import os
#
# def save_weights(w_rgb, w_res, epoch_num, log_dir):
#     # 使用 with 语句来打开文件，这样可以自动管理文件的关闭
#     with open(os.path.join(log_dir, 'Weights.txt'), 'a') as file:
#         fusion_data_save = f'Epoch_{epoch_num + 1}: {w_rgb} {w_res}\n'
#         file.write(fusion_data_save)
#
# def save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch_num, log_dir):
#     with open(os.path.join(log_dir, 'final_results.txt'), 'a') as file:
#         # 使用字符串格式化来构造保存的字符串
#         fusion_data_save = (f'Epoch_{epoch_num + 1}: '
#                             f'acc:{epoch_acc:.4f} '
#                             f'ap:{ap_score:.4f} '
#                             f'auc:{epoch_auc:.4f} '
#                             f'eer:{epoch_eer:.4f} '
#                             f'TPR_2:{TPR_2:.4f} '
#                             f'TPR_3:{TPR_3:.4f} '
#                             f'TPR_4:{TPR_4:.4f}\n')
#         file.write(fusion_data_save)
#
# def save_final_results(flops, params_count, time, best_acc, best_auc, log_dir):
#     with open(os.path.join(log_dir, 'final_results.txt'), 'a') as file:
#         # 使用字符串格式化来构造保存的字符串
#         fusion_data_save = ('-' * 10 + '\n'
#                             f'FLOPs: {flops}\n'
#                             f'Params: {params_count}\n'
#                             f'All time: {time}\n'
#                             f'Best accuracy:{best_acc:.4f}\n'
#                             f'Best AUC:{best_auc:.4f}\n\n')
#         file.write(fusion_data_save)