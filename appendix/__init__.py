"""
Analysis tools package for model evaluation and visualization.

This package will contain various analysis tools for:
- Model performance metrics
- Visualization utilities
- Statistical analysis
- Comparison tools
"""

from argparse import ArgumentParser
import os 
from .computation_utils import get_predictions, calculate_accuracy, get_angle_matrix, make_confusion_matrix
from .plot_util import plot_accuracy_histogram, plot_confusion_matrix, plot_angle_with_confusion, plot_micro_accuracy_histogram
from .loads import load_model, load_dataset
from torch.utils.data import DataLoader
import torch 
import numpy as np
# Version
__version__ = '0.1.0' 


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--trained_ckpt')
    parser.add_argument('--pretrained_ckpt', default=False)
    parser.add_argument('--dataset_name', choices=['cifar10', 'cifar100', 'imagenet', 
                                                   'svhn', 'AffectNet7', 'RAF-DB', 'inatural2019',
                                                     'inatural2018'])
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--store_train', default=False)
    parser.add_argument('--mode',choices=['analysis','comparision'])
    parser.add_argument('--compare_ckpt')
    args = parser.parse_args()
    args.save_dir = os.path.join('results',args.save_dir)
    return args 

def anal_model(args):
    device = torch.device('cuda')
    model1, model2 = load_model(args,args.trained_ckpt), load_model(args, args.compare_ckpt)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir,exist_ok=True)
    train_set, valid_set = load_dataset(args)
    criterion = torch.nn.CrossEntropyLoss()
    train_preds1, train_losses1, train_labels1 = get_predictions(model1, train_set, criterion, device)
    train_preds2, train_losses2, train_labels2 = get_predictions(model2, train_set, criterion, device)
    valid_preds1, valid_losses1, valid_labels1 = get_predictions(model1, valid_set, criterion, device)
    valid_preds2, valid_losses2, valid_labels2 = get_predictions(model2, valid_set, criterion, device)
    train_micro_acc1 = calculate_accuracy(train_preds1, train_labels1, metric='micro')
    train_micro_acc2 = calculate_accuracy(train_preds2, train_labels2, metric='micro')
    valid_micro_acc1 = calculate_accuracy(valid_preds1, valid_labels1, metric='micro')
    valid_micro_acc2 = calculate_accuracy(valid_preds2, valid_labels2, metric='micro')
    print(f'Train Accuracy 1: {train_micro_acc1:.4f}, Train Accuracy 2: {train_micro_acc2:.4f}')
    print(f'Valid Accuracy 1: {valid_micro_acc1:.4f}, Valid Accuracy 2: {valid_micro_acc2:.4f}')
    train_macro_acc1 = calculate_accuracy(train_preds1, train_labels1, metric='macro')
    train_macro_acc2 = calculate_accuracy(train_preds2, train_labels2, metric='macro')
    valid_macro_acc1 = calculate_accuracy(valid_preds1, valid_labels1, metric='macro')
    valid_macro_acc2 = calculate_accuracy(valid_preds2, valid_labels2, metric='macro')
    print(f'Train Accuracy 1: {np.mean(train_macro_acc1):.4f}, Train Accuracy 2: {np.mean(train_macro_acc2):.4f}')
    print(f'Valid Accuracy 1: {np.mean(valid_macro_acc1):.4f}, Valid Accuracy 2: {np.mean(valid_macro_acc2):.4f}')
    anglemat_1 = get_angle_matrix(model1.get_kernel()).detach().cpu().numpy()
    anglemat_2 = get_angle_matrix(model2.get_kernel()).detach().cpu().numpy()
    plot_micro_accuracy_histogram(train_micro_acc1, train_micro_acc2, labels=('Model 1', 'Model 2'), title='Train Micro Accuracy', save_path=os.path.join(args.save_dir, 'train_micro_accuracy.png'))
    plot_micro_accuracy_histogram(valid_micro_acc1, valid_micro_acc2, labels=('Model 1', 'Model 2'), title='Valid Micro Accuracy', save_path=os.path.join(args.save_dir, 'valid_micro_accuracy.png'))
    plot_accuracy_histogram(accuracies1=train_macro_acc1, accuracies2=train_macro_acc2, labels=('Model 1', 'Model 2'), title='Train Macro Accuracy', save_path=os.path.join(args.save_dir, 'train_macro_accuracy.png'))
    plot_accuracy_histogram(accuracies1=valid_macro_acc1, accuracies2=valid_macro_acc2, labels=('Model 1', 'Model 2'), title='Valid Macro Accuracy', save_path=os.path.join(args.save_dir, 'valid_macro_accuracy.png'))
    plot_confusion_matrix(confusion_matrix=make_confusion_matrix(train_preds1, train_labels1), title='Train Confusion Matrix', save_path=os.path.join(args.save_dir, 'train_confusion_matrix.png'))
    plot_confusion_matrix(confusion_matrix=make_confusion_matrix(valid_preds1, valid_labels1), title='Valid Confusion Matrix', save_path=os.path.join(args.save_dir, 'valid_confusion_matrix.png'))
    plot_angle_with_confusion(angle_matrix=anglemat_1, preds=valid_preds1, labels=valid_labels1, save_path=os.path.join(args.save_dir, 'model1_angle_with_confusion.png'))
    plot_angle_with_confusion(angle_matrix=anglemat_2, preds=valid_preds2, labels=valid_labels2, save_path=os.path.join(args.save_dir, 'model2_angle_with_confusion.png'))