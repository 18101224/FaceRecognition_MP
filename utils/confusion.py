import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion(y_hats, ys,model_name,num_classes=7):
    conf_matrix = confusion_matrix(ys,y_hats,labels=np.arange(num_classes))
    dia = np.diag(conf_matrix)
    rows = (conf_matrix.sum(axis=1,keepdims=False) - dia).reshape(-1,1)
    cols = (conf_matrix.sum(axis=0,keepdims=False) - dia).reshape(1,-1)
    aug = np.zeros((conf_matrix.shape[0]+1,conf_matrix.shape[0]+1))
    '''aug[:-1,:-1] = conf_matrix
    aug[-1,:-1] = cols
    aug[:-1,-1] = rows'''

    fig,ax = plt.subplots(figsize=(8,7))
    sns.heatmap(
        conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, square=True,
        annot_kws={'size':14}, linewidths=0.5, linecolor='gray', ax=ax
    )
    normed_r = rows/rows.sum(keepdims=False)
    normed_c = cols/cols.sum(keepdims=False)
    cax_row = fig.add_axes([0.82, 0.11, 0.04, 0.75])  # [x, y, width, height]
    sns.heatmap(rows, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=False, yticklabels=False, ax=cax_row)

    # --- Add Column Sum Heatmap (Bottom Side) ---
    cax_col = fig.add_axes([0.11, 0.05, 0.7, 0.03])  # [x, y, width, height]
    sns.heatmap(cols, annot=True, fmt="d", cmap="Oranges", cbar=False,
                xticklabels=False, yticklabels=False, ax=cax_col)
    '''for i,v in enumerate(rows.ravel()):
        ax.text(conf_matrix.shape[1],i+0.5, f'{v}',ha='center', va='center', fontsize=10,
            bbox=dict(facecolor='lightblue', edgecolor='gray'),alpha=(1-normed_r[i]))
    for j,v in enumerate(cols.ravel()):
        ax.text(j+0.5, conf_matrix.shape[1], f'{v}', ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='lightblue', edgecolor='gray'),alpha=(1-normed_c[j]))'''
    ax.set_xlabel(r'$\hat{y}$')
    ax.set_ylabel('$y$')
    ax.set_title(f'{model_name}')
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    plt.savefig(f'{model_name}confusion.png')