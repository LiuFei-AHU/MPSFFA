
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# calculate metrics
def metric(label,pred):
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(label)):
        if label[i] == pred[i] and label[i] == 1:
            tp = tp + 1
        if label[i] == pred[i] and label[i] == 0:
            tn = tn + 1
        if label[i] == 1 and pred[i] == 0:
            fn = fn + 1
        if label[i] == 0 and pred[i] == 1:
            fp = fp + 1
    
    acc = (tp+tn)/(tp+fp+fn+tn)
    spe = tn/(tn+fp)
    sen = tp/(tp+fn)

    return acc,spe,sen


# one group
def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.show()


# multi groups
def roc(y_label, y_predict,aucs=None, num_class=None, model=None, save_path='rocs.png'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(y_label)):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_label[i]).ravel(), np.array(y_predict[i]).ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])

    fig = plt.figure("beauty")
    lw = 2

    colors = ['aqua', 'darkorange', 'cornflowerblue', 'navy', 'blue', 'red', 'green']
    for i in range(len(y_label)):
        # plt.plot(fpr[i], tpr[i], color=colors[i],linestyle='', linewidth=1, label=model[i]+'(area={:0.2f})'
        # .format(roc_auc[i]))
        plt.plot(fpr[i], tpr[i], color=colors[i], linewidth=1, label=model[i] + '(area={:0.3f})'
                 .format(aucs[i]))  # roc_auc[i]

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])  # -0.05 1.0
    plt.ylim([0.0, 1.0])  # 1.05 1.0
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.savefig(save_path, dpi=120)
    plt.show()
