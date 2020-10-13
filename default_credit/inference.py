import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def read_file(name):
    df = pd.read_csv(name, header=None)
    df.pop(1)

    data = df.to_numpy()

    w = data[:-1]
    b = data[-1]
    return w, b


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def forward(w, b, x):
    out = np.dot(x, w) + b
    return sigmoid(out)

if __name__ == '__main__':
    xdf = pd.read_csv('default_credit_tfe_host.csv', header=None)
    xdf.insert(len(xdf.columns), len(xdf.columns), 0.)
    ydf = pd.read_csv('default_credit_tfe_guest.csv', header=None)
    ydf.pop(1)

    x = xdf.to_numpy()
    y = ydf.to_numpy()
    print(y.shape)

    def do_one(index):
        w, b = read_file(f'epoch_{index}')
        #print(w, b)

        y_p = forward(w, b, x)
        #print(y, np.around(y_p))

        fpr, tpr, thresholds = roc_curve(y, y_p)
        ks = max(tpr - fpr)
        print("{} auc:{}, ks:{}".format(index, roc_auc_score(y, y_p), ks))


    if False:
        for i in range(10):
            do_one(i+1)

    do_one(1)
    do_one(2)
    do_one(4)
    do_one(8)
    do_one(16)

    #do_one(16)
    #do_one(32)
    #do_one(50)
    #do_one(100)


