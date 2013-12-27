# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold

(X,y) = datasets.load_svmlight_file('partial_train.txt')
print X.shape
print y.shape

X_train = X.todense()
clf = svm.LinearSVC(C=100)
skf = StratifiedKFold(y, 5)

for train_idx, test_idx in skf:
    X_train_k, X_test_k = X_train[train_idx], X_train[test_idx]
    y_train_k, y_test_k = y[train_idx], y[test_idx] 
    clf = svm.LinearSVC(C=100)
    clf.fit(X_train_k, y_train_k)
    z = clf.predict(X_test_k)
    print( np.sum( np.abs( z - y_test_k) / 2.0 ) / len(z) )
    idx = np.where( y_test_k > 0 )
    z_1 = z[idx]
    print len( np.where( z_1 < 0)[0] ) / (1.0*len(z_1))

# <codecell>


