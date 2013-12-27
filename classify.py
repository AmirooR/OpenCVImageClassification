# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
from xattr import xattr
import os

def kFoldClassify():
    (X,y) = datasets.load_svmlight_file('partial_train.txt')
    print X.shape
    print y.shape

    X_train = X.todense()
    clf = svm.LinearSVC(C=10)
    skf = StratifiedKFold(y, 3)

    for train_idx, test_idx in skf:
        X_train_k, X_test_k = X_train[train_idx], X_train[test_idx]
        y_train_k, y_test_k = y[train_idx], y[test_idx] 
        clf = svm.LinearSVC(C=100)
        clf.fit(X_train_k, y_train_k)
        z = clf.predict(X_test_k)
        print( np.sum( np.abs( z - y_test_k) / 2.0 ) / len(z) )
        idx = np.where( y_test_k > 0 )
        idx2 = np.where( y_test_k < 0)
        z_0 = z[idx2]
        z_1 = z[idx]
        print '1:  ',len( np.where( z_1 < 0)[0] ) / (1.0*len(z_1))
        print '-1: ',len( np.where( z_0 > 0)[0] ) / (1.0*len(z_0))

# <codecell>

def set_label(filename, color_name):
    colors = ['none', 'gray', 'green', 'purple', 'blue', 'yellow', 'red', 'orange']
    key = u'com.apple.FinderInfo'
    attrs = xattr(filename)
    current = attrs.copy().get(key, chr(0)*32)
    changed = current[:9] + chr(colors.index(color_name)*2) + current[10:]
    attrs.set(key, changed)
    
def classifyAll():
    (X,y) = datasets.load_svmlight_file('partial_train.txt')
    print X.shape
    print y.shape

    X_train = X.todense()
    clf = svm.LinearSVC(C=10)
    clf.fit(X_train, y)
    return clf

def colorizeAll(root_path, names_file, clf, features_file):
    fd = open(names_file)
    (X,y) = datasets.load_svmlight_file(features_file)
    X_test = X.todense()
    i = 0
    y_pred = clf.predict(X_test)
    for x in fd.readlines():
        if i % 100 == 0:
            print '[',i,']',x[:-1]
        if( y_pred[i] == 1 ):
            set_label(root_path + x[:-1], 'red')
        else:
            set_label(root_path + x[:-1], 'gray')
        i += 1
        


# <codecell>

clf = classifyAll()
colorizeAll('/Users/amirrahimi/temp/Gomrok/data/isLicense/partial_not_labeled/','partial_test.names.txt', clf, 'partial_test.txt')

# <codecell>


