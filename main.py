from __future__ import print_function
from __future__ import division

import csv
import sys
import time,datetime
import functools
from sklearn import linear_model

import random
#~ from sklearn.neighbors import KNeighborsClassifier
#~ from sklearn.model_selection import GridSearchCV
#~ from sklearn.naive_bayes import GaussianNB
#~ from sklearn.naive_bayes import MultinomialNB
#~ from sklearn.naive_bayes import BernoulliNB
#~ from sklearn.ensemble import RandomForestClassifier
#~ from sklearn.ensemble import BaggingClassifier
#~ from sklearn.preprocessing import CategoricalEncoder
from sklearn.feature_extraction import DictVectorizer

#~ from sklearn.neural_network import MLPClassifier
#~ from sklearn.neighbors import KNeighborsClassifier
#~ from sklearn import svm
#~ from sklearn.model_selection import GridSearchCV
#~ from sklearn.tree import DecisionTreeClassifier
#~ from sklearn.multiclass import OutputCodeClassifier

#~ from sklearn.svm import LinearSVC
#~ from sklearn.metrics import confusion_matrix
#~ from sklearn.ensemble import BaggingClassifier
#~ from itertools import product
#~ from sklearn.ensemble import VotingClassifier
#~ from sklearn.ensemble import RandomForestClassifier
#~ from sklearn import tree
#~ from sklearn.ensemble import GradientBoostingClassifier
#~ from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

import numpy as np

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.metrics import roc_curve, auc

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm as svm_mod
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OutputCodeClassifier

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
from sklearn.feature_extraction import FeatureHasher
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import interp
from itertools import cycle


from sklearn.metrics import roc_curve, auc

#~ from create_csv import create_csv
from sklearn import preprocessing


from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
#~ from needle import similarity as NeedleSim
import time

enc = CountVectorizer(ngram_range=(1, 10), token_pattern=r'\b\w+\b', min_df=1)

train_ratio = 0.5

N_FOLDS = 5
VIEW_SELECTED_GENES = False
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2],
                     'C': [1,10,20,40,80,100]}]


nameLabels = []

#~ def createDictFeatures(features):

import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import math

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR



from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

import genetic_selection


# Metrics used to evaluate classifiers
# this use the average of auc roc classes
def metrics(testX,testY,n_classes,classificadores):
    X_test = testX


    ########################################################################



    roc_auc_classifiers = defaultdict(list)

    for iClassPlot in range(n_classes):
        for ic, clf in enumerate(classificadores):
            y_score = np.zeros(len(testX)*n_classes).reshape((len(testX),n_classes))

            for i,x in enumerate(testX):
                y_score[i][  classificadores[ic][1][i] ] = 1
            y_test = y_score.copy()

            for i,y in enumerate(testY):
                y_test[i] = np.zeros(n_classes)
                y_test[i][int(y) ] = 1
            y_score = classificadores[ic][0].predict_proba(X_test)
            # print(y_test,y_score)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

                roc_auc[i] = auc(fpr[i], tpr[i])
                if(not math.isnan(roc_auc[i])):
                    roc_auc_classifiers[classificadores[ic][2]].append(roc_auc[i])

    ########################################################################
    for clf in roc_auc_classifiers:
        auc_score = np.mean( np.array(roc_auc_classifiers[clf]) )
        roc_auc_classifiers[clf] = auc_score
    return roc_auc_classifiers





if(__name__ == "__main__"):


    # Load the data, normalize and change the labels to integer
    X,y = [[],[]]
    file = open(sys.argv[1])
    file.readline()
    for l in file:
        l = l.replace("\n","").split(" ")
        label = l[0]
        del l[0]

        X.append( np.array([float(x) for x in l if x!=""]) )
        y.append(label)

    X = np.array(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)
    # X = X[:,:160]
    y = np.array(y)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    n_classes = len(set(y))







    # mlp = BaggingClassifier(MLPClassifier(
        # solver='lbfgs',
        # alpha=1e-4,
        # activation=('logistic'),
        # hidden_layer_sizes=(5),
        # learning_rate_init=0.001,
        # max_iter=1000, random_state=1
    # ))

    # Define the classifiers and parameters that will be used
    mlp = MLPClassifier(
        solver='lbfgs',
        alpha=1e-4,
        activation=('logistic'),
        hidden_layer_sizes=(5),
        learning_rate_init=0.0001,
        max_iter=100000, random_state=1
    )

    parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': 0.001, 'kernel': ['rbf']}
    ]
    svm = svm_mod.SVC(probability=True)
    #~ svm = GridSearchCV(svr, parameters)


    gnb = GaussianNB()

    knn = KNeighborsClassifier(n_neighbors=1)

    bnb = BernoulliNB()

    mnb = MultinomialNB()


    clf = RandomForestClassifier(max_depth=40, random_state=0,n_estimators=4)
    parameters = {'max_depth': [2,40], 'n_estimators':[4,30]}
    rdf = GridSearchCV(clf, parameters)



    votacaoProduto = VotingClassifier(
    estimators=[
        # ('mlp',mlp),
        ('rdf', rdf),
        # ('svm', svm),
        # ('gnb', gnb),
        ('knn', knn),
        # ('bnb', bnb),
        # ('mnb', mnb)
    ]

    #~ , voting='hard', weights=[1,2,1,1,5,1,1])
    #~ , voting='hard', weights=[1,1])
    , voting='soft', weights=[1,1])

    classificadores = [
        (mlp,[], "MLP"),
        (rdf,[], "RDF"),
        (svm,[], "SVM"),
        (knn,[], "KNN"),
        (gnb,[], "GNB"),
        (bnb,[], "BNB"),
        (mnb,[], "MNB"),
        (votacaoProduto,[],"Votacao Produto"),
    ]






    media=0.0
    total = 0
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    n_samples, n_features = X.shape
    y = np.array(y)
    i = 0

    clfs_scores = defaultdict(list)
    clfs_fs_scores = defaultdict(lambda: defaultdict(list))


    # Define the features selectors that will be used
    estimator = SVR(kernel="linear")
    selector_svm_rfe = RFE(estimator, 150, step=1)
    estimator = svm_mod.SVC(probability=True)
    genetic_svm = genetic_selection.GeneticSelection(estimator=estimator)

    estimator = ExtraTreesClassifier()
    selector_rdf_rfe = RFE(estimator, 150, step=1)
    genetic_rdf = genetic_selection.GeneticSelection(estimator=estimator)

    estimator = mnb
    selector_mnb_rfe = RFE(estimator, 150, step=1)
    genetic_mnb = genetic_selection.GeneticSelection(estimator=estimator)

    # #
    selector_variance = VarianceThreshold()

    clf_etc = ExtraTreesClassifier()
    clf_etc = clf_etc.fit(X, y)
    selector_rf = SelectFromModel(clf_etc, prefit=True)


    # if the user only want to see the most commons features selected
    if(VIEW_SELECTED_GENES):
        selector_rdf_rfe.fit(X,y)
        selector_svm_rfe.fit(X,y)
        selector_mnb_rfe.fit(X,y)
        selector_variance.fit(X,y)

        supports = [
            selector_svm_rfe.get_support(),
            selector_rdf_rfe.get_support(),
            selector_mnb_rfe.get_support(),
            genetic_svm.get_support(X,y),
            genetic_rdf.get_support(X,y),
            genetic_mnb.get_support(X,y),
            selector_variance.get_support(),
            selector_rf.get_support(indices=False)
        ]
        intersect_support = supports[0]
        for s in supports:
            intersect_support = np.logical_and(s,intersect_support)

        print(intersect_support)
        print(sum(intersect_support))
        exit(0)



    # Measure the time of each feature selector
    feature_selectors = {}

    start = time.clock() # CPU TIME
    feature_selectors["GEN-SVM"] = genetic_svm.select_genes(X,y)
    print("GEN SVM CPU TIME: ",(time.clock() - start))

    start = time.clock() # CPU TIME
    feature_selectors["GEN-RDF"] = genetic_rdf.select_genes(X,y)
    print("GEN RDF CPU TIME: ",(time.clock() - start))

    start = time.clock() # CPU TIME
    feature_selectors["GEN-MNB"] = genetic_mnb.select_genes(X,y)
    print("GEN MNB CPU TIME: ",(time.clock() - start))


    start = time.clock() # CPU TIME
    feature_selectors["RFE-SVM"] = selector_svm_rfe.fit_transform(X,y)
    print("RFE SVM CPU TIME: ",(time.clock() - start))

    # start = time.clock() # CPU TIME
    # feature_selectors["RFE-RDF"] = selector_rdf_rfe.fit_transform(X,y)
    # print("RFE RDF CPU TIME: ",(time.clock() - start))

    start = time.clock() # CPU TIME
    feature_selectors["RFE-MNB"] = selector_mnb_rfe.fit_transform(X,y)
    print("RFE MNB CPU TIME: ",(time.clock() - start))

    start = time.clock() # CPU TIME
    feature_selectors["RF"]  = selector_rf.transform(X)
    print("RF CPU TIME: ",(time.clock() - start))

    start = time.clock() # CPU TIME
    feature_selectors["VAR"] = selector_variance.fit_transform(X,y)
    print("VAR CPU TIME: ",(time.clock() - start))










    # For each combination of classifier and selector, evaluate an metric
    # that define the average of auc roc classes
    # This will be executed with cross validation and the average of each fold
    # will be printed for each combination
    cv = StratifiedKFold(n_splits=N_FOLDS)
    for train, test in cv.split(X, y):

        testX = X[test]
        testY = y[test]

        for i,c in enumerate(classificadores):
            classificadores[i][1].clear()
            classificadores[i][0].fit(X[train],y[train])

        for x in testX:
            for i,c in enumerate(classificadores):
                if(classificadores[i][2] != "SVM"):
                    classificadores[i][1].append(classificadores[i][0].predict(np.array(x).reshape(1, -1) ))
                else:
                    classificadores[i][1].append(svm.predict( np.array(x).reshape(1, -1) )[0])

        scores = metrics(testX,testY,n_classes,classificadores)
        for s in scores:
            clfs_scores[s].append(scores[s])

        ##

        for fs_name in feature_selectors:
            X_fs = feature_selectors[fs_name]
            testX_fs = X_fs[test]

            for i,c in enumerate(classificadores):
                classificadores[i][1].clear()
                classificadores[i][0].fit(X_fs[train],y[train])

            for x in testX_fs:
                for i,c in enumerate(classificadores):
                    if(classificadores[i][2] != "SVM"):
                        classificadores[i][1].append(classificadores[i][0].predict(np.array(x).reshape(1, -1) ))
                    else:
                        classificadores[i][1].append(svm.predict( np.array(x).reshape(1, -1) )[0])

            scores = metrics(testX_fs,testY,n_classes,classificadores)
            for s in scores:
                clfs_fs_scores[fs_name][s].append(scores[s])

    for clf in clfs_scores:
        print("Normal", clf, np.mean( np.array(clfs_scores[clf]) ))
        for fs_name in feature_selectors:
            print(fs_name, clf, np.mean( np.array(clfs_fs_scores[fs_name][clf]) ))
