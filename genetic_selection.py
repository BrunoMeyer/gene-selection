import random
from deap import creator, base, tools, algorithms
import numpy as np
from sklearn.svm import SVC
from sklearn import svm as svm_mod
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from sklearn.metrics import roc_curve, auc
import math


# Define an mock that will be similar to an feature model selector
class GeneticSelection:
    estimator = None
    X = None
    y = None

    # Define the classifier that will be used to Fitness
    # The default parameter is an RandomForest
    def __init__(self, estimator):
        self.estimator = estimator

    # Select an set of features that represent the features selecteds
    # the output is an array [number of features] bool values
    def get_support(self,X,y, init_pop = 70, NGEN = 2000):
        self.X = np.array(X)
        self.y = y

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()


        # Initial gene values
        def setAttr(a,b):
            # return random.choice([1,1,1,1,0])
            # return random.choice([0,0,0,0,1])
            if(random.random() > 0.5):
                return 1
            else:
                return 0


        toolbox.register("attr_bool", setAttr, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X[0]))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        n_classes = len(set(y))

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

        def evaluate2(individual):
            X = self.X[:,np.array(individual)==1]

            # svm = svm_mod.SVR()
            # parameters = [
                # {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                # {'C': [1, 10, 100, 1000], 'gamma': 0.001, 'kernel': ['rbf']}
            # ]
            # svm = svm_mod.SVC(probability=True)
            # svm = GridSearchCV(clf, parameters)


            clf = self.estimator
            if(clf == None):
                clf = RandomForestClassifier(max_depth=40, random_state=0,n_estimators=4)
                parameters = {'max_depth': [2,40], 'n_estimators':[4,30]}
                clf = GridSearchCV(clf, parameters)

            classificadores = [
                (clf,[], "RF")
            ]
            clfs_scores = defaultdict(list)
            cv = StratifiedKFold(n_splits=5)
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

            return np.mean( np.array(clfs_scores["RF"]) ),




        toolbox.register("evaluate", evaluate2)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        population = toolbox.population(n=init_pop)

        for gen in range(NGEN):
            offspring = algorithms.varAnd(population, toolbox, cxpb=0.3, mutpb=0.05)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
        top10 = tools.selBest(population, k=10)
        selected = top10[0]
        # print("escolhido: ", evaluate2(selected))

        return np.array(selected)==1


    # Filter the numpy array data with only the features selected
    def select_genes(self,X,y, init_pop = 70, NGEN = 2000):
        features_selected = self.get_support(X,y,init_pop, NGEN)
        return np.array(X)[:, features_selected == True]
