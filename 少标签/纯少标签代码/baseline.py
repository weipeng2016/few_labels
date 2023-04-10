import pandas as pd
import sys
import numpy as np
import time

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import Normalizer,StandardScaler

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM, SVC
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.rod import ROD
from pyod.models.inne import INNE
from pyod.models.lscp import LSCP
from pyod.models.loda import LODA
from pyod.models.suod import SUOD
from pyod.models.vae import VAE
from pyod.models.so_gaal import SO_GAAL
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.anogan import AnoGAN
from pyod.models.alad import ALAD
from pyod.models.rgraph import RGraph
from pyod.models.lunar import LUNAR

from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier

from sklearn.neural_network import MLPClassifier
MLPClassifier3 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3,2), random_state=1)
MLPClassifier4 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(4,2), random_state=1)
MLPClassifier5 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)

from sklearn import svm
SVM = svm.SVC()

res = [str(i) for i in range(9)]
tmp = [i for i in range(9)]
for i in tmp:
    tmp_res = [str(i)]
    for j in tmp:
        if i == j:
            continue
        tmp_res.append(str(j))
        res.append("+".join(tmp_res[:]))
print(len(res), res)
#res = [i for i in range(9)]
Note=open(r'D:\shaobiaoqiian\pythonProject3\try_again\dataset\new\debug1\res\res_test_for_all.txt',mode='w')
#Note.writelines("wcnm" + "\n") #\n 换行符
#Note.writelines("cnm" + "\n") #\n 换行符
uns_models, s_models, semi_models = {}, {}, {}
#无监督算法
iforest = IsolationForest()
oneclasssvm = OneClassSVM()
ellipticenvelope = EllipticEnvelope()
localoutlierfactor = LocalOutlierFactor(novelty=True)
#uns_models = {"iforest":iforest, "oneclasssvm":oneclasssvm, "ell":ellipticenvelope, "loc":localoutlierfactor,"ABOD":ABOD(), "ECOD":ECOD(), "COPOD":COPOD()}
uns_models = {"iforest":iforest, "oneclasssvm":oneclasssvm, "loc":localoutlierfactor,"ABOD":ABOD()}

#有监督算法
rf = RandomForestClassifier()
s_models = {"rf":rf, "MLP3":MLPClassifier3,"MLP4":MLPClassifier4,"MLP5":MLPClassifier5, "SVM":SVM}

#半监督算法
svc = SVC(probability=True, gamma="auto")
LabelPropagation, LabelSpreading, SelfTrainingClassifier = LabelPropagation(), LabelSpreading(), SelfTrainingClassifier(svc)
semi_models = {"LabelPropagation":LabelPropagation, "LabelSpreading":LabelSpreading, "SelfTrainingClassifier":SelfTrainingClassifier}
all_models = [uns_models, s_models, semi_models]
n = 0
for models in all_models:
    #if n <= 1: n+=1; continue
    for model in models:
        Note.writelines(model+"\n")
        print(n, model)
        for i in res:
            print(i)
            #读取训练集，测试集，并把有标签和无标签分出来
            Note.writelines(str(i)+"\n")
            train_data = pd.read_csv(
                r"D:\shaobiaoqiian\pythonProject3\try_again\dataset\new\debug1\%s_0.02_50.csv" % str(i))
            #test_data = pd.read_csv(r"D:\shaobiaoqiian\pythonProject3\try_again\dataset\new\debug\%s_test.csv" % str(i))
            test_data = pd.read_csv(r"D:\shaobiaoqiian\pythonProject3\try_again\dataset\new\debug\test_for_all.csv")

            col_names = list(train_data)
            train_data, test_data = train_data.values, test_data.values
            scaler = StandardScaler().fit(train_data[:, :-1])
            X1 = scaler.transform(train_data[:, :-1])
            T1 = scaler.transform(test_data[:, :-1])
            scaler = Normalizer().fit(X1)
            train_data[:, :-1] = scaler.transform(X1)
            test_data[:, :-1] = scaler.transform(T1)
            train_data, test_data = pd.DataFrame(train_data, columns=col_names), pd.DataFrame(test_data,
                                                                                              columns=col_names)

            labeled = train_data[(train_data["labels"] == 1)]
            unlabeled = train_data[(train_data["labels"] == 0)]
            #三大类方法开始测试
            clf = models[model]
            train_start = time.time()
            if n == 0:#无监督
                #print()
                clf.fit(train_data.values[:, :-1])
            elif n == 1:#有监督
                #print()
                clf.fit(pd.concat([labeled, unlabeled]).values[:, :-1], pd.concat([labeled, unlabeled]).values[:, -1])
            else:#半监督
                #print()
                clf.fit(train_data.values[:, :-1], train_data.values[:, -1])
            train_end = time.time()
            train_time = train_end - train_start
            Note.writelines("train_time" + str(train_time) + "\n")
            test_start = time.time()
            pred = clf.predict(test_data.values[:, :-1])
            test_end = time.time()
            test_time = test_end - test_start
            Note.writelines("test_time" + str(test_time) + "\n")
            roc = roc_auc_score(test_data.values[:, -1], pred)
            pr = average_precision_score(test_data.values[:, -1], pred)
            Note.writelines("AUC_ROC" + str(roc) + "\n")
            Note.writelines("AUC_PR" + str(pr) + "\n")
    n += 1