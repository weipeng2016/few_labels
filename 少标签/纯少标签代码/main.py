import os
print(os.path)

import sys
sys.path.append(r"D:\shaobiaoqiian\pythonProject4")

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
tf.device("/gpu:0")

from DPLAN import DPLAN
from ADEnv import ADEnv
from utils import writeResults
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import Normalizer,StandardScaler
#import pudb;pu.db

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score

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
res = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
#sys.exit(0)

### Basic Settings
# data path settings
data_path="../pythonProject3/try_again/dataset/new"
data_folders=["debug"]
# data_subsets={"NB15_unknown1":["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance"]}
data_subsets={"debug": res}#, "1", "2", "3", "4", "5", "6", "7", "8"]}
testdata_subset="test_for_all.csv"#"0_0.02_50.csv"#"test_for_all.csv" # test data is the same for subsets of the same class
# scenario settings
num_knowns=50
contamination_rate=0.02
# experiment settings
runs=1
model_path="./model"
result_path="./results"
result_file="results.csv"
Train=True
Test=True

### Anomaly Detection Environment Settings
size_sampling_Du=32#1000
prob_au=0.5
label_normal=-1
label_anomaly=1

### DPLAN Settings
settings={}
settings["hidden_layer"]=20 # l
settings["memory_size"]=1000 # M
settings["warmup_steps"]=1000#10000
settings["episodes"]=10#10
settings["steps_per_episode"]=200#2000
settings["epsilon_max"]=1
settings["epsilon_min"]=0.1
settings["epsilon_course"]=1000#10000
settings["minibatch_size"]=32#32
settings["discount_factor"]=0.99 # gamma
settings["learning_rate"]=0.00025
settings["minsquared_gradient"]=0.01
settings["gradient_momentum"]=0.95
settings["penulti_update"]=200#2000 # N
settings["target_update"]=1000#10000 # K

# different datasets
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)
names=[]
rocs=[]
prs=[]
train_times=[]
test_times=[]
for item in res:
    data_subsets = {"debug": [item]}
    #testdata_subset = "%s_test.csv" % str(woc)
    for data_f in data_folders:
        # different unknown datasets for each dataset
        print("read test")
        subsets=data_subsets[data_f]
        testdata_path=os.path.join(data_path,data_f,testdata_subset)
        test_table=pd.read_csv(testdata_path)
        test_table.dropna(inplace=True)
        test_dataset=test_table.values
        print("done")

        m_test, n_test = test_dataset.shape
        n_feature_test = n_test - 1

        for subset in subsets:
            np.random.seed(42)
            tf.random.set_seed(42)
            # location of unknwon datasets
            data_name="{}_{}_{}".format(subset,contamination_rate,num_knowns)
            unknown_dataname=data_name+".csv"
            undata_path=os.path.join(data_path,data_f,unknown_dataname)
            # get unknown dataset
            print("subset", subset)
            #print("")
            table=pd.read_csv(undata_path)
            table.dropna(inplace=True)
            undataset=table.values
            print("done")
            #print(np.isnan(undataset).any())
            #np.dropna(undataset, inplace=True)
            #print(np.isnan(undataset).any())
            #print(np.isinf(undataset).any())
            #print(np.isfinite(undataset).all())


            m_train, n_train = undataset.shape
            n_feature_train = n_train - 1
            scaler = StandardScaler().fit(undataset[:, :n_feature_train])
            X1 = scaler.transform(undataset[:, :n_feature_train])
            T1 = scaler.transform(test_dataset[:, :n_feature_test])
            scaler = Normalizer().fit(X1)
            undataset[:, :n_feature_train] = scaler.transform(X1)
            test_dataset[:, :n_feature_test] = scaler.transform(T1)


            print()
            #rocs=[]
            #prs=[]
            #train_times=[]
            #test_times=[]
            # run experiment
            for i in range(runs):
                print("#######################################################################")
                print("Dataset: {}".format(subset))
                print("Run: {}".format(i))

                weights_file=os.path.join(model_path,"{}_{}_{}_weights.h4f".format(subset,i,data_name))
                # initialize environment and agent
                tf.compat.v1.reset_default_graph()
                env=ADEnv(dataset=undataset,
                          sampling_Du=size_sampling_Du,
                          prob_au=prob_au,
                          label_normal=label_normal,
                          label_anomaly=label_anomaly,
                          name=data_name)
                model=DPLAN(env=env,
                            settings=settings)

                # train the agent
                train_time=0
                if Train:
                    # train DPLAN
                    train_start=time.time()
                    model.fit(weights_file=weights_file)
                    #model.fit()
                    train_end=time.time()
                    train_time=train_end-train_start
                    print("Train time: {}/s".format(train_time))

                # test the agent
                test_time=0
                if Test:
                    test_X, test_y=test_dataset[:,:-1], test_dataset[:,-1]
                    model.load_weights(weights_file)
                    # test DPLAN
                    test_start=time.time()
                    pred_y=model.predict(test_X)
                    pred_y_label = model.predict_label(test_X)
                    test_end=time.time()
                    test_time=test_end-test_start
                    print("Test time: {}/s".format(test_time))

                    #print(pred_y)
                    roc = roc_auc_score(test_y, pred_y)
                    pr = average_precision_score(test_y, pred_y)

                    acc = accuracy_score(test_y, pred_y_label)
                    print("acc", acc)

                    pre = precision_score(test_y, pred_y_label)
                    print("pre", pre)

                    rec = recall_score(test_y, pred_y_label)
                    print("rec", rec)

                    f1 = f1_score(test_y, pred_y_label)
                    print("f1", f1)
                    print("{} Run {}: AUC-ROC: {:.4f}, AUC-PR: {:.4f}, train_time: {:.2f}, test_time: {:.2f}".format(subset,
                                                                                                                     i,
                                                                                                                     roc,
                                                                                                                     pr,
                                                                                                                     train_time,
                                                                                                                     test_time))

                    rocs.append(roc)
                    prs.append(pr)
                    train_times.append(train_time)
                    test_times.append(test_time)
                    names.append(data_name)

            if Test:
                # write results
                writeResults(subset, rocs, prs, train_times, test_times, os.path.join(result_path,result_file))

    #print(rocs)
    #print(prs)
    #print(train_times)
    #print(test_times)
wp_res = {}
print(rocs)
wp_res["name"] = names
wp_res["roc"] = rocs
wp_res["pr"] = prs
wp_res["train_time"] = train_times
wp_res["test_time"] = test_times
wp_res = pd.DataFrame(wp_res)
wp_res.to_csv(r"D:\shaobiaoqiian\pythonProject3\try_again\dataset\new\debug\res\DPLAN_u_test_for_all.csv")