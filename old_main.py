#!/usr/bin/env python3
import os,sys,re,math,operator,string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from svm import SupportVectorMachine
from data_class import *
from tree_classes import *
from utils import *

def cross_validation(tr_data,tr_labels,my_data):
    dt_zo_loss = []
    bt_zo_loss = []
    rf_zo_loss = []
    svm_zo_loss = []

    for i in range(my_data.cv_split_number):

        cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels = my_data.cv_split(tr_data,tr_labels,i)

        ## add bias terms
        cv_tr_data = add_bias_term(cv_tr_data)
        cv_te_data = add_bias_term(cv_te_data)

        ## just in case, make labels {0,1} if {-1,1}
        input_vector_from_svm(cv_tr_labels)
        input_vector_from_svm(cv_te_labels)
        
        ## create&train TREE structures
        dt = DecisionTree(cv_tr_data,cv_tr_labels)
        bg = Bagging(cv_tr_data,cv_tr_labels)
        rf = RandomForest(cv_tr_data,cv_tr_labels)
        dt.train(),bg.train(),rf.train()
        
        ## dt
        preds = dt.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        dt_zo_loss += [loss]

        ## bg
        preds = bg.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        bt_zo_loss += [loss]

        ## rf
        preds = rf.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        rf_zo_loss += [loss]

        ## svm
        svm = SupportVectorMachine(len(cv_tr_data[0]),0.5,0.01)

        input_vector_to_svm(cv_tr_labels)
        input_vector_to_svm(cv_te_labels)

        svm.train(cv_tr_data,cv_tr_labels)
        preds = svm.predict(cv_te_data)

        input_vector_from_svm(cv_tr_labels)
        input_vector_from_svm(cv_te_labels)

        preds = svm.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)        
        svm_zo_loss += [loss]

    return np.array(dt_zo_loss), np.array(bt_zo_loss),\
        np.array(rf_zo_loss), np.array(svm_zo_loss)

def graded_function():
    if len(sys.argv) != 4:
        print("System Usage: python main.py <trainingDataFilename> <testingDataFilename> <modelIdx>")
        print("modelIdx\n\t1 : decision tree\n\t2 : bagging\n\t3: random forests")
        sys.exit()
    else:
        trData = Data(sys.argv[1])
        tr_data,tr_labels,_,_ = trData.split_data(1.0)
        teData = Data(sys.argv[2],features=trData.features)
        te_data,te_labels,_,_ = teData.split_data(1.0)
        if sys.argv[3] == "1":
            model = DecisionTree(tr_data,tr_labels)
            post_fix = "DT"
        elif sys.argv[3] == "2":
            model = Bagging(tr_data,tr_labels)
            post_fix = "BT"
        elif sys.argv[3] == "2":
            model = RandomForest(tr_data,tr_labels)
            post_fix = "RF"
        else:
            sys.exit("modelIdx must be in {1,2,3}")

        ## MY WORK -- ORIGINAL
        model.train()
        preds = model.predict(te_data)
        input_vector_from_svm(te_labels)
        loss = zero_one_loss(preds,te_labels)
        print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))

def short_tree_test():
    trData = Data(sys.argv[1])
    teData = Data(sys.argv[2],features=trData.features)
    tr_data,tr_labels,_,_ = trData.split_data(1.0)
    te_data,te_labels,_,_ = teData.split_data(1.0)
    tr_data = add_bias_term(tr_data)
    te_data = add_bias_term(te_data)

    a = DecisionTree(tr_data,tr_labels)
    a.train()
    preds = a.predict(te_data)
    loss = zero_one_loss(preds,te_labels)       
    print(loss)

    b = Bagging(tr_data,tr_labels)
    b.train()
    preds = b.predict(te_data)
    loss = zero_one_loss(preds,te_labels)       
    print(loss)

    c = RandomForest(tr_data,tr_labels)
    c.train()
    preds = c.predict(te_data)
    loss = zero_one_loss(preds,te_labels)       
    print(loss)

    
if __name__ == "__main__":

    if False:
        graded_function()
    else:
        
        
        cv_split_number=10
        tr_data,tr_labels = get_data(sys.argv[1],c_words=None)
        # my_data = Data(sys.argv[1],cv_split_num=cv_split_number)
        # tr_data,tr_labels,_,_ = my_data.split_data(1.0)
        # tr_data_size = len(tr_data)

        dt_losses = []
        bt_losses = []
        rf_losses = []
        svm_losses = []

        #data_balance = [0.01,0.03,0.05,0.08,0.10,0.15]
        data_balance = [0.01,0.03]
        for i in data_balance:
            tr_data,tr_labels,_,_ = my_data.split_data(i)
            dt_loss,bt_loss,rf_loss,svm_loss = cross_validation(tr_data,tr_labels,my_data)
            dt_losses += [dt_loss]
            bt_losses += [bt_loss]
            rf_losses += [rf_loss]
            svm_losses += [svm_loss]
        dt_losses = np.array(dt_losses)
        bt_losses = np.array(bt_losses)
        rf_losses = np.array(rf_losses)
        svm_losses = np.array(svm_losses)

        
        # dt_losses_ste = [i / math.sqrt(j * tr_data_size)\
        #                  for i,j in zip(np.std(dt_losses,1),data_balance)]
        # bt_losses_ste = [i / math.sqrt(j * tr_data_size)\
        #                  for i,j in zip(np.std(dt_losses,1),data_balance)]
        # rf_losses_ste = [i / math.sqrt(j * tr_data_size)\
        #                  for i,j in zip(np.std(dt_losses,1),data_balance)]
        # svm_losses_ste = [i / math.sqrt(j * tr_data_size)\
        #                   for i,j in zip(np.std(dt_losses,1),data_balance)]

        dt_losses_ste = np.std(dt_losses,1)/cv_split_number
        bt_losses_ste = np.std(bt_losses,1)/cv_split_number
        rf_losses_ste = np.std(rf_losses,1)/cv_split_number
        svm_losses_ste = np.std(svm_losses,1)/cv_split_number

        
        dt_losses_means = np.mean(dt_losses,1)
        bt_losses_means = np.mean(bt_losses,1)
        rf_losses_means = np.mean(rf_losses,1)
        svm_losses_means = np.mean(svm_losses,1)

        model_losses = {"dt": [dt_losses,dt_losses_means,dt_losses_ste],
                        "bt": [bt_losses,bt_losses_means,bt_losses_ste],
                        "rf": [rf_losses,rf_losses_means,rf_losses_ste],
                        "svm": [svm_losses,svm_losses_means,svm_losses_ste]}

        #write_losses(model_losses)

        cv_data_size = [i*tr_data_size for i in data_balance]
        dt_line = plt.errorbar(cv_data_size,dt_losses_means,\
                                dt_losses_ste,fmt="g--",label="dt")
        bt_line = plt.errorbar(cv_data_size,bt_losses_means,\
                                bt_losses_ste,fmt="r--",label="bt")
        rf_line = plt.errorbar(cv_data_size,rf_losses_means,\
                                rf_losses_ste,fmt="b--",label="nbc")
        svm_line = plt.errorbar(cv_data_size,svm_losses_means,\
                                svm_losses_ste,fmt="k--",label="nbc")

        plt.legend([dt_line,bt_line,rf_line,svm_line],["dt error","bg error","rf error","svm error"])
        #plt.axis([min(cv_data_size),max(cv_data_size),0,1])
        plt.xlabel("Cross Validation Sample Size")
        plt.ylabel("Zero-One Loss")
        plt.show()

