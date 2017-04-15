#!/usr/bin/env python3
import os,sys,re,math,operator,string
import numpy as np
import matplotlib.pyplot as plt
from svm import SupportVectorMachine
from data_class import *
from tree_classes import *
from utils import *

def svm(features, labels):
    # test sub-gradient SVM
    total = features.shape[1]
    lam = 1.; D = total
    x = features; y = (labels-0.5)*2
    w = np.zeros(D); wpr = np.ones(D)
    eta = 0.5; lam = 0.01; i = 0; MAXI = 100; tol = 1e-6
    while True:
        if np.linalg.norm(w-wpr) < tol or i > MAXI:
            break
        f = w @ x.T
        pL = np.where(np.multiply(y,f) < 1, -x.T @ np.diag(y), 0)
        pL = np.mean(pL,axis=1) + lam*w
        wpr = w
        w = w - eta*pL
        i += 1

    return w

def svm_pred(w, features):
    return np.where((features @ w) >= 0, 1, 0)

def k_fold_cross_validation(tr_data,tr_labels,split_number,tree_depth=10,tree_count=50,only_ensemble=False,use_svm=True):

    dt_zo_loss = []
    bt_zo_loss = []
    rf_zo_loss = []
    bdt_zo_loss = []
    svm_zo_loss = []

    for i in range(split_number):

        cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels\
            = split_data(tr_data,tr_labels,split_number,i)

        ## dt
        if not only_ensemble:
            dt = DecisionTree(cv_tr_data,cv_tr_labels)
            dt.train()
            preds = dt.predict(cv_te_data)
            loss = zero_one_loss(preds,cv_te_labels)
            dt_zo_loss += [loss]
        
        ## bg
        bg = Bagging(cv_tr_data,cv_tr_labels)
        bg.train()
        preds = bg.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        bt_zo_loss += [loss]

        ## rf
        rf = RandomForest(cv_tr_data,cv_tr_labels)
        rf.train()
        preds = rf.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        rf_zo_loss += [loss]

        ## adamax
        bdt = AdaMaxDT(cv_tr_data,cv_tr_labels)
        bdt.train()
        preds = bdt.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        bdt_zo_loss += [loss]

        ## svm
        if not only_ensemble and use_svm:

            # add bias term for only svm
            cv_tr_data = add_bias_term(cv_tr_data)
            cv_te_data = add_bias_term(cv_te_data)

            svm = SupportVectorMachine(len(cv_tr_data[0]),0.5,0.01)

            svm.train(cv_tr_data,cv_tr_labels)
            preds = svm.predict(cv_te_data)

            input_vector_from_svm(cv_te_labels)

            preds = svm.predict(cv_te_data)
            loss = zero_one_loss(preds,cv_te_labels)        
            svm_zo_loss += [loss]
            input_vector_to_svm(cv_te_labels)

    return np.array(dt_zo_loss), np.array(bt_zo_loss),\
        np.array(rf_zo_loss), np.array(bdt_zo_loss), np.array(svm_zo_loss)

def k_fold_cross_validation_edit(tr_data,tr_labels,split_number,tree_depth=10,tree_count=50,only_ensemble=False,use_svm=True):

    dt_zo_loss = []
    bt_zo_loss = []
    rf_zo_loss = []
    bdt_zo_loss = []
    svm_zo_loss = []

    for i in range(split_number):

        cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels\
            = split_data(tr_data,tr_labels,split_number,i)
        
        ## dt
        if not only_ensemble:
            dt = DecisionTree(cv_tr_data,cv_tr_labels)
            dt.train()
            preds = dt.predict(cv_te_data)
            loss = zero_one_loss(preds,cv_te_labels)
            dt_zo_loss += [loss]
            #print(loss)

        #bg
        # bg = Bagging(cv_tr_data,cv_tr_labels)
        # bg.train()
        # preds = bg.predict(cv_te_data)
        # loss = zero_one_loss(preds,cv_te_labels)
        # bt_zo_loss += [loss]

        #rf
        # rf = RandomForest(cv_tr_data,cv_tr_labels)
        # rf.train()
        # preds = rf.predict(cv_te_data)
        # loss = zero_one_loss(preds,cv_te_labels)
        # rf_zo_loss += [loss]


        ## adamax
        bdt = AdaMaxDT(cv_tr_data,cv_tr_labels)
        bdt.train()
        preds = bdt.predict(cv_te_data)
        loss = zero_one_loss(preds,cv_te_labels)
        bdt_zo_loss += [loss]

        if not only_ensemble and use_svm:
            ## svm
            #svm = SupportVectorMachine(len(cv_tr_data[0]),0.5,0.01)

            # svm.train(cv_tr_data,cv_tr_labels)
            # preds = svm.predict(cv_te_data)

            w = svm(cv_tr_data, cv_tr_labels)
            preds = svm_pred(w, cv_te_data)

            #preds = svm.predict(cv_te_data)

            loss = zero_one_loss(preds,cv_te_labels)        
            svm_zo_loss += [loss]

    return np.array(dt_zo_loss), np.array(bt_zo_loss),\
        np.array(rf_zo_loss), np.array(bdt_zo_loss), np.array(svm_zo_loss)

def single_model_function():
    if len(sys.argv) != 4:
        print("System Usage: python main.py <trainingDataFilename> <testingDataFilename> <modelIdx>")
        print("modelIdx\n   1 : decision tree\n   2 : bagging\n   3 : random forests")
        sys.exit(1)
    else:
        tr_data,tr_labels,cw = get_data(sys.argv[1],c_words=None,total_f=1000)
        te_data,te_labels,_ = get_data(sys.argv[2],c_words=cw,total_f=1000)

        input_vector_to_svm(tr_labels)
        input_vector_to_svm(te_labels)

        print(len(tr_data))
        print(len(te_data))
        if sys.argv[3] == "1":
            model = DecisionTree(tr_data,tr_labels)
            post_fix = "DT"
        elif sys.argv[3] == "2":
            model = Bagging(tr_data,tr_labels)
            post_fix = "BT"
        elif sys.argv[3] == "3":
            model = RandomForest(tr_data,tr_labels)
            post_fix = "RF"
        elif sys.argv[3] == "4":## TODO: KENT DELETE THIS OPTION
            model = AdaMaxDT(tr_data,tr_labels)
            post_fix = "BDT"
        elif sys.argv[3] == "5":## TODO: KENT DELETE THIS OPTION
            model = SupportVectorMachine(len(tr_data[0])+1,0.5,0.01)
            tr_data = add_bias_term(tr_data)
            te_data = add_bias_term(te_data)
            post_fix = "SVM"
        elif sys.argv[3] == "6":## TODO: KENT DELETE THIS OPTION

            ## predict() on TRAIN DATA of yelp_train0.txt to a
            ### 0.0007 error on the TEST LABELS of yelp_test0.txt

            model = DecisionTree(tr_data,tr_labels)
            post_fix = "DT"
            model.train()
            preds = model.predict(tr_data)
            loss = zero_one_loss(preds,tr_labels)
            print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))

            model = AdaMaxDT(tr_data,tr_labels)
            post_fix = "BDT"
            model.train()
            preds = model.predict(te_data)
            loss = zero_one_loss(preds,te_labels)
            print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))

            return
        else:
            sys.exit("modelIdx must be in {1,2,3}")

        if post_fix == "SVM":
            model.train(tr_data,tr_labels)
        else:
            model.train()

        preds = model.predict(te_data)
        input_vector_to_svm(np.array(preds))
        print(np.sum(preds))
        loss = zero_one_loss(preds,te_labels)
        print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))

        w = svm(tr_data, tr_labels)
        preds = svm_pred(w, te_data)
        input_vector_to_svm(preds)
        #preds = svm.predict(te_data)
        loss = zero_one_loss(preds,te_labels)
        print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))

        from sklearn import svm as sksvm
        clf = sksvm.LinearSVC(max_iter=100,tol=0.001,loss='hinge',C=5)
        clf.fit(tr_data, tr_labels)
        preds = clf.predict(te_data)
        loss = zero_one_loss(preds,te_labels)
        print("ZERO-ONE-LOSS-" + post_fix +" {0:.4f}".format(round(float(loss),4)))


def short_tree_test():
    # trData = Data(sys.argv[1])
    # teData = Data(sys.argv[2],features=trData.features)
    # tr_data,tr_labels,_,_ = trData.split_data(1.0)
    # te_data,te_labels,_,_ = teData.split_data(1.0)
    # tr_data = add_bias_term(tr_data)
    # te_data = add_bias_term(te_data)

    tr_data,tr_labels,cw = get_data(sys.argv[1],c_words=None)
    te_data,te_labels,_ = get_data(sys.argv[2],c_words=cw)

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

    bdt = AdaMaxDT(cv_tr_data,cv_tr_labels)
    bdt.train()
    preds = bdt.predict(cv_te_data)
    loss = zero_one_loss(preds,cv_te_labels)
    print(loss)


def plot_experiment(dt_losses,bt_losses,rf_losses,bdt_losses,svm_losses,xaxis_var,xlabel,cv_split_number,fn=None):

    model_losses = {"dt":[dt_losses,"r--"],
                    "bt":[bt_losses,"b--"],
                    "rf":[rf_losses,"g--"],
                    "bdt":[bdt_losses,"y--"],
                    "svm":[svm_losses,"k--"]}

    model_lines = {}
    for model in model_losses.keys():
        if model_losses[model][0] is not None and np.size(model_losses[model][0]) != 0:
            mean = np.mean(model_losses[model][0],1)
            stderr = np.std(model_losses[model][0],1)/cv_split_number
            model_lines[model] = plt.errorbar(xaxis_var,mean,\
                    stderr,fmt=model_losses[model][1],label=model)

    plt.legend(list(model_lines.values()),list(model_lines.keys()))
    plt.xlabel(xlabel)
    plt.ylabel("Zero-One Loss")

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn,bbox_inches='tight')

    plt.gcf().clear()
    
def experiment_1(data,labels,cv_split_number):
    ## NUMBER OF EXAMPLES
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)

    dt_losses = []
    bt_losses = []
    rf_losses = []
    bdt_losses = []
    svm_losses = []

    data_balance = [0.025,0.05,0.125,0.25]    
    for i in data_balance:
        cv_tr_data = tr_data[:int(len(tr_data)*i),:1000]
        cv_tr_labels = tr_labels[:int(len(tr_labels)*i)]

        # data labels from {0,1} to {-1,1}
        input_vector_to_svm(cv_tr_labels)

        #tr_data,tr_labels,_,_ = my_data.split_data(i)
        dt_loss,bt_loss,rf_loss,bdt_loss,svm_loss = \
            k_fold_cross_validation(cv_tr_data,cv_tr_labels,cv_split_number)
        dt_losses += [dt_loss]
        bt_losses += [bt_loss]
        rf_losses += [rf_loss]
        bdt_losses += [bdt_loss]
        svm_losses += [svm_loss]

    model_losses = {"dt": dt_losses,
                    "bt": bt_losses,
                    "rf": rf_losses,
                    "bdt": bdt_losses,
                    "svm": svm_losses}

    write_losses(model_losses,data_balance,"exp1")

    return np.array(dt_losses),np.array(bt_losses),\
        np.array(rf_losses),np.array(bdt_losses),np.array(svm_losses),\
        np.array(data_balance)*tr_data_size,\
        "Cross Validation Sample Size"

def experiment_2(data,labels,cv_split_number,no_examples = 500):
    ## NUMBER OF FEATURES 
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)

    dt_losses = []
    bt_losses = []
    rf_losses = []
    bdt_losses = []
    svm_losses = []

    feature_number = [200,500,1000,1500]
    #feature_number = [200,500]
    for i in feature_number:
        cv_tr_data = tr_data[:no_examples,:i]
        cv_tr_labels = tr_labels[:no_examples]

        # data labels from {0,1} to {-1,1}
        input_vector_to_svm(cv_tr_labels)

        dt_loss,bt_loss,rf_loss,bdt_loss,svm_loss = \
            k_fold_cross_validation(cv_tr_data,cv_tr_labels,cv_split_number)
        dt_losses += [dt_loss]
        bt_losses += [bt_loss]
        rf_losses += [rf_loss]
        bdt_losses += [bdt_loss]
        svm_losses += [svm_loss]

    model_losses = {"dt": dt_losses,
                    "bt": bt_losses,
                    "rf": rf_losses,
                    "bdt": bdt_losses,
                    "svm": svm_losses}

    write_losses(model_losses,feature_number,"exp2")

    return np.array(dt_losses),np.array(bt_losses),\
        np.array(rf_losses),np.array(bdt_losses),np.array(svm_losses),\
        feature_number,"Number of Features" ## last is the x-axis for plots


def experiment_3(data,labels,cv_split_number,no_examples = 500):
    ## TREE DEPTH
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)

    dt_losses = []
    bt_losses = []
    rf_losses = []
    bdt_losses = []

    tree_depth = [5,10,15,20]
    #tree_depth = [5,10]
    for i in tree_depth:
        cv_tr_data = tr_data[:no_examples,:1000]
        cv_tr_labels = tr_labels[:no_examples]

        # data labels from {0,1} to {-1,1}
        input_vector_to_svm(cv_tr_labels)

        dt_loss,bt_loss,rf_loss,bdt_loss,_ = \
        k_fold_cross_validation(tr_data,tr_labels,cv_split_number,tree_depth=i,use_svm=False)
        dt_losses += [dt_loss]
        bt_losses += [bt_loss]
        rf_losses += [rf_loss]
        bdt_losses += [bdt_loss]

    model_losses = {"dt": dt_losses,
                    "bt": bt_losses,
                    "rf": rf_losses,
                    "bdt": bdt_losses}

    write_losses(model_losses,tree_depth,"exp3")

    return np.array(dt_losses),np.array(bt_losses),\
        np.array(rf_losses),np.array(bdt_losses),tree_depth,"Tree Depth" ## last one for xaxis


def experiment_4(data,labels,cv_split_number,no_examples = 500):
    ## NUMBER OF TREES
    
    tr_data = data
    tr_labels = labels
    tr_data_size = len(tr_data)

    bt_losses = []
    rf_losses = []

    tree_count = [10,25,50,100]
    #tree_count = [10,25]
    for i in tree_count:
        cv_tr_data = tr_data[:no_examples,:1000]
        cv_tr_labels = tr_labels[:no_examples]

        # data labels from {0,1} to {-1,1}
        input_vector_to_svm(cv_tr_labels)

        _,bt_loss,rf_loss,_ = \
    k_fold_cross_validation(tr_data,tr_labels,cv_split_number,tree_count=i,only_ensemble=True)
        bt_losses += [bt_loss]
        rf_losses += [rf_loss]

    model_losses = {"bt": bt_losses,
                    "rf": rf_losses}

    write_losses(model_losses,tree_count,"exp4")

    return np.array(bt_losses),np.array(rf_losses),\
        tree_count,"Tree Count" ## last is the x-axis for plots


if __name__ == "__main__":

    if False:
        single_model_function()
    else:
        if len(sys.argv) < 3:
            print("Usaege: python main.py <data_file> <exp_no>")
            sys.exit(1)
        exp_no = sys.argv[2]
        if exp_no not in ["1","2","3","4","-1"]:
            print("Invalid <exp_no>. Must be in {1,2,3,4,-1}")
            sys.exit(1)
        #short_tree_test()

        # a = {"A":np.array([[1,2,3,4,5],[11,12,13,14,15]]),\
        #      "B":np.array([[5,7,8,9,10],[16,17,18,19,20]])}

        # data_balance = [0.01,0.03]
        # write_losses(a,data_balance)
        
        # my_data = Data(sys.argv[1],cv_split_num=cv_split_number)
        # tr_data,tr_labels,_,_ = my_data.split_data(1.0)

        cv_split_number=10
        fast = False
        tr_data,tr_labels,_ = get_data(sys.argv[1],c_words=None,total_f=1000)

        no_examples = 500
        if fast:
            no_examples = 50
            
        if exp_no == "1" or exp_no == "-1":
            dt_losses,bt_losses,rf_losses,bdt_losses,svm_losses,xaxis_var,xlabel =\
                   experiment_1(tr_data,tr_labels,cv_split_number)
            plot_experiment(dt_losses,bt_losses,rf_losses,bdt_losses,svm_losses,\
                            xaxis_var,xlabel,cv_split_number,"./figures/exp_1.pdf")
        if exp_no == "2" or exp_no == "-1":
            dt_losses,bt_losses,rf_losses,bdt_losses,svm_losses,xaxis_var,xlabel =\
                   experiment_2(tr_data,tr_labels,cv_split_number)
            plot_experiment(dt_losses,bt_losses,rf_losses,bdt_losses,svm_losses,\
                            xaxis_var,xlabel,cv_split_number,"./figures/exp_2.pdf")
        if exp_no == "3" or exp_no == "-1":
            dt_losses,bt_losses,rf_losses,bdt_losses,xaxis_var,xlabel =\
                   experiment_3(tr_data,tr_labels,cv_split_number,no_examples)
            plot_experiment(dt_losses,bt_losses,rf_losses,bdt_losses,None,\
                            xaxis_var,xlabel,cv_split_number,"./figures/exp_3.pdf")
        if exp_no == "4" or exp_no == "-1":
            bt_losses,rf_losses,bdt_losses,xaxis_var,xlabel =\
                   experiment_4(tr_data,tr_labels,cv_split_number,no_examples)
            plot_experiment(None,bt_losses,rf_losses,bdt_losses,None,\
                            xaxis_var,xlabel,cv_split_number,"./figures/exp_4.pdf")
