#!/usr/bin/env python

import os,sys,glob,re
import numpy as np
import matplotlib.pyplot as plt

def loadFile(fn):
    with open(fn,"r") as f:
        lines = f.readlines()
    header = lines[0].strip().split(",")
    return header,np.array(list(map(lambda x: np.array([float(y) for y in  x.strip().split(",")]),lines[1:])))

def get_file_list():
    files = ','.join(glob.glob("./output/*"))
    rex = r"\./output/(?P<fn>exp(?P<exp>[0-9])_2017-[0-9]{2}-(?P<day>[0-9]{2})_(?P<hr>[0-9]{2})-(?P<min>[0-9]{2})-[0-9]+)_data_bal_(?P<bal>[0-9.]+)\.csv\,"
    info = np.asarray(re.findall(rex,files))
    names = info[:,0]
    uniq_exp = np.unique(info[:,1])
    br_exp = info[:,1]
    #exp_range = uniq_exp[:,np.newaxis][:,:,np.newaxis].repeat(len(br_exp[0]),2)    
    #res = np.squeeze(br_exp == exp_range)
    exp_list = []
    for i in range(len(uniq_exp)):
        rows_exp = np.squeeze(info[np.where(br_exp == uniq_exp[i]),:])
        uniq_types = np.unique(rows_exp[:,-1])
        type_list = []
        for j in range(len(uniq_types)):
            br_type = rows_exp[:,5]
            rows_type = np.squeeze(rows_exp[np.where(br_type == uniq_types[j]),:])
            indx = np.lexsort((rows_type[:,4],rows_type[:,3],rows_type[:,2]))
            type_list += [rows_type[indx[-1],:]]
        exp_list += [type_list]
    return np.array(exp_list)

def prepare_files(exp_list):

    ## load in files into dictionaries with/means
    exp_dicts = []
    for exp in exp_list:
        exp_vals = []
        type_label = []
        mean_dict = {}
        stde_dict = {}
        for idx in range(len(exp[:,-1])):
            fn = get_filename(exp[idx])
            heads,vals = loadFile(fn)
            mean = np.mean(vals,0)
            stde = np.std(vals,0)/10
            for jdx in range(len(heads)):
                if heads[jdx] not in mean_dict.keys():
                    mean_dict[heads[jdx]] = [mean[jdx]]
                    stde_dict[heads[jdx]] = [stde[jdx]]
                else:
                    mean_dict[heads[jdx]] += [mean[jdx]]
                    stde_dict[heads[jdx]] += [stde[jdx]]
            exp_vals += [vals]
            type_label += [exp[idx,-1]]
        tp = np.array([float(x) for x in exp[:,-1]])
        indx = np.argsort(tp)
        for k,v in mean_dict.items():
            mean_dict[k] = np.array(v)[indx]
        for k,v in stde_dict.items():
            stde_dict[k] = np.array(v)[indx]
        print(exp)
        exp_dicts += [[mean_dict,stde_dict,tp[indx],exp[0,1]]]
    return exp_dicts
    

def get_filename(_exp):
    return "./output/" + _exp[0]+"_data_bal_" + _exp[-1] + ".csv"

def plot_experiments(exp_dicts):
    i = 0 
    xlabels = ["Cross Validation Sample Size","Number of Features",\
               "Tree Depth","Tree Count"]
    for exp in exp_dicts:
        plot_experiment(exp,xlabels[int(int(exp[-1])-1)])
        i+=1
        
def plot_experiment(exp,xlabel):

    print(exp)
    model_fmt = {"dt":"r--","bt":"b--","rf":"g--","bdt":"y--","svm":"k--"}
    model_lines = {}
    xaxis_var = exp[2]
    for model in exp[0].keys():
        model_lines[model] = plt.errorbar(xaxis_var,exp[0][model],\
                    exp[1][model],fmt=model_fmt[model],label=model)

    plt.legend(list(model_lines.values()),list(model_lines.keys()))
    plt.xlabel(xlabel)
    plt.ylabel("Zero-One Loss")

    fn = "exp_" + exp[-1] + ".pdf"
    #plt.show()
    plt.savefig(fn,bbox_inches='tight')
    plt.gcf().clear()


if __name__ == "__main__":

    file_list = get_file_list()
    exp_dicts = prepare_files(file_list)    
    plot_experiments(exp_dicts)
