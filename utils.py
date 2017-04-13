import sys,string,datetime,csv
import numpy as np
from collections import Counter

def input_vector_to_svm(labels_i):
    for i in range(len(labels_i)):
        if labels_i[i] == 0: 
            labels_i[i] = -1

def input_vector_from_svm(labels_i):
    for i in range(len(labels_i)):
        if labels_i[i] == -1: 
            labels_i[i] = 0

def zero_one_loss(preds,labels):
    return len([1 for i,j in zip(preds,labels) if i != j])/len(labels)

def add_bias_term(data):
    return np.hstack((data,np.ones(data.shape[0])[:,np.newaxis]))

def remove_bias_term(data):
    return data[:,:-1]

def calc_error(pred, labels):
    error = sum(np.where(pred != labels, 1, 0))
    return (error / labels.size)

def process_str(s):
    rem_punc = str.maketrans('', '', string.punctuation)
    return s.translate(rem_punc).lower().split()

def read_dataset(file_name):
    dataset = []
    with open(file_name) as f:
        for line in f:
            index, class_label, text = line.strip().split('\t')
            words = process_str(text)
            dataset.append( (int(class_label), words) )
    return dataset

def get_most_commons(dataset, skip=100, total=1000):
    counter = Counter()

    for item in dataset:
        counter = counter + Counter(set(item[1]))
    temp = counter.most_common(total+skip)[skip:]
    words = [item[0] for item in temp]
    return words

def generate_vectors(dataset, common_words):
    d = {}
    for i in range(len(common_words)):
        d[common_words[i]] = i

    vectors = []
    labels = []
    for item in dataset:
        vector = [0] * len(common_words)
        # Intercept term.
        vector.append(1)

        for word in item[1]:
            if word in d:
                vector[d[word]] = 1

        vectors.append(vector)
        labels.append(item[0])

    return np.array(vectors), np.array(labels)


def get_data(fn,skip=100,total_f=1500,c_words=None):
    train_data_file = fn
    train_data = read_dataset(train_data_file)
    common_words = c_words
    if common_words is None:
        common_words = get_most_commons(train_data, skip=100, total=total_f)
    train_f, train_l = generate_vectors(train_data, common_words)
    return train_f,train_l,common_words

def split_data(data,labels,split_number,cv_index):

    data_size = len(data)
    cv_size = int(data_size/split_number)

    tr_index_A = int((cv_index+1)*cv_size % data_size)
    tr_index_B = int((cv_index+2)*cv_size % data_size)

    tr_index_C = int((cv_index+2)*cv_size % data_size)
    tr_index_D = int((cv_index+3)*cv_size % data_size)

    te_index_A = int(cv_index*cv_size)
    te_index_B = int((cv_index+1)*cv_size)
    
    
    cv_tr_data = np.concatenate((data[tr_index_A:tr_index_B],\
                                 data[tr_index_C:tr_index_D]),0)
    cv_tr_labels = np.concatenate((labels[tr_index_A:tr_index_B],\
                                 labels[tr_index_C:tr_index_D]),0)
    cv_te_data = data[te_index_A:te_index_B]
    cv_te_labels = labels[te_index_A:te_index_B]

    return cv_tr_data,cv_tr_labels,cv_te_data,cv_te_labels

def write_stats(model_losses):
    fn = "./output/output_"+str(datetime.datetime.now()).replace(" ","_").replace(":","-").replace(".","") + ".csv"
    suffix = ["_losses","_means","_stderr"]
    suffix_idx = 0
    with open(fn, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, values in model_losses.items():
            for item in values:
                output = [key+suffix[suffix_idx]]
                for val in values[suffix_idx]:
                    output += [str(val)]
                suffix_idx = (suffix_idx + 1) % len(suffix)
                writer.writerow(output)

        
def write_losses(model_losses,data_balance,exp_no):


    fn = "./output/" + exp_no + "_" +str(datetime.datetime.now()).replace(" ","_").replace(":","-").replace(".","")
    for idx in range(len(data_balance)):
        fn_c = fn + "_data_bal_"+str(data_balance[idx]) + ".csv"
        with open(fn_c, 'w') as csv_file:
            writer = csv.writer(csv_file)
            title = []
            data = []
            for key,value in model_losses.items():
                #title += [key+"_"+str(data_balance[idx])]
                title += [key]
                data += [value[idx]]
            data = np.array(data)
            writer.writerow(title)
            for val in data.T:
                writer.writerow(val)
        

