import os,sys,re,math,string
import numpy as np
from collections import Counter

class Data():
    
    def __init__(self,data_path,ftr_num = 1000,cv_split_num = 10,features=None,skip=100):
        self._data_path = data_path
        self.feature_number = ftr_num
        self.skip = skip
        self._data,self.labels = self.parse_data()
        self._d_size = len(self._data)
        self.features = features
        if features == None:
            #print(self._data[0:2])
            self.features = self.loadFeatures()
        self.data = np.zeros([self._d_size,self.feature_number])
        self.data_to_numbers() ## augment data with bias term here
        self.randomize_indicies(False)
        self.cv_split_number = cv_split_num
        self.cv_index = 0
        
    def process_str(self,s):
        rem_punc = str.maketrans('', '', string.punctuation)
        return s.translate(rem_punc).lower().split()

    def parse_data(self):
        lines = self.readfile()
        data = [list(re.sub(r'[^a-zA-Z0-9 ]','', l.split("\t")[2:][0]\
                            .lower()).split()) for l in lines]
        labels = np.array([int(l.split("\t")[1][0]) for l in lines])
        return data,labels

    def readfile(self):
        if not os.access(self._data_path, os.R_OK):
            print("File does not exist")
            return None
        with open(self._data_path,"r") as f:
            lines = f.readlines()
        return lines

    def loadFeatures(self):
        counter = Counter()
        for item in self._data:
            counter = counter + Counter(set(item))
        temp = counter.most_common(self.feature_number+self.skip)[self.skip:]
        words = [item[0] for item in temp]
        return words
        #nlines = [item for sub in [list(l) for l in self._data] for item in sub]
        #flis = Counter(nlines).most_common(self.feature_number+100) # 100+feature_number
        #print(Counter(nlines).most_common(self.feature_number+100)[:100],"K")
        #return [flis[k][0] for k in range(len(flis)) if k >= 100] # grab AFTER top 100

    def data_to_numbers(self):
        for i in range(self._d_size):
            for j in self._data[i]:
                if j in self.features:
                    if self.data[i][self.features.index(j)] != 1:
                        self.data[i][self.features.index(j)] += 1

    def labels_to_svm(self):
        for i in range(self._d_size):
            if self.labels[i] == 0: 
                self.labels[i] = -1

    def labels_to_lr(self):
        for i in range(self._d_size):
            if self.labels[i] == -1: 
                self.labels[i] = 0

    def randomize_indicies(self,random):
        if random:
            self._indicies = np.random.permutation(self._d_size)
        else:
            self._indicies = np.arange(self._d_size)

    def split_data(self,per):
        # per -- percent is a float from 0 to 1
        self.tr_size = int(math.floor(self._d_size * per))
        self.cv_size = self.tr_size / self.cv_split_number
        return self.data[self._indicies[0:int(math.floor(self._d_size * per))]],self.labels[self._indicies[0:int(math.floor(self._d_size * per))]],\
            self.data[self._indicies[int(math.floor(self._d_size * per)):]],self.labels[self._indicies[int(math.floor(self._d_size * per)):]]

    def cv_split(self,tr_data,tr_labels,cv_index):
        tr_index_start = int(self.cv_size * cv_index)
        tr_index_end = int(tr_index_start + self.cv_size)
        return np.concatenate((tr_data[0:tr_index_start],tr_data[tr_index_end:]),0),\
            np.concatenate((tr_labels[0:tr_index_start],tr_labels[tr_index_end:]),0),\
            tr_data[tr_index_start:tr_index_end],tr_labels[tr_index_start:tr_index_end]
        #tr_index_start = int(math.floor(self._d_size * float(cv_index/ self.cv_split_number)))
        # return tr_data[int(tr_index * self.cv_size) : int(tr_index*self.cv_size + self.cv_size)],\
        #     tr_labels[int(tr_index * self.cv_size) : int(tr_index*self.cv_size + self.cv_size)],\
        #     tr_data[int(tr_index*self.cv_size + self.cv_size):],\
        #     tr_labels[int(tr_index*self.cv_size + self.cv_size):]
