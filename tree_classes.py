import numpy as np

class DecisionTree():

    def __init__(self,data,labels,max_depth=10,randomForestMode=False,randomForestIndicies=None):
        assert(len(labels) != 0)
        self.max_depth = max_depth
        self.current_depth = 0
        self.data = data
        self.labels = labels
        self.label_values = np.unique(self.labels)
        self.index = None
        self.left = None
        self.right = None
        self.label = -1
        self.current_gini_index = self.gini_index(self.labels)
        self.randomForestIndicies = randomForestIndicies
        self.randomForestMode = randomForestMode
        if self.randomForestMode is True:
            assert(randomForestIndicies is not None)
            if len(randomForestIndicies) != 0:
                self.random_features = randomForestIndicies\
                    [np.random.permutation(len(randomForestIndicies))]\
                    [:int(np.sqrt(len(randomForestIndicies)))]

    def predict(self,data):
        return list(map(lambda x: self.evaluate_sample(x),data))
        
    def train(self):
        if self.max_depth == 1 or len(self.labels) < self.max_depth: ## check if leaf node
            self.label = np.argmax(np.bincount(self.labels))
            return
        self.index = self.cost() ## returns the data values with the best split of data
        if self.index == -1:
            self.label = np.argmax(np.bincount(self.labels))
            return
        data_col = self.data[:,self.index]
        self.left = DecisionTree(self.data[np.where(data_col==1)],\
                            self.labels[np.where(data_col==1)],\
                            self.max_depth-1,randomForestMode=self.randomForestMode,\
                                 randomForestIndicies=self.randomForestIndicies)
        self.right = DecisionTree(self.data[np.where(data_col==0)],\
                            self.labels[np.where(data_col==0)],\
                            self.max_depth-1,randomForestMode=self.randomForestMode,\
                                 randomForestIndicies=self.randomForestIndicies)
        self.right.train()
        self.left.train()

        
    def cost(self):
        gains = []
        for x in range(self.data.shape[1]):
            if self.randomForestMode is True and x not in self.randomForestIndicies:
                gains.append(0)
                continue
            split_a = self.labels[np.where(self.data[:,x]==1)]
            split_b = self.labels[np.where(self.data[:,x]==0)]
            if len(split_a) is 0 or len(split_b) is 0:
                gain = 0
            else:
                gain = self.gini_gain([split_a,split_b])
            gains.append(gain)
        argmax = np.argmax(gains)
        split_a = self.labels[np.where(self.data[:,argmax]==1)]
        split_b = self.labels[np.where(self.data[:,argmax]==0)]
        if len(split_a) is 0 or len(split_b) is 0:# or max(gains) == 0:
            argmax = -1
        return argmax

    def gini_gain(self,split_labels):
        return self.current_gini_index\
            - (np.size(split_labels[0])*self.gini_index(split_labels[0])\
               +np.size(split_labels[1])*self.gini_index(split_labels[1]))\
               /np.size(self.labels)

    def gini_index(self,labels):
        return 1 - (np.square(np.size(np.where(labels==0)))\
            +np.square(np.size(np.where(labels==1))))/np.square(np.size(labels))

    def getLeftChild(self):
        return self.left

    def getRightChild(self):
        return self.right

    def to_str_index(self):
        if self.right is None or self.left is None:
            return ""
        a = str(self.index) + ","
        a+=self.left.to_str_index()
        a+=self.right.to_str_index()
        return a

    def print_index(self):
        if self.right is None or self.left is None:
            return 

        self.left.print_index()
        print(self.index)
        self.right.print_index()
        return

    def check_index(self):
        if self.right is None or self.left is None:
            return []
        t = self.left.check_index()
        t+= self.right.check_index()
        if self.index in t:
            print("\n\nERROR\n\n")
        return t


    def evaluate_sample(self,data):
        ## data is a numpy array of length n
        if self.label is not -1:
            return self.label
        if data[self.index] == 1:
            return self.left.evaluate_sample(data)
        else:
            return self.right.evaluate_sample(data)

class Bagging():

    def __init__(self,data,labels,tree_count=50,max_depth=10):
        assert(len(labels) != 0)
        assert(len(labels) == len(data))
        assert(type(data) is np.ndarray)
        self.data = data
        self.labels = labels
        self.label_values = np.unique(self.labels)
        self.tree_count = tree_count
        self.max_depth = max_depth
        self.trees = None
        self.subset_size = self.data.shape[0]

    def train(self):
        dts = []
        for index in range(self.tree_count):
            sample_data,sample_label = self.sample_data()
            dts += [DecisionTree(sample_data,sample_label)]
            dts[-1].train()
        self.trees = dts
        
    def predict(self,data):
        if self.trees is None:
            return None
        count = 0
        preds = []
        for tree in self.trees:
            pred = tree.predict(data)
            preds += [np.array(pred)]
        # print(np.array(np.mean(np.array(preds).T,1)>0.5).astype(np.int))
        # print(len(np.array(np.mean(np.array(preds).T,1)>0.5).astype(np.int)))
        # print(len(data))
        return \
                np.array(np.mean(np.array(preds).T,1)>0.5).astype(np.int)

    def sample_data(self):
        indicies = np.random.randint(0,self.data.shape[0],self.subset_size)
        return self.data[indicies,:],\
            self.labels[indicies]

class RandomForest():

    def __init__(self,data,labels,tree_count=50,max_depth=10):
        assert(len(labels) != 0)
        assert(len(labels) == len(data))
        assert(type(data) is np.ndarray)
        self.data = data
        self.labels = labels
        self.label_values = np.unique(self.labels)
        self.tree_count = tree_count
        self.max_depth = max_depth
        self.trees = None
        self.subset_size = self.data.shape[0] # or *.68?

    def train(self):
        dts = []
        for index in range(self.tree_count):
            sample_data,sample_label = self.sample_data()
            dts += [DecisionTree(sample_data,sample_label,\
                randomForestMode=True,randomForestIndicies=np.arange(len(sample_data)))]
            dts[-1].train()
        self.trees = dts

        
    def predict(self,data):
        if self.trees is None:
            return None
        count = 0
        preds = []
        for tree in self.trees:
            pred = tree.predict(data)
            #print(pred[0:30])
            preds += [np.array(pred)]
        # print(np.array(preds).shape)
        # print(np.mean(np.array(preds).T,1).shape)
        # print(len(np.mean(np.array(preds).T,1)))
        return \
                np.array(np.mean(np.array(preds).T,1)>0.5).astype(np.int)

    def sample_data(self):
        indicies = np.random.randint(0,self.data.shape[0],self.subset_size)
        return self.data[indicies,:],\
            self.labels[indicies]
