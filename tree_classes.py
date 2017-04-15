import numpy as np

class DecisionTree():

    def __init__(self,data,labels,max_depth=10,current_depth=0\
                 ,randomForestMode=False,weights=None):
        assert(len(labels) != 0)
        self.max_depth = max_depth
        self.current_depth = current_depth
        #print(len(labels))
        self.data = data
        self.data_length = len(data)
        self.labels = labels
        self.label_values = np.unique(self.labels)
        self.index = -1
        self.left = None
        self.right = None
        self.label = 0
        self.weights = weights
        self.data_weight_sum = None
        self.current_gini_index = None
        self.randomForestMode = randomForestMode
        if self.randomForestMode is True:
            self.random_features = np.random.permutation(self.data.shape[1])\
                                   [:int(np.sqrt(len(labels)))]

    def predict(self,data):
        return list(map(lambda x: self.evaluate_sample(x),data))
        
    def train(self):
        if self.weights is not None:
            self.data_weight_sum = np.sum(self.weights)
        if self.max_depth <= self.current_depth \
           or len(self.labels) < self.max_depth or \
        self.data_weight_sum == 0: ## check if leaf node
            self.assign_label()
            #print(self.label,self.current_depth,len(self.labels))
            return

        self.index = self.cost() ## returns the data values with the best split of data

        if self.index == -1:
            self.assign_label()
            return
        data_col = self.data[:,self.index]
        weights_left = None
        weights_right = None
        if self.weights is not None:
            weights_left = self.weights[np.where(data_col==1)]
            weights_right = self.weights[np.where(data_col==0)]

            
        self.left = DecisionTree(self.data[np.where(data_col==1)],\
                            self.labels[np.where(data_col==1)],\
                            self.max_depth,randomForestMode=True,\
                            #self.randomForestMode,\
                            current_depth=self.current_depth+1,\
                            weights=weights_left)
        self.right = DecisionTree(self.data[np.where(data_col==0 )],\
                            self.labels[np.where(data_col==0)],\
                            self.max_depth,randomForestMode=True,\
                            #self.randomForestMode,\
                            current_depth=self.current_depth+1,\
                            weights=weights_right)
        self.left.train()
        self.right.train()
        
    def cost(self):
        gains = []
        self.current_gini_index = self.gini_index(self.labels)
        if self.weights is not None:
            self.current_gini_index = self.gini_index(np.arange(len(self.labels)))
        for x in range(self.data.shape[1]):
            if self.randomForestMode is True and x not in self.random_features:
                gains.append(0)
                continue
            if self.weights is not None:
                split_a = np.where(self.data[:,x]==1)[0]
                split_b = np.where(self.data[:,x]==0)[0]
            else:
                split_a = self.labels[np.where(self.data[:,x]==1)[0]]
                split_b = self.labels[np.where(self.data[:,x]==0)[0]]

            if len(split_a) is 0 or len(split_b) is 0:
                gain = 0
            else:
                gain = self.gini_gain([split_a,split_b])
            gains.append(gain)

        argmax = np.argmax(gains)
        ## Check argmax has at least some split
        split_a = self.labels[np.where(self.data[:,argmax]==1)[0]]
        split_b = self.labels[np.where(self.data[:,argmax]==0)[0]]

        if len(split_a) == 0 or len(split_b) == 0 or max(gains) <= 0:
            argmax = -1

        return argmax

    def gini_gain(self,split_labels):
        if self.weights is not None:
            return self.current_gini_index -\
                (np.sum(self.weights[split_labels[0]])*self.gini_index(split_labels[0])+
                 np.sum(self.weights[split_labels[1]])*self.gini_index(split_labels[1]))/\
                self.data_weight_sum
        return self.current_gini_index\
            - (np.size(split_labels[0])*self.gini_index(split_labels[0])\
               +np.size(split_labels[1])*self.gini_index(split_labels[1]))\
               /self.data_length

    def gini_index(self,labels):
        if self.weights is not None:
            return 1 - (np.square(np.sum(self.weights[labels[(self.labels[labels]==-1)]]))+\
                        np.square(np.sum(self.weights[labels[self.labels[labels]==1]])))/\
                        np.square(np.sum(self.weights[labels]))
        return 1 - (np.square(np.size(np.where(labels==-1)[0]))\
            +np.square(np.size(np.where(labels==1)[0])))/np.square(np.size(labels))

# def gini_index(xlabels):
#     xlabels = np.array(xlabels)
#     if weights is not None:
#         return 1 - (np.square(np.sum(weights[xlabels[labels[xlabels]==-1]]))+\
#                     np.square(np.sum(weights[xlabels[labels[xlabels]==1]])))/\
#                     np.square(np.sum(weights[xlabels]))
#     return 1 - (np.square(np.size(np.where(xlabels==-1)[0]))\
#         +np.square(np.size(np.where(xlabels==1)[0])))/np.square(np.size(xlabels))


    def assign_label(self):
        if self.weights is None:
            val,cnt = np.unique(self.labels,return_counts=True)
            self.label = val[np.argmax(np.unique(self.labels,return_counts=True)[1])]
        else:
            sum_neg = np.sum(self.weights[np.where(self.labels==-1)])
            sum_pos = np.sum(self.weights[np.where(self.labels==1)])
            self.label = 1 if sum_pos > sum_neg else -1

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
        if self.index == -1:
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
        self.subset_size = int(self.data.shape[0])#*.68)

    def train(self):
        dts = []
        for index in range(self.tree_count):
            sample_data,sample_label = self.sample_data()
            dts += [DecisionTree(sample_data,sample_label,max_depth=self.max_depth)]
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
        rtn_val = np.array(np.mean(np.array(preds).T,1)>=0).astype(np.int)
        rtn_val[rtn_val==0] = -1
        return rtn_val

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
        self.subset_size = int(self.data.shape[0])#*.68) # or *.68?

    def train(self):
        dts = []
        for index in range(self.tree_count):
            sample_data,sample_label = self.sample_data()
            dts += [DecisionTree(sample_data,sample_label,\
                                 randomForestMode=False)]
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
        rtn_val = np.array(np.mean(np.array(preds).T,1)>0).astype(np.int)
        rtn_val[rtn_val==0] = -1
        return rtn_val
                

    def sample_data(self):
        indicies = np.random.randint(0,self.data.shape[0],self.subset_size)
        return self.data[indicies,:],\
            self.labels[indicies]

class AdaMaxDT():

    epsilon = 10**-10

    def __init__(self,data,labels,tree_count=50,max_depth=10):
        assert(len(labels) != 0)
        assert(len(labels) == len(data))
        assert(type(data) is np.ndarray)
        self.data = data
        self.labels = labels
        self.weights = np.ones(len(data))/len(data)
        self.label_values = np.unique(self.labels)
        self.tree_count = tree_count
        self.max_depth = max_depth
        self.trees = None
        self.subset_size = self.data.shape[0] #*.68 # or *.68?

    def train(self):
        self.trees = []
        prev_weights = np.copy(self.weights)
        for index in range(self.tree_count):

            dt = DecisionTree(self.data,self.labels,\
                                 weights=self.weights)
            dt.train()
            preds = dt.predict(self.data)
            error = self.zero_one_loss(preds,self.labels)
            alpha = 0.5*np.log((1-error+self.epsilon)/(error+self.epsilon))
            self.weights *= np.exp(-self.labels*alpha*preds)
            self.weights /= np.sum(self.weights)
            if np.array_equal(self.weights,prev_weights):
                rands = np.random.randint(0,2,len(self.weights))
                self.weights += (rands)/1000
                self.weights /= np.sum(self.weights)
            prev_weights = np.copy(self.weights)
            self.trees += [[dt,alpha]]
        
    def predict(self,data):
        if self.trees is None:
            return None
        count = 0
        preds = []
        for tree in self.trees:
            pred = tree[0].predict(data)
            preds += [np.array(pred)*tree[1]]
        rtn_val = np.array(np.mean(np.array(preds).T,1)>0).astype(np.int)
        rtn_val[rtn_val==0] = -1
        return rtn_val

    def sample_data(self):
        indicies = np.random.randint(0,self.data.shape[0],self.subset_size)
        return self.data[indicies,:],\
            self.labels[indicies]

    def zero_one_loss(self,preds,labels):
        return len(np.where(preds != labels)[0])/len(labels)
        #return np.sum(self.weights[np.where(preds != labels)[0]])/np.sum(self.weights)
