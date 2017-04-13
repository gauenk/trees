import numpy as np

class SupportVectorMachine():

    def __init__(self,weight_size,lr,decay,optim_method="grad",tol=10**-6,iterations=100):
        self.tolerance = tol
        self.iterations = iterations
        self.weights = np.array([0 for i in range(weight_size)])
        self.lr = lr
        self.decay = decay
        self.update_function = self.sub_gradient_update
        if optim_method in ["grad"]:
            self.update_function = self.sub_gradient_update

    def predict(self,te_data):
        pred = np.matmul(te_data,self.weights)
        preds = [0 for _ in range(len(pred))]
        for idx in range(len(pred)):
            if pred[idx] > 0:
                preds[idx] = 1
        return np.array(preds)

    def forward(self):
        return np.matmul(self.tr_data,self.weights)

    def computer_error(self):
        preds = self.forward()
        preds = np.multiply(self.tr_labels,preds)
        prev_pred = len(preds)
        weight_length = len(self.weights)
        out_preds = []
        for i in range(len(preds)):
            if preds[i] < 1:
                out_preds += [self.tr_labels[i]*self.tr_data[i]]
            else:
                out_preds += [[0 for _ in range(weight_length)]]
        if len(out_preds) == 0:
            m_pred = [0 for i in range(len(self.tr_data))]
        else:
            m_pred = np.mean(out_preds,0)
        return m_pred

    def sub_gradient_update(self,error):
        self.weights = self.weights - self.lr * ( self.decay * self.weights - error )

    def train(self,tr_data_i,tr_labels):
        # init weights
        prev_weights = self.weights
        self.tr_data = tr_data_i
        self.tr_labels = tr_labels
        for i in range(self.iterations):
            error = self.computer_error()
            prev_weights = self.weights
            self.update_function(error)
            if np.linalg.norm(self.weights - prev_weights) < self.tolerance:
                break
