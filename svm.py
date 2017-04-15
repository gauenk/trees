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
        loc = np.where(preds < 1, -self.tr_data.T @ np.diag(self.tr_labels), 0)
        return np.mean(loc,1)

    def sub_gradient_update(self,error):
        self.weights = self.weights - self.lr * ( self.decay * self.weights + error )

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
