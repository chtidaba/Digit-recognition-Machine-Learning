import numpy as np
import math

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters


    def softmax(self, W, x, b): 
        z = np.dot(W, x) + b
        y_hat = []
            
        denominator = 0
        denominator = sum(math.exp(z_i) for z_i in z)

        for i in range (z.shape[0]):
            y_hat.append(math.exp(z[i]/denominator))  

        return y_hat

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        # first we need to construct our W matrix 
        W = np.zeros((10, training_data.shape[1]))
        b = np.zeros(10)
        
        for _ in range(self.max_iters):
            dW = np.zeros_like(W)
            db = np.zeros_like(b)
            
            for i in range(training_data.shape[0]):
                y_hat = self.softmax(W, training_data[i], b)
                

                y_true = np.zeros(10)
                y_true[training_labels[i]] = 1
                
                for k in range(10):
                    dW[k, :] += (y_hat[k] - y_true[k]) * training_data[i]
                    db[k] += (y_hat[k] - y_true[k])
            
            dW /= training_data.shape[0]
            db /= training_data.shape[0]
            
            W -= self.lr * dW
            b -= self.lr * db
        
        self.W = W
        self.b = b

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        pred_labels = []
        for i in range(test_data.shape[0]):
            y_hat = self.softmax(self.W, test_data[i], self.b)
            pred_labels.append(np.argmax(y_hat))
        return pred_labels
