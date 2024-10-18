"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC


class SVM(object):
    """
    SVM method.
    """

    def __init__(self, C, kernel, gamma=1., degree=5, coef0=0.):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """

        self.C = C
        self.kernel = kernel 
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        
        self.model = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        self.model.fit(training_data, training_labels)
        
        # Use the trained model to predict the labels of the training data
        pred_labels = self.model.predict(training_data)

        return pred_labels

        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
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
        pred_labels = self.model.predict(test_data)

        return pred_labels