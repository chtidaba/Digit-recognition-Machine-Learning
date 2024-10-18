import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=1000):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        D = data.shape[1]

        cluster_assignments = np.zeros((data.shape[0]), dtype=int)
        centers = data[np.random.choice(data.shape[0], self.K, replace=False)]


        for z in range(max_iter):
        
            for i in range(data.shape[0]): 
                min_distance = float('inf')
                for j in range(self.K): 
                    distance = np.linalg.norm(data[i] - centers[j])
                    if distance < min_distance:
                        min_distance = distance
                        cluster_assignments[i] = j

            for i in range(self.K):
                points_in_cluster = data[cluster_assignments == i]
                if len(points_in_cluster) > 0:
                    centers[i] = np.mean(points_in_cluster, axis=0)

        return centers, cluster_assignments
    


    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        centers, cluster_assignments = self.k_means(training_data, self.max_iters)
    
    # Determine the most frequent label for each cluster
        cluster_labels = np.zeros(self.K)
        for i in range(self.K):
            # Get the labels of data points that belong to cluster `i`
            labels_in_cluster = training_labels[cluster_assignments == i]
        
        if len(labels_in_cluster) > 0:
            # Find the most common label in this cluster
            cluster_labels[i] = np.bincount(labels_in_cluster).argmax()
    
        # Save the learned centers and cluster labels for prediction
        self.centers = centers
        self.cluster_labels = cluster_labels

        # Return the labels assigned during training
        pred_labels = np.array([cluster_labels[cluster_assignments[i]] for i in range(cluster_assignments.shape[0])])

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        pred_labels = []
        for i in range(test_data.shape[0]):
            min_distance = float('inf')
            closest_cluster = -1
            for j in range(self.K):
                distance = np.linalg.norm(test_data[i] - self.centers[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = j
            # Use the label of the closest cluster
            pred_labels.append(self.cluster_labels[closest_cluster])
    
        return np.array(pred_labels)