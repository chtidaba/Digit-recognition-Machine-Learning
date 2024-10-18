# ML Digit Recognition Project

## Overview
This project implements multiple **machine learning algorithms** to classify handwritten digit images, similar to the **MNIST** dataset. The goal is to compare the performance of different models, including **K-Means clustering**, **Logistic Regression**, and **Support Vector Machines (SVM)**, on the task of digit recognition.

### Algorithms Implemented:
1. **K-Means Clustering**
2. **Logistic Regression**
3. **Support Vector Machine (SVM)**

## Dataset
- The dataset consists of images of handwritten digits ranging from **0 to 9**.
- Each image is flattened into a feature vector that serves as input to the algorithms.
- The dataset is split into a **training set** and a **test set** for evaluating the models.

## Models

### 1. K-Means Clustering
**K-Means** is an **unsupervised learning** algorithm that groups data points into **clusters** based on their similarity. For digit recognition:
- Each digit image is treated as a point in a high-dimensional space.
- K-Means tries to **cluster similar images** together.
- After clustering, we assign labels to the clusters based on the majority class within each cluster.

#### Performance:
- **100 Iterations**:
  - **Train Set**: Accuracy = 16.370%, F1-Score = 0.062758
  - **Test Set**: Accuracy = 17.137%, F1-Score = 0.060141
- **1000 Iterations**:
  - **Train Set**: Accuracy = 18.616%, F1-Score = 0.095807
  - **Test Set**: Accuracy = 20.151%, F1-Score = 0.099558

**Conclusion**: 
- K-Means performed poorly for this task because it is **unsupervised** and does not leverage the label information during training.
- It was useful to see how data clusters together, but it’s not effective for precise classification on this dataset.

### 2. Logistic Regression
**Logistic Regression** is a **supervised learning** algorithm used for **classification tasks**. It learns a **linear decision boundary** that separates classes by estimating probabilities:
- We applied **softmax** to handle the multiclass classification problem.
- Different **learning rates** were tested to find the best performance.

#### Performance:
- **Learning Rate = 0.1**:
  - **Train Set**: Accuracy = 27.689%, F1-Score = 0.206576
  - **Test Set**: Accuracy = 26.742%, F1-Score = 0.175537
- **Learning Rate = 0.01**:
  - **Train Set**: Accuracy = 90.178%, F1-Score = 0.902184
  - **Test Set**: Accuracy = 90.395%, F1-Score = 0.904680

**Conclusion**:
- Logistic Regression achieved **high accuracy** with a learning rate of **0.01**, showing it is effective for digit classification.
- When the learning rate was too high (0.1), the model was not able to converge well, leading to poor performance.
- Proper tuning of **learning rate** is crucial for achieving good performance.

### 3. Support Vector Machine (SVM)
**SVM** is a supervised learning algorithm that finds the **optimal hyperplane** to separate classes. It can use **different kernels** to handle non-linear decision boundaries:
- We experimented with different **polynomial kernels** and **hyperparameters** to achieve the best performance.

#### Performance:
- **SVM with C = 1, Kernel = Poly, Gamma = 1, Degree = 1**:
  - **Train Set**: Accuracy = 100.000%, F1-Score = 1.000000
  - **Test Set**: Accuracy = 95.104%, F1-Score = 0.950515
- **SVM with C = 1, Kernel = Poly, Gamma = 1, Degree = 5**:
  - **Train Set**: Accuracy = 100.000%, F1-Score = 1.000000
  - **Test Set**: Accuracy = 96.422%, F1-Score = 0.962477

**Conclusion**:
- SVM achieved the **best performance** among the three methods.
- The model trained on **polynomial kernels** of **higher degrees** achieved **better accuracy**, suggesting that a more complex decision boundary was needed to capture the patterns in the data.
- **Overfitting** was not an issue here, even though the train accuracy was 100%, because the **test accuracy** was also high.

## Summary of Results:
| Model                 | Train Accuracy | Train F1-Score | Test Accuracy | Test F1-Score |
|-----------------------|----------------|----------------|---------------|---------------|
| K-Means (100 iters)   | 16.370%        | 0.062758       | 17.137%       | 0.060141      |
| K-Means (1000 iters)  | 18.616%        | 0.095807       | 20.151%       | 0.099558      |
| Logistic Regression (lr=0.1) | 27.689% | 0.206576       | 26.742%       | 0.175537      |
| Logistic Regression (lr=0.01) | 90.178% | 0.902184       | 90.395%       | 0.904680      |
| SVM (C=1, Degree=1)   | 100.000%       | 1.000000       | 95.104%       | 0.950515      |
| SVM (C=1, Degree=5)   | 100.000%       | 1.000000       | 96.422%       | 0.962477      |

## Conclusion:
- **SVM** was the **most effective** model for this task, achieving over **95% accuracy** on the test set, showing its robustness for digit classification.
- **Logistic Regression** also performed well, particularly when the **learning rate** was appropriately set.
- **K-Means**, being unsupervised, was less effective but provided insights into how the data naturally clusters.
- **Model tuning** (hyperparameters and learning rates) played a significant role in the success of the models.

## Future Improvements:
1. **Data Augmentation**: Enhance the dataset by including transformed versions of the images (rotations, shifts) to improve generalization.
2. **Model Tuning**: Experiment with other **kernels** for SVM, such as **RBF**, and other **learning rates** for Logistic Regression.
3. **Dimensionality Reduction**: Apply **PCA** or other dimensionality reduction techniques to speed up training and potentially improve performance.

## How to Run:
python main.py --data <where you placed the data folder> --method <method you want to use (svm for example)>

### Acknowledgments:
Thank you for using our digit recognition project. We hope this serves as a helpful guide for understanding different machine learning techniques and their applications in image classification tasks.