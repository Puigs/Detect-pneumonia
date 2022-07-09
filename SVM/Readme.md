# Supervised Learning Models For Classification

## Get Started

Two more libraries to install :
- PyWavelets : The wavelet transform is a mathematical technique which can decompose a signal into multiple lower resolution levels by controlling the scaling and shifting factors of a single wavelet function. We will use the Haar Wavelet in order to prepocessing the images.
- opencv-python : Library used to process an image in real time and perform operations on it.
- scikit-learn : a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.

```
pip install PyWavelets
pip install opencv-python
pip install scikit-learn
```

## The Models

### Linear model - Logistic regression

#### Definition

Logistic regression, despite its name, is a linear model for classification rather than regression. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

#### Parameters

- C : float, default=1.0
Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization. We will test different C : 1, 5, 10

- solver : ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’, default=’lbfgs’
Algorithm to use in the optimization problem. We will use the ‘liblinear’, a good choice for small datasets

- multi_class : ‘auto’, ‘ovr’, ‘multinomial’, default=’auto’
If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.

- max_iterint, default=100
Maximum number of iterations taken for the solvers to converge. We will choose 20000.

### Support Vector Machines - Classification (SVC)

#### Definition

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:
- Effective in high dimensional spaces.
- Still effective in cases where number of dimensions is greater than the number of samples.
- Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
- Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:
- If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
- SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

SVC is a class capable of performing binary and multi-class classification on a dataset.

#### Parameters

- C : float, default=1.0 
Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. We will try [1,5,10].

- kernal : ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’, default=’rbf’
Specifies the kernel type to be used in the algorithm. If none is given, ‘rbf’ will be used. We will try ‘linear’, ‘poly’, ‘rbf’

- degree : int, default=3
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels. We will keep it at default.

- coef0	: float, default=0.0
Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’. We will try 0 and 1.

### Ensemble methods - Forests of randomized trees - Random Forests

#### Defnition

The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability / robustness over a single estimator.

The sklearn.ensemble module includes two averaging algorithms based on randomized decision trees: the RandomForest algorithm and the Extra-Trees method. Both algorithms are perturb-and-combine techniques [B1998] specifically designed for trees. This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.

As other classifiers, forest classifiers have to be fitted with two arrays: a sparse or dense array X of shape (n_samples, n_features) holding the training samples, and an array Y of shape (n_samples,) holding the target values (class labels) for the training samples.

Like decision trees, forests of trees also extend to multi-output problems (if Y is an array of shape (n_samples, n_outputs)).

In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set.

Furthermore, when splitting each node during the construction of a tree, the best split is found either from all input features or a random subset of size max_features. (See the parameter tuning guidelines for more details).

### Nearest Neighbors - Nearest Neighbors Classification

#### Defnition

Neighbors-based classification is a type of instance-based learning or non-generalizing learning: it does not attempt to construct a general internal model, but simply stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point: a query point is assigned the data class which has the most representatives within the nearest neighbors of the point.

## The preprocessing

### Haar wavelet 

### HOG

### Images size
