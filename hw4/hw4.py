###### Your ID ######
# ID1: 305674731
# ID2: 203639646
#####################

import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
    r = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if isinstance(X, pd.DataFrame):
        # Drop the 'id' and 'date' column
        X = X.drop(['date', 'id'], axis=1)
        feature_names = X.columns
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0).values
    else:
        feature_names = np.arange(X.shape[1])
    
    y = np.asarray(y, dtype=np.float64)
    
    correlations = []
    for i in range(X.shape[1]):
        feature = X[:, i]
        corr = pearson_correlation(feature, y)
        correlations.append((i, corr))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    best_features = [feature_names[idx] for idx, _ in correlations[:n_features]]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    # Helper functions
    def sigmoid(self, z):
        """
        Compute the sigmoid of z
        """
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y):
        """
        Compute the cost for logistic regression
        """
        m = len(y)
        h = self.sigmoid(X @ self.theta)
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.insert(X, 0, 1, axis=1)
        m, n = X.shape
        self.theta = np.zeros(n)
        self.thetas = []

        for i in range(self.n_iter):
            h = self.sigmoid(X @ self.theta)
            gradient = (1 / m) * (X.T @ (h - y))
            self.theta -= self.eta * gradient

            # Store the theta vector in self.thetas
            self.thetas.append(self.theta.copy())

            cost = self.compute_cost(X, y)
            self.Js.append(cost)

            if i > 0 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.insert(X, 0, 1, axis=1)
        preds = self.sigmoid(X @ self.theta)
        preds = (preds >= 0.5).astype(int)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    fold_size = len(X) // folds
    cv_accuracy = []

    for i in range(folds):
        start = i * fold_size
        end = start + fold_size if i != folds - 1 else len(X)

        X_test = X[start:end]
        y_test = y[start:end]
        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        model = algo
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        cv_accuracy.append(accuracy)

    cv_accuracy = np.mean(cv_accuracy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    data = np.asarray(data)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    
    if data.ndim == 1:
        # One-dimensional case
        p = np.exp(-0.5 * ((data - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
    else:
        # Multidimensional case
        p = np.exp(-0.5 * np.sum(((data - mu) / sigma) ** 2, axis=1)) / (np.sqrt((2 * np.pi) ** data.shape[1]) * np.prod(sigma))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_samples, n_features = data.shape
        self.weights = np.ones(self.k) / self.k
        self.mus = data[np.random.choice(n_samples, self.k, replace=False)]
        self.sigmas = np.array([np.std(data, axis=0)] * self.k)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_samples = data.shape[0]
        self.responsibilities = np.zeros((n_samples, self.k))

        for i in range(self.k):
            pdf_values = norm_pdf(data, self.mus[i], self.sigmas[i])
            self.responsibilities[:, i] = self.weights[i] * pdf_values
        
        total_responsibilities = np.sum(self.responsibilities, axis=1).reshape(-1, 1)
        self.responsibilities /= total_responsibilities
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_samples, n_features = data.shape

        for i in range(self.k):
            responsibility = self.responsibilities[:, i]
            total_responsibility = np.sum(responsibility)

            self.mus[i] = np.sum(data * responsibility[:, np.newaxis], axis=0) / total_responsibility
            diff = data - self.mus[i]
            self.sigmas[i] = np.sqrt(np.sum(responsibility[:, np.newaxis] * (diff ** 2), axis=0) / total_responsibility)
            self.weights[i] = total_responsibility / n_samples
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.costs = []
        self.init_params(data)
        prev_cost = None

        for _ in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)
            
            # Calculate the cost (negative log-likelihood)
            cost = -np.sum(np.log(np.sum([self.weights[j] * norm_pdf(data, self.mus[j], self.sigmas[j]) for j in range(self.k)], axis=0)))
            self.costs.append(cost)
            
            if prev_cost is not None and abs(prev_cost - cost) < self.eps:
                break
            
            prev_cost = cost
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    n_components = len(weights)
    pdf = 0.0  # Initialize the PDF as 0

    for i in range(n_components):
        mu = mus[i]
        sigma = sigmas[i]
        weight = weights[i]
        component_density = norm_pdf(data, mu, sigma)
        pdf += weight * component_density  # Add the weighted density of the current component

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.priors = {}
        self.params = {}

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        n_features = X.shape[1]
        classes = np.unique(y)

        for cls in classes:
            self.priors[cls] = np.mean(y == cls)  # Find prior probability
            self.params[cls] = {'means': [], 'stds': [], 'weights': []}
            data_cls = X[y == cls]
            for feature in range(n_features):
                em = EM(self.k)
                em.fit(data_cls[:, feature][:, np.newaxis])  # Fit EM to the data of the current class and feature
                weights, mus, sigmas = em.get_dist_params()
                self.params[cls]['means'].append(mus)
                self.params[cls]['stds'].append(sigmas)
                self.params[cls]['weights'].append(weights)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []
        posteriors = {}
        classes = list(self.params.keys())
        features = np.arange(X.shape[1])

        for cls in classes:
            likelihood = 1
            for feature in features:
                feature_data = X[:, feature]
                probs = np.zeros((self.k, len(feature_data)))
                for j in range(self.k):
                    mean = self.params[cls]['means'][feature][j]
                    std = self.params[cls]['stds'][feature][j]
                    weight = self.params[cls]['weights'][feature][j]
                    probs[j] = weight * norm_pdf(feature_data, mean, std)
                # Sum the probabilities and add to the likelihood
                likelihood *= np.sum(probs, axis=0)
            posterior = likelihood * self.priors[cls]
            posteriors[cls] = (posterior)

        # For each sample add the class with the highest posterior
        for i in range(len(X)):
            curr_posteriors = [posterior_class[i] for posterior_class in posteriors.values()]
            preds.append(classes[np.argmax(curr_posteriors)])

        preds = np.array(preds).reshape(-1,1)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    lor = LogisticRegressionGD(eta=best_eta, n_iter=10000, eps=best_eps)
    lor.fit(x_train, y_train)

    # Manual accuracy calculation for logistic regression
    predictions_train_lor = lor.predict(x_train)
    predictions_test_lor = lor.predict(x_test)
    lor_train_acc = np.mean(predictions_train_lor == y_train)
    lor_test_acc = np.mean(predictions_test_lor == y_test)

    # Fit Naive Bayes Gaussian with k Gaussians
    bayes = NaiveBayesGaussian(k)
    bayes.fit(x_train, y_train)
    predictions_train_bayes = bayes.predict(x_train)
    predictions_test_bayes = bayes.predict(x_test)
    bayes_train_acc = np.mean(predictions_train_bayes == y_train.reshape(-1, 1))
    bayes_test_acc = np.mean(predictions_test_bayes == y_test.reshape(-1, 1))

    print(f'Logistic Regression: Train Accuracy = {lor_train_acc}')
    print(f'Logistic Regression: Test Accuracy = {lor_test_acc}')
    print(f'Naive Bayes: Train Accuracy = {bayes_train_acc}')
    print(f'Naive Bayes: Test Accuracy = {bayes_test_acc}')

    # Plot decision regions for Logistic Regression
    plot_decision_regions(x_train, y_train, lor, title="Decision boundaries for Logistic Regression")

    # Plot decision regions for Naive Bayes
    plot_decision_regions(x_train, y_train, bayes, title="Decision boundaries for Naive Bayes")

    # Plot the cost vs iteration for the Logistic Regression model
    plt.figure(figsize=(8, 6))  # New figure for cost plotting
    plt.plot(lor.Js)  # Js should contain the cost history
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title("Logistic Regression Cost vs. Iterations")
    plt.xscale('log')  # Setting x-axis to logarithmic scale for better visualization
    plt.show()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    np.random.seed(42)
    n = 200

    # Parameters for Dataset A (Favoring Naive Bayes)
    mean_a1 = [0, 0, 0]
    cov_a1 = [[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]]
    mean_a2 = [1, 1, 1]
    cov_a2 = [[1, -0.5, -0.5], [-0.5, 1, -0.5], [-0.5, -0.5, 1]]

    # Parameters for Dataset B (Better suited for Logistic Regression)
    mean_b1 = [1, 1, 1]
    cov_b1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean_b2 = [-1, -1, -1]
    cov_b2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # Generate samples
    dataset_a_features = np.vstack([
        multivariate_normal.rvs(mean_a1, cov_a1, n),
        multivariate_normal.rvs(mean_a2, cov_a2, n)
    ])
    dataset_a_labels = np.hstack([
        np.zeros(n),
        np.ones(n)
    ])

    dataset_b_features = np.vstack([
        multivariate_normal.rvs(mean_b1, cov_b1, n),
        multivariate_normal.rvs(mean_b2, cov_b2, n)
    ])
    dataset_b_labels = np.hstack([
        np.zeros(n),
        np.ones(n)
    ])

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }


# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()