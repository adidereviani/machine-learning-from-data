###### Your ID ######
# ID1: 305674731
# ID2: 203639646
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    X = (X - np.mean(X, axis=0)) / np.ptp(X, axis=0)
    y = (y - np.mean(y)) / np.ptp(y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    X = np.column_stack((np.ones(len(X), dtype=X.dtype), X))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X

def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """
    
    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    J = np.sum(np.square(np.dot(X, theta) - y)) / (2 * len(y))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    learning_rate_per_example = alpha / len(y)
    for _ in range(num_iters):
        prediction_error = np.dot(X, theta) - y
        J_history.append(np.sum(prediction_error**2) / (2 * len(y)))
        gradient = np.dot(X.T, prediction_error)
        theta -= learning_rate_per_example * gradient
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    pinv_theta = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T, y))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    alpha_per_example = alpha / len(y) # Precompute for efficiency
    for i in range(num_iters):
        cost = compute_cost(X, y, theta)
        J_history.append(cost)
        if i > 0 and abs(J_history[-1] - J_history[-2]) < 1e-8:
            break
        gradient = np.dot(X.T, (np.dot(X, theta) - y))
        theta -= alpha_per_example * gradient
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    start_theta = np.random.rand(X_train.shape[1])
    for learning_rate in alphas:
        theta, _ = efficient_gradient_descent(X_train, y_train, start_theta, learning_rate, iterations)
        alpha_dict[learning_rate] = compute_cost(X_val, y_val, theta)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    #####c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    feature_indices = list(range(X_train.shape[1]))

    for _ in range(5):
        best_feature = None
        best_cost = float('inf')
        for feature in feature_indices:
            if feature not in selected_features:
                X_train_temp = apply_bias_trick(X_train[:, selected_features + [feature]])
                X_val_temp = apply_bias_trick(X_val[:, selected_features + [feature]])
                theta = np.zeros(X_train_temp.shape[1])
                theta, _ = efficient_gradient_descent(X_train_temp, y_train, theta, best_alpha, iterations)
                cost = compute_cost(X_val_temp, y_val, theta)
                
                if cost < best_cost:
                    best_cost = cost
                    best_feature = feature

        if best_feature is not None:
            selected_features.append(best_feature)
        else:
            break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    # Store new features in a temporary dictionary before adding them to df_poly
    new_features = {}
    
    for feature_name in df.columns:
        new_features[f'{feature_name}^2'] = df[feature_name] ** 2
    
    for i, feature_name1 in enumerate(df.columns):
        for feature_name2 in df.columns[i + 1:]:
            new_features[f'{feature_name1}*{feature_name2}'] = df[feature_name1] * df[feature_name2]
    
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df_poly = pd.concat([df_poly, new_features_df], axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly