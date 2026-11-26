import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Perform linear regression using gradient descent.

    Args:
        X: Feature matrix of shape (m, n) where first column is all ones (for intercept)
        y: Target vector of shape (m,)
        alpha: Learning rate
        iterations: Number of gradient descent iterations
    
    Returns:
        Learned weights as a 1D array of shape (n,)
    """
    m, n = X.shape
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    theta = np.zeros((n, 1))  # Initialize weights to zeros

    # Your code here: implement gradient descent

    for i in range(iterations):
        y_hat = np.dot(X, theta) 

        error = y-y_hat

        gradients_wrt_weights = (-1/m) * np.dot(X.T,error)

        theta -= alpha * gradients_wrt_weights 

    return theta.flatten()

'''
class LinearRegression:
    def __init__(self):
        self.bias = 0
        self.loss = []
        
    def fit(self,X, y, alpha = 0.05, n_iterations = 100):
        # Get num observations and num features
        self.n, self.m = X.shape
        
        self.weight = np.ones(self.m)
        
        for i in range(n_iterations):
            
            y_hat = self.predict(X)
            
            error = y-y_hat
            
            mse = np.square(error).mean()
            
            self.loss.append(mse)
            
            gradient_wrt_weight = (-1/self.n) * np.dot(X.T,error)
            gradient_wrt_bias = (-1/self.n) * np.sum(error)
            
            self.weight -= gradient_wrt_weight
            self.bias -= gradient_wrt_bias
            
    def predict(self,X):
        return np.dot(X,self.weight) + self.bias
        
model = LinearRegression()

model.fit(X,y,alpha = 0.05, n_iterations = 100)

predictions = model.predict(X)
'''