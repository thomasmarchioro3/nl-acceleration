import numpy as np

_SONAR = 'dataset/sonar/sonar.csv'

class Loader:

    def __init__(self, dataset='sonar', tau=1e-2):
        self.loaded = False
        if(dataset == 'sonar'):
            data = np.genfromtxt(_SONAR, delimiter=',')
            self.X = data[:,:-1] # delete the last column
            label = np.genfromtxt(_SONAR, dtype='S1', usecols=self.X.shape[1], delimiter=',')
            self.y = (label != label[0]).astype(int) # make labels either zeros or ones
            self.loaded = True
            self.tau = tau
        return

    def get_function(self):
        assert self.loaded

        X = self.X
        y = self.y

        # get f lambda function
        f = lambda w : np.sum(np.log(1+np.exp(-np.matmul(np.diag(y), np.matmul(X, w)))))+0.5*self.tau*np.dot(w,w)

        # get grad_f lambda function
        h = lambda w: np.exp(-np.matmul(np.diag(y), np.matmul(X, w)))
        grad_num = lambda w : np.matmul(np.diag(h(w)), np.matmul(np.diag(y), X))
        grad_den = lambda w : np.linalg.inv(np.eye(y.shape[0]) + np.diag(h(w)))

        grad_f = lambda w : -np.sum(np.matmul(grad_den(w), grad_num(w)), 0)+self.tau*w

        input_size = X.shape[1]

        return f, grad_f, input_size

    def get_constants(self):
        assert self.loaded

        X = self.X
        y = self.y
        sigma = self.tau
        L = 0.25*(np.linalg.norm(X, ord='fro')**2)+self.tau

        return sigma, L
