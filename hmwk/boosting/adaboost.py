import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import matplotlib.pylab as plt


class ThreesAndEights:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set
        X_train, y_train, X_valid, y_valid = pickle.load(f)

        # Extract only 3's and 8's for training set
        self.X_train = X_train[np.logical_or(y_train == 3, y_train == 8), :]
        self.y_train = y_train[np.logical_or(y_train == 3, y_train == 8)]
        self.y_train = np.array([1 if y == 8 else -1 for y in self.y_train])

        # Shuffle the training data
        shuff = np.arange(self.X_train.shape[0])
        np.random.shuffle(shuff)
        self.X_train = self.X_train[shuff, :]
        self.y_train = self.y_train[shuff]

        # Extract only 3's and 8's for validation set
        self.X_valid = X_valid[np.logical_or(y_valid == 3, y_valid == 8), :]
        self.y_valid = y_valid[np.logical_or(y_valid == 3, y_valid == 8)]
        self.y_valid = np.array([1 if y == 8 else -1 for y in self.y_valid])

        f.close()

# data = ThreesAndEights("/home/sayali/Documents/CSCI5622-Machine-Learning/hmwk/data/mnist21x21_3789.pklz")


class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=1), random_state=1234):
        """
        Create a new adaboost classifier.

        Args:
            N (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner
            random_state (int, optional): set random generator.  needed for unit testing.

        Attributes:
            base (estimator): Your general weak learner
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners.
            learners (list): List of weak learner instances.
        """

        np.random.seed(random_state)

        self.n_learners = n_learners
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []

    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners.

        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data
            y_train (ndarray): [n_samples] ndarray of data
        """

        # TODO

        # Note: You can create and train a new instantiation
        # of your sklearn decision tree as follows

        w = np.array([1.0 / float(len(y_train)) for i in range(len(y_train))])
        output = 0

        for i in range(self.n_learners):
            h = clone(self.base)
            h.fit(X_train, y_train, sample_weight=w)
            yhat = h.predict(X_train)
            err = 0
            misclass_ex = []
            for j in range(len(X_train)):
                if (y_train[j] != yhat[j]):
                    err = err + w[j]
                    misclass_ex.append(j)
            err = err / w.sum()
            self.alpha[i] = 0.5 * np.log((1 - err) / float(err))
            for me in range(len(X_train)):
                w[me] = w[me] * (np.exp(-1 * self.alpha[i] * y_train[me] * yhat[me])) #/ w.sum()
            self.learners.append(h)
            output = output + self.alpha[i] * yhat[i]
        if output > 0:
            return 1
        else:
            return -1

    def predict(self, X):
        """
        Adaboost prediction for new data X.

        Args:
            X (ndarray): [n_samples x n_features] ndarray of data

        Returns:
            yhat (ndarray): [n_samples] ndarray of predicted labels {-1,1}
        """

        # TODO
        y = []
        temp = []
        for k in range(len(self.learners)):
            temp.append(self.learners[k].predict(X))

        return y


    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.

        Args:
            X (ndarray): [n_samples x n_features] ndarray of data
            y (ndarray): [n_samples] ndarray of true labels

        Returns:
            Prediction accuracy (between 0.0 and 1.0).
        """

        # TODO

        return 0.0

    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting
        for monitoring purposes, such as to determine the score on a
        test set after each boost.

        Args:
            X (ndarray): [n_samples x n_features] ndarray of data
            y (ndarray): [n_samples] ndarray of true labels

        Returns:
            scores (ndarary): [n_learners] ndarray of scores
        """

        # TODO

        return np.zeros(self.n_learners)

    def staged_margin(self, x, y):
        """
        Computes the staged margin after each iteration of boosting
        for a single training example x and true label y

        Args:
            x (ndarray): [n_features] ndarray of data
            y (integer): an integer {-1,1} representing the true label of x

        Returns:
            margins (ndarary): [n_learners] ndarray of margins
        """

        # TODO

        margins = np.zeros(self.n_learners)

        return margins


X_train = np.array([[6,9.5],[4,8.5],[9,8.75],[8,8.0],[3,7],[1,6.5],[5,6.5],[1.5,2.5],[2,1],[9,2]])
y_train = np.array([1,1,-1,1,-1,1,-1,1,-1,-1])
clf = AdaBoost(n_learners=3)
clf.fit(X_train, y_train)
alphas = clf.alpha
print(alphas)
pred = clf.predict(X_train)