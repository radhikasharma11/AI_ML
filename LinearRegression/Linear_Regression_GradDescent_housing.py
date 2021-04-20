import numpy as np
import pandas as pd
import tqdm


class LinearRegression:

    def __init__(self):
        self.no_of_iterations = 100
        self.learning_rate = 0.01
        self.coeffs = []
        self.loss_history = []

    def train(self, X, y):
        initial_cost = self.compute_cost(X,y)
        self.loss_history.append(initial_cost)
        for t in range(self.no_of_iterations):
            # optimization algorithm : gradient descent
            gradient = np.dot(X.T,(X.dot(self.coeffs)-y))/ len(y)
            self.coeffs -= self.learning_rate * gradient
            loss = self.compute_cost(X, y)
            self.loss_history.append(loss)
        print("Optimised Coeffs. are: ", self.coeffs)

    def fit(self, X, y, learning_rate=0.1, iterations=100):
        self.no_of_iterations = iterations
        self.learning_rate = learning_rate
        self.coeffs = np.zeros(X.shape[1])
        self.train(X,y)

    def predict(self, X):
        return np.dot(X, self.coeffs)

    def compute_cost(self, X, y):
        # Mean squared error aka MSE
        return (np.sum((X.dot(self.coeffs))- y)** 2)/ 2*len(y)

    def score(self, X, y):
        # r2 coefficient
        predicted_y = self.predict(X)
        se_mean = np.sum((y - y.mean()) ** 2)
        se = np.sum((y - predicted_y) ** 2)
        r2_coeff = 1 - se/se_mean
        return r2_coeff


def datapreprosessing(data, train=True, val_split = 0.8, seed = 123):
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data.reset_index(drop=True)
    print("Dataset:", data.shape)
    if train:
        # remove id column
        data = data.iloc[:, 1:]

        shuffle_data = data.sample(frac=1)
        size = int(val_split * len(shuffle_data))
        X_train, y_train, X_val, y_val = shuffle_data.iloc[:size, :-1], shuffle_data.iloc[:size, -1], \
                                           shuffle_data.iloc[size:, :-1], shuffle_data.iloc[size:, -1]

        # standardize data
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_val = (X_val - X_val.mean()) / X_val.std()

        X_train = np.c_[np.ones(len(X_train), dtype='int64'), X_train.to_numpy()]  # append 1s to the features in train set
        X_val = np.c_[np.ones(len(X_val), dtype='int64'), X_val.to_numpy()]  # append 1s to the features in validation set

        print("Training size: ", X_train.shape[0])
        print("Validation size: ", X_val.shape[0])
        print("No. of features: train ", X_train.shape[1])
        print("No. of features: test ", X_train.shape[1])
        return X_train, y_train, X_val, y_val
    else:
        data = data.iloc[:, 1:]
        normalized_df = (data - data.mean()) / data.std()
        X_test = np.c_[np.ones(len(normalized_df), dtype='int64'), normalized_df.to_numpy()]  # append 1s to the features in test set
        print("Testing size: ", X_test.shape[0])
        print("No. of features: ", X_test.shape[1])
        return X_test


if __name__ == '__main__':
    lin_reg = LinearRegression()
    # read boston data
    housing_data_train = pd.read_csv(r"C:\Users\Radhika\Desktop\datasets\demo-main\Linear Regression\data\train.csv")
    housing_data_test = pd.read_csv(r"C:\Users\Radhika\Desktop\datasets\demo-main\Linear Regression\data\test.csv")
    # choose the columns to be used for training and testing, here we discard columns which do not have numerical values
    columns = ['Id', 'LotFrontage', 'LotArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'MasVnrArea',
               '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
               'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
               'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'SalePrice']

    # remove extra columns from the train data
    print(f'Train Data Size: {housing_data_train.shape}')
    train_data = housing_data_train[columns]
    print(f'Test Data Size: {housing_data_test.shape}')
    test_data = housing_data_test[columns[:-1]]

    X_train, y_train, X_val, y_val = datapreprosessing(train_data)
    X_test = datapreprosessing(test_data, train=False)

    lin_reg.fit(X_train, y_train, 0.1, 100)
    print(lin_reg.score(X_train, y_train))
    print(lin_reg.score(X_val, y_val))






