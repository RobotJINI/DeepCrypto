import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_source.crypto_compare
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import logging
import os

# Initial ltsm code building off of

def import_tensorflow():
    # Filter tensorflow version warnings
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    
    tf.get_logger().setLevel(logging.ERROR)
    
    return tf
tf = import_tensorflow()  

class StockProphet:
    def __init__(self):
        pass
    
    def load_training(self, file_name):
        data_train = pd.read_csv(file_name)
        train_set = data_train.iloc[:, 1:2].values
        
        self._sc = MinMaxScaler(feature_range = (0, 1))
        train_set = self._sc.fit_transform(train_set)
        logging.debug(f'training set:\n{train_set}')
        
        # Creating a data structure with 60 timesteps and 1 output
        history = 60
        self._features_train = []
        self._results_train = []
        for i in range(history, len(train_set)):
            self._features_train.append(train_set[i-history:i, 0])
            self._results_train.append(train_set[i, 0])
        self._features_train, self._results_train = np.array(self._features_train), np.array(self._results_train)
        
        # Reshaping
        self._features_train = np.reshape(self._features_train, (self._features_train.shape[0], self._features_train.shape[1], 1))
        
    def create_rnn(self, model_name):
        # Initialising the RNN
        self._regressor = Sequential()
        
        # Adding the first LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (self._features_train.shape[1], 1)))
        self._regressor.add(Dropout(0.2))
        
        # Adding a second LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = 50, return_sequences = True))
        self._regressor.add(Dropout(0.2))
        
        # Adding a third LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = 50, return_sequences = True))
        self._regressor.add(Dropout(0.2))
        
        # Adding a fourth LSTM layer and some Dropout regularisation
        self._regressor.add(LSTM(units = 50))
        self._regressor.add(Dropout(0.2))
        
        # Adding the output layer
        self._regressor.add(Dense(units = 1))
        
        # Compiling the RNN
        self._regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # Fitting the RNN to the Training set
        self._regressor.fit(self._features_train, self._results_train, epochs = 100, batch_size = 32)
        self._regressor.save(model_name)
        
    def load(self, model_name):
        self._regressor = load_model(model_name)
        
    def test_model(self, train_file_name, test_file_name):
        # Getting the real stock price of 2017
        dataset_test = pd.read_csv(test_file_name)
        real_stock_price = dataset_test.iloc[:, 1:2].values
        
        dataset_train = pd.read_csv(train_file_name)
        train_set = dataset_train.iloc[:, 1:2].values
        
        self._sc = MinMaxScaler(feature_range = (0, 1))
        train_set = self._sc.fit_transform(train_set)
        
        # Getting the predicted stock price of 2017
        dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = self._sc.transform(inputs)
        features_test = []
        for i in range(60, 80):
            features_test.append(inputs[i-60:i, 0])
        features_test = np.array(features_test)
        features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))
        predicted_stock_price = self._regressor.predict(features_test)
        predicted_stock_price = self._sc.inverse_transform(predicted_stock_price)
        
        # Visualising the results
        plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
        plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.show()



if __name__ == "__main__":
    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)
    
    model_name = './predictors/saved/google_stock.h5'
    train_file_name = './datasets/Google_Stock_Price_Train.csv'
    test_file_name = './datasets/Google_Stock_Price_Test.csv'
    
    sp = StockProphet()
    
    #sp.load_training(train_file_name)
    #sp.create_rnn(model_name)
    
    sp.load(model_name)
    sp.test_model(train_file_name, test_file_name)
