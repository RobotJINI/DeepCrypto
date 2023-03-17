import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging
import os

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


if __name__ == "__main__":
    logging.basicConfig(format='\n%(message)s\n', level=logging.DEBUG)
    
    sp = StockProphet()
    sp.load_training('./datasets/Google_Stock_Price_Train.csv')
