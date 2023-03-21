# DeepCrypto
This project uses a Long Short-Term Memory (LSTM) neural network to predict the price of Bitcoin in US dollars. The LSTM model is implemented using the Keras library in Python.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/571631/226732799-c76fa7c1-bc2c-4192-8bb8-7005770d8dd1.png">
</p>

## Dataset
The daily historical price data for Bitcoin in US dollars is retrieved from the CryptoCompare API using the `CryptoCompare` class defined in the `data_source/crypto_compare.py` module. The data is split into training and test sets using a specified percentage. The training set is used to train the LSTM model, while the test set is used to evaluate the performance of the model.

The data is stored as CSV files in the `datasets` folder.

## Model Architecture
The LSTM model has several layers with dropout regularization to prevent overfitting. The model is trained using the mean squared error loss function and the Adam optimizer. The LSTM model is designed to take in a sequence of 60 historical Bitcoin prices and predict the next price. The predicted price is then used as input for the next prediction, and so on.

The LSTM model is defined in the `BtcLtsm` class in the `predictors/btc_ltsm.py` module. The class has several methods for updating the dataset, training the model, loading a pre-trained model, and testing the model on the test data set.

## Usage
To use the project, you will need to install the necessary dependencies using `pip`. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

Once the dependencies are installed, you can use the project to predict Bitcoin prices by running the `deep_crypto.py` script with the appropriate command-line arguments. The available arguments are:

* `--update`: Update the dataset with the latest Bitcoin price data from the CryptoCompare API.
* `--train`: Train the LSTM model using the updated data set.
* `--test`: Test the LSTM model on the test data set and visualize the results in btc_price_prediction.png.
For example, to update the dataset, train and test the LSTM model, you can run the following command:

```bash
python deep_crypto.py --update --train --test
```

## License
This project is licensed under the MIT License. See the LICENSE file for more information.
