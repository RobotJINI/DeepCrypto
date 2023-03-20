import data_source.crypto_compare as cc
import models.csv_writer as csv_writer

class BtcLtsm:
    def __init__(self):
        self._data_source = cc.CryptoCompare()
        self._train_name_base = 'btc_price_train'
        self._test_name_base = 'btc_price_test'
    
    def update_dataset(self, percent_train=0.98, limit=2000):
        try:
            ohlcv_list = self._data_source.get_daily_history('BTC', 'USDT', limit=limit)
            
            test_start_idx = int(len(ohlcv_list) * percent_train)
            
            csv_writer.write_ohlcv(ohlcv_list[:test_start_idx], self._train_name_base)
            csv_writer.write_ohlcv(ohlcv_list[test_start_idx:], self._test_name_base)
            return True
        except Exception as e:
            # Catch all exceptions and print the error message
            print(f"An error occurred: {e}")
            return False
            