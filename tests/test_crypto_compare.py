import unittest
import sys
sys.path.append('..')
import data_source.crypto_compare as cc

class TestCryptoCompare(unittest.TestCase):
    def test_get_daily_history(self):
        cc_api = cc.CryptoCompare()
        daily_history = cc_api.get_daily_history('BTC', 'USDT', limit=5)
        self.assertNotEqual(daily_history, None)
        self.assertEqual(daily_history['Response'], 'Success')
        cc_api.save_price_history_to_csv(daily_history)
        
if __name__ == '__main__':
    unittest.main()