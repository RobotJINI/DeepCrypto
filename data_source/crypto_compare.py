import requests
import os
import urllib.parse
 
class CryptoCompare:
    def __init__(self, env_name='CRYPTO_COMPARE_API_KEY'):
        self._api_key = os.environ.get(env_name)
        
    def get_daily_history(self, base, quote, limit=2000, last_time=None):
        base_url = 'https://min-api.cryptocompare.com/data/v2/histoday'
        params = {
            'fsym': base,
            'tsym': quote,
            'limit': limit
        }
        if last_time is not None:
            params['toTs'] = last_time
            
        params['api_key'] = self._api_key
            
        url = base_url + "?" + urllib.parse.urlencode(params)
        
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            print("Error: ", response.status_code)
            return None

    def save_price_history_to_csv(self, history):
        print(history)
        for data_point in history['Data']['Data']:
            print(data_point)
