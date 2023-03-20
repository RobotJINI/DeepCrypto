from predictors.btc_ltsm import BtcLtsm

if __name__ == "__main__": 
    btc_ltsm = BtcLtsm()
    #btc_ltsm.update_dataset()
    #btc_ltsm.train()
    btc_ltsm.load()
    btc_ltsm.test_model()