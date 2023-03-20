

class OHLCV:
    def __init__(self, time, open, high, low, close, volume):
        self.time = time
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        
    def list(self):
        return [self.time, self.open, self.high, self.low, self.close, self.volume]
    
    def to_str(self):
        return f'time: {self.time}, open: {self.open}, high: {self.high}, low: {self.low}, close: {self.close}, volume: {self.volume}'
        