'''Apple stock trader using LSTM model implemented with Tensorflow
    Trading Frequency: 1 day
    Data API: Yahoo Finance
    Trading API: Alpaca trade API
    
    @author: Pedro Lourenco
'''


import tensorflow as tf
import yfinance as yf
from datetime import datetime, timedelta
from pickle import load
from time import sleep
import numpy as np

from alpaca_trade_api.rest import REST
import os


class AlpacaApi:
    def __init__(self, apiKey, secretKey, demoMode = True):
        self.apiKey = apiKey
        self.secretKey = secretKey
        self.demoMode = demoMode

        self.apiInstance = REST(key_id = self.apiKey, 
                                 secret_key = self.secretKey, 
                                 base_url = 'https://paper-api.alpaca.markets' if self.demoMode else 'https://api.alpaca.markets')


    def getOpenPositions(self):
        portfolio = self.apiInstance.list_positions()
        return [[position.qty, position.symbol, position.side] for position in portfolio]

    #place order
    def openPosition(self, ticker, shareNumber, isBuy):
        self.apiInstance.submit_order(symbol = ticker, 
                                    qty = shareNumber, 
                                    side = 'buy' if isBuy else 'sell', 
                                    type = 'market')
            
    def closePosition(self, ticker):
        self.apiInstance.close_position(ticker)


if __name__ == '__main__':
    #create instance of alpaca api
    secretKey =  os.environ.get('SECRET_KEY')
    alpacaApiKey = os.environ.get('ALPACA_KEY')
    tradeAPI = AlpacaApi(alpacaApiKey, secretKey)
    
    
    # required difference between predicted price and current price to trigger trade
    triggerMargin = 5 
    
    # load data scalers
    with open('./Scalers/x_scaler', 'rb') as inputFile:
        xScaler = load(inputFile)
        
    with open('./Scalers/y_scaler', 'rb') as inputFile:
        yScaler = load(inputFile)
        

    #record weather one already possesses any AAPL shares 
    hasPosition = False
    positionSide = None
    
    
    sleepTime = 8 * 60 * 60
    
    while True:
        #extract stock data from the last 50 days (one chunk of data)
        currentDate = datetime.now()
        startDate = currentDate - timedelta(days = 70) #some days may not have any data so 70 days is used to ensure at least 50 data points are fetched

        startDateString = startDate.strftime('%Y-%m-%d')
        endDateString = currentDate.strftime('%Y-%m-%d')
        
        stockData = yf.download('AAPL', start = startDateString, end = endDateString)[-50:]
        
        highs = stockData['High'] 
        lows = stockData['Low'] 
        closes = stockData['Close']

        #merge the three arrays into a single array
        #each element of the array will contain the high, low and close for each date
        lastChunk = list(map(list, list(zip(highs, lows, closes))))
        lastChunk = xScaler.transform(lastChunk)
        lastChunk = np.stack(lastChunk) #conver to 2d numpy array
        
        # load trained model and predict next price 
        model = tf.keras.models.load_model('./model')
        predictedPrice = model.predict(lastChunk.reshape(1, *lastChunk.shape))
        predictedPrice = yScaler.inverse_transform(predictedPrice)[0][0] #convert back to original range
        
        currentPrice = closes[-1]
        print(f'Prediction: {predictedPrice}')
        print(f'Current price: {currentPrice}')
        
        if predictedPrice - currentPrice > triggerMargin:
            if not hasPosition:
                print(f'Buying 5 shares of AAPL')
                tradeAPI.openPosition('AAPL', 5, True) #buy 5 shares
                hasPosition = True
                positionSide = 'buy'
            else:
                if positionSide == 'sell':
                    print('Closing position')
                    tradeAPI.closePosition('AAPL') 
                    hasPosition = False 
            
        if currentPrice - predictedPrice > triggerMargin:
            if not hasPosition:
                print(f'Selling 5 shares of AAPL')
                tradeAPI.openPosition('AAPL', 5, False) #sell 5 shares
                hasPosition = True
                positionSide = 'sell'
            else:
                if positionSide == 'buy':
                    print('Selling 5 shares of AAPL')
                    tradeAPI.closePosition('AAPL')
                    hasPosition = False
        
        print(f'Sleeping for {sleepTime} seconds')
        sleep(sleepTime)