import Quandl
from datetime import date, timedelta

def get_data(symbol):
    today = date.today()
    yearsAgo = today - timedelta(days=365)
    #yearsAgo = today - timedelta(days=365*1)
    data = Quandl.get(symbol, collapse="daily", 
        trim_start=yearsAgo.strftime('%Y-%m-%d'), authtoken="jnizAYP-5ScP5u_qd4uk")
    stockData = []
    for index, row in data.iterrows():
        currDate = index.strftime('%Y%m%d')
        stockData.append('{},{},{},{},{},{}'.format(
            currDate,
            row['Last'],
            row['High'],
            row['Low'],
            row['Open'],
            row['Volume']
        ))
    return stockData

def run_analysis(data):
    """
    Takes an array of 2dPlot objects from matplotlib.  The data is formatted
    in triplets.
        0 - ((volume, volume), (low, high))
        1 - ((volume, volume), (open, open))
        2 - ((volume, volume), (close, close))
    The buy and sell functions take the last 13 days worth of data, and run
    essentially the opposite algorithm against it.  They review the days, and 
    if appropriate, color the bars red/blue.  The buy/sell funcs take 51 data 
    points to accommodate for the weird data structure
    """
    ACTIONS =  {'buy': 'blue', 'sell': 'red'}
    def setup_action(list, action):
        for lineIndex in range(12,len(data)-42,3):
        #for lineIndex in range(12,len(data)-12,3):
            checkSum = 0
            for innerLineIndex in range(lineIndex, lineIndex+42, 3):
                low_high = data[innerLineIndex]
                theOpen = data[innerLineIndex+1]
                theClose = data[innerLineIndex+2]
                theCloseValue = theClose.get_data()[1][0]
                print "{} {} {}".format(checkSum, theCloseValue, data[innerLineIndex-10].get_data()[1][0])
                #if theCloseValue < data[lineIndex-12].get_data()[1][0]:
                if action == 'buy':
                    if theCloseValue < data[innerLineIndex-10].get_data()[1][0]:
                        checkSum = checkSum + 1
                if action == 'sell':
                    if theCloseValue > data[innerLineIndex-10].get_data()[1][0]:
                        checkSum = checkSum + 1
            if checkSum >= 13:
                for i in range(innerLineIndex-42, innerLineIndex):
                    data[i].set_c(ACTIONS[action])
    def setup_sell():
        pass

    setup_action(data, 'buy')
    setup_action(data, 'sell')
