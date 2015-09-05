'''
This code is copyright Harrison Kinsley.

The open-source code is released under a BSD license:

Copyright (c) 2013, Harrison Kinsley
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the  nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL  BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import os
import urllib2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick, plot_day_summary_ohlc
import matplotlib
import pylab
matplotlib.rcParams.update({'font.size': 9})

from stock_data import run_analysis, gen_probs, annotate_chart

import Quandl
from Quandl import DatasetNotFound
from datetime import date, timedelta


def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def ExpMovingAverage(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def computeMACD(x, slow=26, fast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = ExpMovingAverage(x, slow)
    emafast = ExpMovingAverage(x, fast)
    return emaslow, emafast, emafast - emaslow

def getSourceData(provider, stock):
    if provider not in ('quandl', 'yahoo'):
        raise Exception('Invalid provider specified: %s', provider)
    return globals()["getSourceData_{}".format(provider)](stock)

def getSourceData_yahoo(stock):
    try:
    # Pull stock data from Yahoo
        print 'Currently Pulling',stock
        print str(datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
        #Keep in mind this is close high low open data from Yahoo
        urlToVisit='http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=6m/csv'
        stockFile =[]
        try:
            sourceCode = urllib2.urlopen(urlToVisit).read()
            splitSource = sourceCode.split('\n')
            for eachLine in splitSource:
                splitLine = eachLine.split(',')
                if len(splitLine)==6:
                    if 'values' not in eachLine:
                        stockFile.append(eachLine)
            print "STOCKFILE = %s" % stockFile
            return stockFile
        except Exception, e:
            print str(e), 'failed to organize pulled data.'
            raise e
    except Exception,e:
        print str(e), 'failed to pull pricing data'
        raise e

def getSourceData_quandl(symbol):
    print 'Currently Pulling %s' % symbol
    database = symbol.split('/')[0]
    if database not in ('GOOG', 'YAHOO', 'CME'):
        raise BadTickerException({'reason': 'Provided database %s is not supported' % database})
    DATABASE_ATTRS = {
            """
            'DATABASE' : {
                'common name': 'db specific name'
            }
            """
            'GOOG' : {
                'Close'  : 'Close',
                'High'   : 'High',
                'Low'    : 'Low',
                'Open'   : 'Open',
                'Volume' : 'Volume'
            },
            'YAHOO' : {
                'Close'  : 'Close',
                'High'   : 'High',
                'Low'    : 'Low',
                'Open'   : 'Open',
                'Volume' : 'Volume'
            },
            'CME' : {
                'Close'  : 'Settle',
                'High'   : 'High',
                'Low'    : 'Low',
                'Open'   : 'Open',
                'Volume' : 'Volume'
            }
        }
    today = date.today()
    #start = today - timedelta(days=365)
    start = today - timedelta(days=730)
    try:
        data = Quandl.get(symbol, collapse="daily",
            trim_start=start.strftime('%Y-%m-%d'), authtoken="jnizAYP-5ScP5u_qd4uk")
    except DatasetNotFound:
        return []
    stockData = []
    for index, row in data.iterrows():
        currDate = index.strftime('%Y%m%d')
        stockData.append('{},{},{},{},{},{}'.format(
            currDate,
            row[DATABASE_ATTRS[database]['Close']],
            row[DATABASE_ATTRS[database]['High']],
            row[DATABASE_ATTRS[database]['Low']],
            row[DATABASE_ATTRS[database]['Open']],
            row[DATABASE_ATTRS[database]['Volume']]
        ))
    return stockData

class BadTickerException(Exception):
    pass

def graphData(stock):
    '''
    Use this to dynamically pull a stock:
    '''
    desiredDays=90
    provider='quandl'
    try:
        stockFile = getSourceData(provider, stock)
    except BadTickerException as bte:
        exDict = bte.message.copy()
        exDict.update({'provider': provider})
        raise BadTickerException(exDict)

    if stockFile == []:
       raise BadTickerException({'provider': provider})

    # Process raw CSV with numpy
    date, closep, highp, lowp, openp, volume = np.loadtxt(stockFile,delimiter=',', unpack=True,
                                                          converters={ 0: mdates.strpdate2num('%Y%m%d')})
    y = len(date)
    newAr = []
    # Reformat data into list
    while len(newAr) < y:
        x = len(newAr)
        appendLine = date[x],openp[x],highp[x],lowp[x],closep[x],volume[x]
        newAr.append(appendLine)

    fig = plt.figure(facecolor='#07000d', figsize=(12,10))

    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
    plot_day_summary_ohlc(ax1, newAr[-desiredDays:], ticksize=3, colorup='#53c156', colordown='#53c156')
    visibleYlow, visibleYhigh = ylim()

    run_analysis(ax1.lines)

    ax1.grid(True, color='w')
    #ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.yaxis.label.set_color("w")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.spines['bottom'].set_color("#5998ff")
    ax1.spines['top'].set_color("#5998ff")
    ax1.spines['left'].set_color("#5998ff")
    ax1.spines['right'].set_color("#5998ff")
    ax1.tick_params(axis='y', colors='w')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    plt.gca().yaxis.set_label_position("right")
    #ax1.tick_params(axis='x', colors='w')
    plt.ylabel('Price')

    volumeMin = 0

    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

    # Probability Chart
    ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    #long_prob, short_prob, neutral_prob = gen_probs(ax1.lines, 0)
    # close price is the 5th element in newAr
    closePrices = [info[4] for info in newAr ]
    # date is first
    dates = [info[0] for info in newAr ]
    long_prob, short_prob, neutral_prob = gen_probs(closePrices, dates, len(ax1.lines)/3)
    highPrices= [info[2] for info in newAr ]
    lowPrices = [info[3] for info in newAr ]
    annotate_chart(ax1, date, lowPrices, highPrices, long_prob, short_prob, visibleYhigh - visibleYlow)
    ax2.plot(date[-len(long_prob):], long_prob, 'b-')
    ax2.plot(date[-len(short_prob):], short_prob, 'r-')
    ax2.plot(date[-len(neutral_prob):], neutral_prob, 'y-')
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    ax2.spines['bottom'].set_color("#5998ff")
    ax2.spines['top'].set_color("#5998ff")
    ax2.spines['left'].set_color("#5998ff")
    ax2.spines['right'].set_color("#5998ff")
    ax2.tick_params(axis='y', colors='w')
    ax2.tick_params(axis='y', colors='w')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    plt.gca().yaxis.set_label_position("right")
    plt.ylabel('Probability', color='w')
    ax2.tick_params(axis='x', colors='w')
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(45)

    plt.suptitle(stock.upper(),color='w')

    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=True)
    """
    ax1.annotate('aaa', xy=(date[-1], 35),
        xycoords='data', fontsize=14, color='w',
        xytext=(date[-1], 36), textcoords='data',
        arrowprops=dict(facecolor='white'))
    """
    #'''ax1.annotate('Big news!',(date[510],Av1[510]),
    #    xytext=(0.8, 0.9), textcoords='axes fraction',
    #    arrowprops=dict(facecolor='white', shrink=0.05),
    #    fontsize=14, color = 'w',
    #    horizontalalignment='right', verticalalignment='bottom')'''

    plt.tight_layout()
    #plt.subplots_adjust(left=.09, bottom=.14, right=.94, top=.95, wspace=.20, hspace=0)
    plt.show()
    fig.savefig(
      os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'static/chart.png'
      ),facecolor=fig.get_facecolor())

    plt.close('all')


