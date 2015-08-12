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
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.finance import candlestick, plot_day_summary_ohlc
import matplotlib
import pylab
matplotlib.rcParams.update({'font.size': 9})

from stock_data import get_data, run_analysis


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

def getSourceData(stock):
    try:
    # Pull stock data from Yahoo
        print 'Currently Pulling',stock
        print str(datetime.datetime.fromtimestamp(int(time.time())).strftime('%Y-%m-%d %H:%M:%S'))
        #Keep in mind this is close high low open data from Yahoo
        urlToVisit = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=2y/csv'
        stockFile =[]
        try:
            sourceCode = urllib2.urlopen(urlToVisit).read()
            splitSource = sourceCode.split('\n')
            for eachLine in splitSource:
                splitLine = eachLine.split(',')
                if len(splitLine)==6:
                    if 'values' not in eachLine:
                        stockFile.append(eachLine)
            return stockFile
        except Exception, e:
            print str(e), 'failed to organize pulled data.'
            raise e
    except Exception,e:
        print str(e), 'failed to pull pricing data'
        raise e
    
class BadTickerException(Exception):
    pass

def graphData(stock):
    '''
        Use this to dynamically pull a stock:
    '''
    stockFile = getSourceData(stock)
    
    if stockFile == []:
       raise BadTickerException({'ticker': stock})

    #stockFile = get_data(stock)

    # Process raw CSV with numpy
    date, closep, highp, lowp, openp, volume = np.loadtxt(stockFile,delimiter=',', unpack=True,
                                                          converters={ 0: mdates.strpdate2num('%Y%m%d')})
    x = 0
    y = len(date)
    newAr = []
    # Reformat data into list
    while x < y:
        #appendLine = date[x],openp[x],closep[x],highp[x],lowp[x],volume[x]
        appendLine = date[x],openp[x],highp[x],lowp[x],closep[x],volume[x]
        newAr.append(appendLine)
        x+=1
    
    #Av1 = movingaverage(closep, MA1)
    #Av2 = movingaverage(closep, MA2)

    #SP = len(date[MA2-1:])
    SP = len(newAr)
        
    fig = plt.figure(facecolor='#07000d', figsize=(12,10))

    ax1 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4, axisbg='#07000d')
    #candlestick(ax1, newAr[-SP:], width=.6, colorup='#53c156', colordown='#ff1717')
    #plot_day_summary_ohlc(ax1, newAr[-SP:], ticksize=3, colorup='#53c156', colordown='#ff1717')
    plot_day_summary_ohlc(ax1, newAr[-SP:], ticksize=3, colorup='#53c156', colordown='#53c156')

    #Label1 = str(MA1)+' SMA'
    #Label2 = str(MA2)+' SMA'

    #ax1.plot(date[-SP:],Av1[-SP:],'#e1edf9',label=Label1, linewidth=1.5)
    #ax1.plot(date[-SP:],Av2[-SP:],'#4ee6fd',label=Label2, linewidth=1.5)
    
    #import pdb;pdb.set_trace()    
    """
    i = 0    
    for lineIndex in range(0,len(ax1.lines),3):
        low_high = ax1.lines[lineIndex]
        theOpen = ax1.lines[lineIndex+1]
        theClose = ax1.lines[lineIndex+2]
        theCloseValue = theClose.get_data()[1][0]
        if 200 < i < 400:
            [line.set_c('red') for line in [low_high, theOpen, theClose]]
        i = i + 1
    """
    run_analysis(ax1.lines)

    ax1.grid(True, color='w')
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
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
    ax1.tick_params(axis='x', colors='w')
    plt.ylabel('Price')

    #maLeg = plt.legend(loc=9, ncol=2, prop={'size':7},
    #           fancybox=True, borderaxespad=0.)
    #maLeg.get_frame().set_alpha(0.4)
    #textEd = pylab.gca().get_legend().get_texts()
    #pylab.setp(textEd[0:5], color = 'w')

    volumeMin = 0
    
    #ax0 = plt.subplot2grid((6,4), (0,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')
    #rsi = rsiFunc(closep)
    #rsiCol = '#c1f9f7'
    #posCol = '#386d13'
    #negCol = '#8f2020'
    
    #ax0.plot(date[-SP:], rsi[-SP:], rsiCol, linewidth=1.5)
    #ax0.axhline(70, color=negCol)
    #ax0.axhline(30, color=posCol)
    #ax0.fill_between(date[-SP:], rsi[-SP:], 70, where=(rsi[-SP:]>=70), facecolor=negCol, edgecolor=negCol, alpha=0.5)
    #ax0.fill_between(date[-SP:], rsi[-SP:], 30, where=(rsi[-SP:]<=30), facecolor=posCol, edgecolor=posCol, alpha=0.5)
    #ax0.set_yticks([30,70])
    #ax0.yaxis.label.set_color("w")
    #ax0.spines['bottom'].set_color("#5998ff")
    #ax0.spines['top'].set_color("#5998ff")
    #ax0.spines['left'].set_color("#5998ff")
    #ax0.spines['right'].set_color("#5998ff")
    #ax0.tick_params(axis='y', colors='w')
    #ax0.tick_params(axis='x', colors='w')
    #plt.ylabel('RSI')

    #ax1v = ax1.twinx()
    #ax1v.fill_between(date[-SP:],volumeMin, volume[-SP:], facecolor='#00ffe8', alpha=.4)
    #ax1v.axes.yaxis.tick_right()
    #ax1v.axes.yaxis.set_label_position("right")
    #ax1v.axes.yaxis.set_ticklabels([])
    #ax1v.grid(False)
    #ax1v.set_ylim(0, 3*volume.max())
    #ax1v.spines['bottom'].set_color("#5998ff")
    #ax1v.spines['top'].set_color("#5998ff")
    #ax1v.spines['left'].set_color("#5998ff")
    #ax1v.spines['right'].set_color("#5998ff")
    #ax1v.tick_params(axis='x', colors='w')
    #ax1v.tick_params(axis='y', colors='w', labelright="on")

    
    #ax2 = plt.subplot2grid((6,4), (5,0), sharex=ax1, rowspan=1, colspan=4, axisbg='#07000d')

    

    # START NEW INDICATOR CODE #

    

    # END NEW INDICATOR CODE #

    

    
    plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
    #ax2.spines['bottom'].set_color("#5998ff")
    #ax2.spines['top'].set_color("#5998ff")
    #ax2.spines['left'].set_color("#5998ff")
    #ax2.spines['right'].set_color("#5998ff")
    #ax2.tick_params(axis='x', colors='w')
    #ax2.tick_params(axis='y', colors='w')
    #ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))




    
    for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(45)

    plt.suptitle(stock.upper(),color='w')

    #plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=True)
    
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
       

