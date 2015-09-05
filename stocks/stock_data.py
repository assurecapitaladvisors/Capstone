from stock_data_const import PatternArray
from datetime import date, timedelta
from matplotlib.dates import num2date
from matplotlib.markers import MarkerStyle

from itertools import tee, izip

def run_analysis(data):
    """
    Takes an array of 2dPlot objects from matplotlib.  The data is formatted
    in triplets.
        0 - ((volume, volume), (low, high))
        1 - ((volume, volume), (open, open))
        2 - ((volume, volume), (close, close))
        "
    The buy and sell functions take the last 13 days worth of data, and run
    essentially the opposite algorithm against it.  They review the days, and
    if appropriate, color the bars red/blue.  The buy/sell funcs take 51 data
    points to accommodate for the weird data structure.

    This has only been tested with the axes of a ohlc chart as input.
    """
    ACTIONS =  {'buy': 'blue', 'sell': 'red'}
    def setup_action(list, action):
        for lineIndex in range(12,len(data)-42,3):
            checkSum = 0
            for innerLineIndex in range(lineIndex, lineIndex+42, 3):
                low_high = data[innerLineIndex]
                theOpen = data[innerLineIndex+1]
                theClose = data[innerLineIndex+2]
                theCloseValue = theClose.get_data()[1][0]
                #print "{} {} {}".format(checkSum, theCloseValue, data[innerLineIndex-10].get_data()[1][0])
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
                    #data[i].set_marker('^')
                    #data[i].set_markersize(5)
                    data[i].set_label('A')

    print 'Generating buy data'
    setup_action(data, 'buy')
    print 'Generating sell data'
    setup_action(data, 'sell')

def gen_probs(closePrices, dates, numOfDays):
    """
    data is an array of (date,openp,highp,lowp,closep,volume) tuples (2 years)
    numOfDays is the # of days I actually want to chart (back from today)
    """
    returnLongProbs = [0] * len(closePrices)
    returnShortProbs = [0] * len(closePrices)
    returnNeutralProbs = [0] * len(closePrices)
    myPattern = 0
    longProb = [0] * len(PatternArray)
    shortProb= [0] * len(PatternArray)
    neutralProb = [0] * len(PatternArray)
    countArray = [0] * len(PatternArray)
    # start 10 days in, and iterate over each day
    print 'Generating probability data'
    for currDay in xrange(5,len(closePrices)):
        pattern = ""
        for i in xrange(0,5):
            idx = currDay - i
            if closePrices[idx-1] > closePrices[idx]:
                pattern = "{}{}".format('1', pattern)
            elif closePrices[idx-1] < closePrices[idx]:
                pattern = "{}{}".format('-1', pattern)
            elif abs(closePrices[idx-1] - closePrices[idx]) < 1e-10:
                pattern = "{}{}".format('0', pattern)
            else:
                raise Exception('Bad comparison: %s & %s', closePrices[idx-1],
                        closePrices[idx])

        if closePrices[currDay] > closePrices[currDay-1]:
            longProb[myPattern] = longProb[myPattern] + 1
        elif closePrices[currDay] < closePrices[currDay-1]:
            shortProb[myPattern] = shortProb[myPattern] + 1
        elif abs(closePrices[currDay] - closePrices[currDay-1]) < 1e-10:
            neutralProb[myPattern] = neutralProb[myPattern] + 1
        else:
            raise Exception('Bad comparison: %s & %s', closePrices[currDay],
                    closePrices[currDay-1])

        for i in xrange(0, len(PatternArray)):
            if pattern == PatternArray[i]:
                countArray[i] = countArray[i] + 1
                myPattern = i
                break

        returnLongProbs[currDay] = float(longProb[myPattern]) / countArray[myPattern]
        returnShortProbs[currDay] = float(shortProb[myPattern]) / countArray[myPattern]
        returnNeutralProbs[currDay] = float(neutralProb[myPattern]) / countArray[myPattern]

        #print "CURRDAY = %s" % currDay
        #print "RETURNLONGPROB = %s" % returnLongProbs[currDay]
        #print "MYPATTERN = %s" % myPattern
        #print "LONGPROB = %s" % longProb
        #print "len(LONGPROB) = %s" % len(longProb)
        #print "DATE = %s | LONGPROB = %s" % (num2date(dates[currDay]), returnLongProbs[currDay],)

    return returnLongProbs[-numOfDays:], returnShortProbs[-numOfDays:],returnNeutralProbs[-numOfDays:]

def annotate_chart(ax, dates, lows, highs, long_prob, short_prob, yrange):
    """
    Annotate a given chart with probability indicators from the probability
    function above.
    ax - Axes object
    dates - array of dates to use for determining x axis location for
            annotation
    lows, highs - arrays of the high + low prices per day
    long_prob, short_prob - probabilities of going long or short, as returned
                            by the gen_probs function above
    yrange - (int) the range from high to low of the y axis of the chart
    """
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return izip(a, b)

    def mark(axis, date, price, color, priceOffset, textOffset):
        axis.annotate('', xy=(date, price+(priceOffset*.1*yrange)), xycoords='data', fontsize=14,
                color=color[0], xytext=(date, price+(textOffset*.05*yrange)), textcoords='data',
                arrowprops=dict(facecolor=color, frac=.18, width=3))

    def mark_long(axis, date, low):
        mark(axis, date, low, 'blue', -.1, -1)

    def mark_short(axis, date, high):
        mark(axis, date, high, 'red', .1, 1)

    L = 'L'
    S = 'S'
    higher = None
    for idx, (longp, shortp) in enumerate(izip(long_prob, short_prob)):
        if not higher: #base case
            if longp > shortp:
                higher = L
            elif shortp > longp:
                higher = S
            continue
        # 'higher' set at this point, so we know which probability was
        # higher on the previous day
        priceIdx = len(long_prob)-idx #get index from end
        if longp > shortp and higher is S:
            mark_long(ax, dates[-priceIdx], lows[-priceIdx])
            higher = L
        elif shortp > longp and higher is L:
            mark_short(ax, dates[-priceIdx], highs[-priceIdx])
            higher = S


