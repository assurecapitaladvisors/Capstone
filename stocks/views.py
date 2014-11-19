from flask import render_template

from stocks import app, graph
from stocks.graph import BadTickerException
from flask.ext.wtf import Form
from wtforms import StringField

from decorators import nocache

class TickerForm(Form):
    ticker = StringField('')

    def validate(self):
        return Form.validate(self)

@app.route("/", methods=["GET", "POST"])
def index():
    # Turn this into a field on the webpage for users to enter a ticker
    tickerIsBad = False
    form = TickerForm()
    if form.validate_on_submit():
        ticker = form.data['ticker']

        # ticker = "CHRIS/CME_CL1"
        if not ticker:
            tickerIsBad = True
        else:
            try:
                graph.graphData(ticker)
            except BadTickerException:
                tickerIsBad = True
            
        # return app.send_static_file("index.html")
        return render_template("index.html", form=form, badTicker=tickerIsBad, ticker=ticker)
    return render_template("index.html", form=form)    

@app.route("/", methods=["GET"])
def get_chart():
    return app.send_static_file("chart.png")
