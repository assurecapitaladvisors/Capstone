from flask import render_template

from stocks import app
from stocks.graph import BadTickerException, graphData
from flask.ext.wtf import Form
from wtforms import StringField

from decorators import nocache

class TickerForm(Form):
    ticker = StringField('')

    def validate(self):
        return Form.validate(self)

@app.route("/", methods=["GET", "POST"])
def index():
    form = TickerForm()
    if form.validate_on_submit():
        ticker = form.data['ticker']
        if not ticker:
            print "NOT TICKER"
            return render_template("index.html", form=form, noTicker=True)
        else:
            try:
                graphData(ticker)
            except BadTickerException as e:
                return render_template("index.html", form=form, badTicker=True,
                        ticker=ticker, provider=e.message['provider'],
                        reason=e.message.get('reason', ''))
    return render_template("index.html", form=form)

@app.route("/", methods=["GET"])
def get_chart():
    return app.send_static_file("chart.png")
