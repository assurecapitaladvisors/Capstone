import os

from flask import Flask
from flask_wtf.csrf import CsrfProtect
from flask.ext.assets import Environment, Bundle

app = Flask(__name__)
CsrfProtect(app)
config_path = os.environ.get("CONFIG_PATH", "stocks.config.DevelopmentConfig")
app.config.from_object(config_path)

assets = Environment(app)
assets.manifest = 'cache'
# Bundle is not 100% correct but good enough
chart = Bundle('chart.png')
assets.register('chart', chart)

import api
import views
