class DevelopmentConfig(object):
    DATABASE_URI = "sqlite:///stocks-development.db"
    DEBUG = True
    SECRET_KEY = 'tacmot69'
    SEND_FILE_MAX_AGE_DEFAULT = 0
class TestingConfig(object):
    DATABASE_URI = "sqlite://"
    DEBUG = True
