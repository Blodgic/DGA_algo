import os

DB_USER = 'blodgic'
DB_PASSWORD = 'f283^fhQ90fe'

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = "postgresql://" + DB_USER + ":" + DB_PASSWORD + "@localhost:5432/blodgic"
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_USE_TLS = True
    MAIL_PORT = 587
    MAIL_USERNAME = 'contact@blodgic.com'
    MAIL_PASSWORD = 'nlhxlkhhdmpwdchu'
    MAIL_DEFAULT_SENDER = 'contact@blodgic.com'
