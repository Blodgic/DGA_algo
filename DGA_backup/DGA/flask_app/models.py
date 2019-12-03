from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from .app import db
from sqlalchemy.dialects.postgresql import INET


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    api_requests = db.relationship('APIRequest', backref='user', lazy=True)
    site_requests = db.relationship('SiteRequest', backref='user', lazy=True)
    tokens = db.relationship('APIToken', backref='user', lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_token(self):
        return self.tokens.filter(APIToken.balance > 0).first()


class DomainDGA(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    family = db.Column(db.String(64), index=True, nullable=False)
    domain = db.Column(db.String(256), index=True, unique=True, nullable=False)
    time = db.Column(db.DateTime, nullable=False)
    source = db.Column(db.String(256), nullable=False)


class TopDomain(db.Model):
    rank = db.Column(db.Integer, primary_key=True)
    domain = db.Column(db.String(256), index=True, unique=True, nullable=False)
    open_page_rank = db.Column(db.Float, nullable=False)


class APIRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(INET)
    token = db.Column(db.Integer, db.ForeignKey('api_token.id'), nullable=True, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    query = db.Column(db.String(256), nullable=False)
    time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())


class SiteRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(INET)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    page = db.Column(db.String(256), nullable=False)
    time = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())


class APIToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True, index=True)
    token = db.Column(db.String(128), unique=True)
    requests = db.relationship('APIRequest', backref='api_token', lazy='dynamic')
    balance = db.Column(db.Integer, nullable=False, default=100)
    created = db.Column(db.DateTime, nullable=False, default=datetime.utcnow())
