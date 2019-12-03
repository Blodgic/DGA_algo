from flask import g, abort, jsonify, Blueprint
from flask_httpauth import HTTPBasicAuth, HTTPTokenAuth
from .app import db
from .models import User, APIToken

tokens = Blueprint('tokens', __name__)
basic_auth = HTTPBasicAuth()
token_auth = HTTPTokenAuth()


@basic_auth.verify_password
def verify_password(username, password):
    user = User.query.filter_by(username=username).first()
    if user is None:
        return False
    g.current_user = user
    return user.check_password(password)


@token_auth.verify_token
def verify_token(token):
    token_from_db = APIToken.query.filter_by(token=token).first()
    if token_from_db is not None:
        balance = token_from_db.balance
        if token != 'TEST_TOKEN':
            token_from_db.balance -= 1
            db.session.add(token_from_db)
            db.session.commit()
        return balance > 0
    return False


@basic_auth.error_handler
def basic_auth_error():
    return abort(401)


@token_auth.error_handler
def token_auth_error():
    payload = {'error': 'No token or invalid token'}
    response = jsonify(payload)
    response.status_code = 401
    return response


@tokens.route('/tokens', methods=['POST'])
@basic_auth.login_required
def get_token():
    token_obj = g.current_user.get_token()
    return jsonify({'token': token_obj.token if token_obj else ''})

