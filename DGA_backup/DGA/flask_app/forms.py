from flask_wtf import FlaskForm
from wtforms.fields.html5 import EmailField
from wtforms import StringField, PasswordField, TextAreaField, IntegerField, DateField, SubmitField, BooleanField
from wtforms.validators import InputRequired, Email, Optional, ValidationError, EqualTo
from wtforms.widgets import TextArea
from .models import User


class ContactForm(FlaskForm):
    email = EmailField('Email address', [InputRequired(), Email()], render_kw={'style': 'min-width: 100%'})
    subject = StringField('Subject', [InputRequired()], render_kw={'style': 'min-width: 100%'})
    message = TextAreaField('Message',  [InputRequired()], widget=TextArea(), render_kw={'style': 'min-width: 100%'})
    submit = SubmitField('Send')


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    email = StringField('Email', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    password2 = PasswordField(
        'Repeat Password', validators=[InputRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('This username is taken.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('There is already a user with this email address.')