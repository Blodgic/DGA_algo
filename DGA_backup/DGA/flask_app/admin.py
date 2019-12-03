from flask_admin import Admin, AdminIndexView, helpers, expose
from flask_admin.contrib.sqla import ModelView
from flask_login import current_user, login_user, logout_user, login_required
from flask import request, redirect, url_for, abort
from .app import db
from .forms import LoginForm, RegistrationForm
from .models import User, APIToken, APIRequest
from datetime import datetime, timedelta


class BlodgicAdminIndexView(AdminIndexView):
    @expose('/')
    def index(self):
        if not current_user.is_authenticated:
            return redirect(url_for('.login_view'))
        num_users = User.query.count()
        num_api_tokens = APIToken.query.count()
        num_api_requests = db.session.query(APIRequest).count()
        return self.render(self._template,
                           num_users=num_users,
                           num_api_tokens=num_api_tokens,
                           num_api_requests=num_api_requests)

    @expose('/login/', methods=('GET', 'POST'))
    def login_view(self):
        # handle user login
        form = LoginForm(request.form)
        if helpers.validate_form_on_submit(form):
            user = User.query.filter_by(username=form.username.data).first()
            if user and user.is_admin:
                login_user(user)
            else:
                abort(401)

        if current_user.is_authenticated:
            return redirect(url_for('.index'))

        return self.render('admin/login.html', title='Sign In', form=form)

    @expose('/register/', methods=('GET', 'POST'))
    def register_view(self):
        form = RegistrationForm(request.form)
        if helpers.validate_form_on_submit(form):
            user = User()

            form.populate_obj(user)
            # we hash the users password to avoid saving it as plaintext in the db,
            # remove to use plain text:
            user.set_password(form.password.data)

            db.session.add(user)
            db.session.commit()

            login_user(user)
            return redirect(url_for('.index'))
        link = '<p>Already have an account? <a href="' + url_for('.login_view') + '">Click here to log in.</a></p>'
        self._template_args['form'] = form
        self._template_args['link'] = link
        return super(BlodgicAdminIndexView, self).index()

    @expose('/logout/')
    def logout_view(self):
        logout_user()
        return redirect(url_for('index'))


class UserView(ModelView):
    column_list = ('username', 'email')
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin


class DomainDGAView(ModelView):
    column_searchable_list = ['domain', 'source', 'time']
    column_default_sort = ('time', True)
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin


class TopDomainView(ModelView):
    column_list = ('rank', 'domain', 'open_page_rank')
    column_sortable_list = ['rank', 'domain', 'open_page_rank']
    column_searchable_list = ['domain', 'open_page_rank']
    column_default_sort = ('rank', True)
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin


class APIRequestView(ModelView):
    column_sortable_list = ['user_id', 'query', 'time']
    column_searchable_list = ['query', 'time', 'ip_address']
    column_default_sort = ('time', True)
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin


class SiteRequestView(ModelView):
    column_sortable_list = ['user_id', 'page', 'time']
    column_searchable_list = ['page', 'time', 'ip_address']
    column_default_sort = ('time', True)
    can_create = False
    can_edit = False
    can_delete = False
    can_view_details = True
    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin


class APITokenView(ModelView):
    column_sortable_list = ['user_id', 'token', 'balance', 'created']
    column_searchable_list = ['created', 'balance', 'created']
    column_default_sort = ('created', True)
    can_create = True
    can_edit = True
    can_delete = True
    can_view_details = True

    def is_accessible(self):
        return current_user.is_authenticated and current_user.is_admin