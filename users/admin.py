# from django.contrib import admin

# # Register your models here.
# from .models import Job

# admin.site.register(Job)
# users/admin.py
from django.contrib import admin
from django.apps import apps

# Register all models in 'users' app automatically
app_models = apps.get_app_config('users').get_models()
for model in app_models:
    try:
        admin.site.register(model)
    except admin.sites.AlreadyRegistered:
        pass
