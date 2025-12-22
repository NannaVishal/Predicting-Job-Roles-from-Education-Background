# jobsite/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

# import users views so we can add top-level aliases (helps older templates)
from users import views as users_views

urlpatterns = [
    path('admin/', admin.site.urls),

    # include users app with namespace 'users'
    path('', include(('users.urls', 'users'), namespace='users')),

    # Top-level aliases (so templates that call {% url 'logout' %} still work)
    path('login/', users_views.login_view, name='login'),
    path('logout/', users_views.logout_view, name='logout'),
    path('dashboard/', users_views.dashboard, name='dashboard'),
    path('profile/', users_views.profile_view, name='profile'),
    path('profile/upload-resume/', users_views.upload_resume, name='upload_resume'),

    # <-- added alias for the edit profile view (fixes NoReverseMatch)
    path('profile/edit/', users_views.edit_profile, name='edit_profile'),

    path('predict/', users_views.predict_from_details, name='predict'),
    path('predict-form/', users_views.predict_from_details, name='predict_form'),

    # back-compat alias used previously
    path('user/predict/', users_views.predict_from_details, name='userpredict'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
