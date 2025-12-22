# users/urls.py
from django.urls import path
from . import views

app_name = "users"

urlpatterns = [
    # ---------------- Auth & account ----------------
    path("register/", views.register_view, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),

    # ---------------- Main pages ----------------
    path("dashboard/", views.dashboard, name="dashboard"),
    path("profile/", views.profile_view, name="profile"),
    path("profile/edit/", views.edit_profile, name="edit_profile"),

    # ---------------- Resume upload + AJAX predict ----------------
    path("profile/upload-resume/", views.upload_resume, name="upload_resume"),
    path("profile/resume-predict/", views.resume_predict, name="resume_predict"),  # AJAX POST endpoint

    # Optional explicit preview endpoint (same semantics as ?preview=1 on profile_view)
    # Frontend can call: /users/profile/preview/?resume_id=42  or ?resume_url=/media/...
    path("profile/preview/", views.profile_view, name="profile_preview"),

    # ---------------- Resume management (used by JS: delete/rename) ----------------
    # These views are small helpers — see suggested implementations below.
    path("resumes/<int:pk>/delete/", views.delete_resume, name="delete_resume"),
    path("resumes/<int:pk>/rename/", views.rename_resume, name="rename_resume"),

    # ---------------- Predict pages (canonical + compatibility aliases) ----------------
    path("predict/", views.predict_from_details, name="predict"),
    path("predict-form/", views.predict_from_details, name="predict_form"),

    # compatibility aliases / wrappers
    path("user-predict/", views.user_predict, name="user_predict"),
    path("predict-legacy/", views.predict_from_details, name="predict_legacy"),

    # ---------------- JSON / utility APIs ----------------
    path("api/predict/", views.predict_api, name="predict_api"),

    # If your view `save_prediction_api` exists in views.py you can re-enable the next line.
    # It was causing import-time errors when missing, so keep it commented until the view is present.
    # path('api/save-prediction/', views.save_prediction_api, name='save_prediction_api'),

    path("api/upload-dataset/", views.upload_dataset, name="upload_dataset"),
]
#


