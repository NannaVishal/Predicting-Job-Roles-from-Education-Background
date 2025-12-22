from django.db import models
from django.conf import settings
import json
from django.db import models
from django.contrib.auth.models import User


class UploadedDataset(models.Model):
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    file = models.FileField(upload_to="datasets/")
    original_name = models.CharField(max_length=255, blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"{self.original_name or self.file.name} by {self.uploaded_by}"
    
class TrainedModel(models.Model):
    uploaded_dataset = models.ForeignKey(UploadedDataset, on_delete=models.CASCADE)
    model_file = models.FileField(upload_to="models/")
    created_at = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField(null=True, blank=True)
    feature_columns = models.TextField(null=True, blank=True,
                                       help_text="JSON-encoded list of feature column names")
    meta = models.TextField(null=True, blank=True,
                            help_text="JSON-encoded metadata (e.g., cat_cols, label_col)")
    def __str__(self):
        return f"TrainedModel {self.id} (dataset={self.uploaded_dataset.id})"
    def set_feature_columns(self, cols):
        self.feature_columns = json.dumps(cols)
    def get_feature_columns(self):
        if not self.feature_columns:
            return []
        try:
            return json.loads(self.feature_columns)
        except Exception:
            return []
    def set_meta(self, meta_obj):
        self.meta = json.dumps(meta_obj)

    def get_meta(self):
        if not self.meta:
            return {}
        try:
            return json.loads(self.meta)
        except Exception:
            return {}
        
 

class Profile(models.Model):
 user = models.OneToOneField(User, on_delete=models.CASCADE)
 phone = models.CharField(max_length=20, blank=True)
 location = models.CharField(max_length=120, blank=True)
 experience_years = models.PositiveSmallIntegerField(null=True, blank=True)
 current_company = models.CharField(max_length=120, blank=True)
 resume = models.FileField(upload_to='resumes/', null=True, blank=True)
 predicted_role = models.CharField(max_length=120, blank=True)
 predicted_confidence = models.FloatField(null=True, blank=True)


def __str__(self):
 return f"{self.user.username} Profile"     


class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    skills = models.TextField(blank=True)
    predicted_role = models.CharField(max_length=200)
    probability = models.FloatField(null=True, blank=True)
    metadata = models.TextField(blank=True, null=True)   # <-- ADD THIS
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} → {self.predicted_role}"
    

# users/models.py
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings
import json

class Resume(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to="resumes/")
    name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.name}"

