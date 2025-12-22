from django import forms
from .models import Profile

# ---- global widget style ----
INPUT_CSS = {'class': 'input', 'style': 'padding:10px;border-radius:10px;'}


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = [
            'phone',
            'location',
            'experience_years',
            'current_company',
            'resume',
        ]
        widgets = {
            'phone': forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Phone number'}),
            'location': forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'City, Country'}),
            'experience_years': forms.NumberInput(attrs={**INPUT_CSS, 'min': 0}),
            'current_company': forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Company name'}),
            'resume': forms.FileInput(attrs={
                'accept': '.pdf,.docx',
                'class': 'input',
                'style': 'padding:10px;border-radius:10px;background:#fff;',
            }),
        }
        help_texts = {
            'phone': 'Include country code if applicable.',
            'experience_years': 'Years of experience (e.g. 1, 3, 7).',
        }


class ResumeForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['resume']
        widgets = {
            'resume': forms.FileInput(
                attrs={
                    'accept': '.pdf,.docx',
                    'class': 'input',
                    'style': 'padding:10px;border-radius:10px;background:#fff;',
                }
            )
        }
        help_texts = {
            'resume': 'Upload PDF or DOCX resume only.',
        }


class PredictForm(forms.Form):
    degree = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Degree (e.g. B.Tech, BE)'}),
    )
    major = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Major / Branch'}),
    )
    specialization = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Specialization'}),
    )
    cgpa = forms.FloatField(
        required=False,
        min_value=0.0,
        max_value=10.0,
        widget=forms.NumberInput(attrs={**INPUT_CSS, 'placeholder': 'CGPA (optional)'}),
    )
    skills = forms.CharField(
        required=False,
        widget=forms.Textarea(
            attrs={
                'rows': 3,
                **INPUT_CSS,
                'placeholder': 'Comma-separated skills (Python, Django, SQL...)',
            }
        ),
    )
    certification = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Certifications'}),
    )
    experience = forms.FloatField(
        required=False,
        min_value=0.0,
        widget=forms.NumberInput(attrs={**INPUT_CSS, 'placeholder': 'Years of experience'}),
    )
    industry = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={**INPUT_CSS, 'placeholder': 'Preferred industry'}),
    )

    def clean_skills(self):
        """Normalize comma-separated skills."""
        skills = self.cleaned_data.get('skills', '')
        return ', '.join(s.strip() for s in skills.split(',') if s.strip())
