# users/views.py
"""
Refactored & optimized views for the users app.
- Defensive checks, consistent defaults
- Works with ml_helper payload (model + label_encoder + metadata)
- AJAX partial rendering for predict results (includes prediction_breakdown_json for Chart.js)
- Resume upload/parse + dataset upload + training endpoints
"""
import json
import logging
import os
import re
import sys
import tempfile
import time
import uuid
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.files.storage import default_storage
from django.http import (
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseServerError,
    JsonResponse,
    Http404,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.utils.html import mark_safe
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST

import PyPDF2
import docx

from . import ml_helper
from .forms import PredictForm, ProfileForm, ResumeForm
from .models import Profile
from .models import PredictionHistory

logger = logging.getLogger(__name__)

# --- preload model (best-effort) ---
try:
    ml_helper.reload_model()
    logger.debug("ml_helper: model payload reloaded at import.")
except Exception:
    logger.debug("ml_helper: no model payload on import.", exc_info=True)


# -------------------- Server plotting helper --------------------
# Prefer an external util if available, otherwise provide a safe fallback
try:
    from .utils.plotting import plot_prediction_bars  # expects (top5_dict, out_path)
except Exception:
    # fallback implementation using Matplotlib
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    def plot_prediction_bars(top5: Dict[str, float], out_path: str) -> bool:
        """
        top5: dict label -> percent (0..100)
        out_path: absolute path to save PNG
        Returns True on success.
        """
        if not plt:
            logger.warning("Matplotlib not available; cannot render plot.")
            return False
        try:
            labels = list(top5.keys())
            values = [float(top5[k] or 0.0) for k in labels]

            # Ensure consistent ordering: largest -> top
            pairs = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
            labels_sorted = [p[0] for p in pairs]
            values_sorted = [p[1] for p in pairs]

            # Plot horizontal bars (largest at top)
            # reverse for plt.barh ordering
            labels_sorted = labels_sorted[::-1]
            values_sorted = values_sorted[::-1]

            n = len(labels_sorted)
            height = max(2.0, 0.6 * n + 1.6)
            fig, ax = plt.subplots(figsize=(8, height))
            y_pos = range(len(labels_sorted))

            # color map: use a qualitative palette, map by normalized value
            try:
                cmap = plt.get_cmap("tab10")
            except Exception:
                cmap = None

            colors = []
            for i, v in enumerate(values_sorted):
                if cmap:
                    colors.append(cmap(i % 10))
                else:
                    colors.append((0.38, 0.65, 0.98, 1.0))

            bars = ax.barh(y_pos, values_sorted, align="center", color=colors, height=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels_sorted, fontsize=10, fontweight=600)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.xaxis.set_ticks_position("bottom")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_color("#dddddd")
            ax.tick_params(axis="x", colors="#666666", labelsize=9)
            ax.tick_params(axis="y", colors="#222222")

            # annotate
            for bar in bars:
                w = bar.get_width()
                ax.text(w + 1.6, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=9, fontweight=700)

            fig.tight_layout(pad=0.6)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            fig.savefig(out_path, dpi=96, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
            return True
        except Exception:
            logger.exception("plot_prediction_bars fallback failed")
            try:
                plt.close("all")
            except Exception:
                pass
            return False


# -------------------- Utilities --------------------
def safe_json(obj: Any) -> str:
    try:
        return mark_safe(json.dumps(obj))
    except Exception:
        return mark_safe(json.dumps({}))


def normalize_prediction(raw_pred: Any) -> Tuple[
    List[Tuple[str, Optional[float]]],
    Optional[int],
    Optional[str],
    Optional[str],
    Optional[float],
]:
    """
    Normalizes different shapes of prediction outputs to:
      (prediction_list, prediction_confidence, prediction_breakdown_json, top_label, top_score)
    prediction_list: [(label, percent), ...] sorted desc
    """
    prediction_list: List[Tuple[str, Optional[float]]] = []
    prediction_confidence: Optional[int] = None
    prediction_breakdown_json: Optional[str] = None
    top_label: Optional[str] = None
    top_score: Optional[float] = None

    try:
        if raw_pred is None:
            return (
                prediction_list,
                prediction_confidence,
                prediction_breakdown_json,
                top_label,
                top_score,
            )

        # dict form: {"role":..., "confidence":..., "breakdown": {...}}
        if isinstance(raw_pred, dict):
            breakdown = raw_pred.get("breakdown") or raw_pred.get("probabilities") or raw_pred.get("scores")
            role = raw_pred.get("role")
            conf = raw_pred.get("confidence")

            if isinstance(breakdown, dict) and breakdown:
                vals = {str(k): float(v or 0.0) for k, v in breakdown.items()}
                total = sum(vals.values()) or 1.0
                pct = {k: round((v / total) * 100, 1) for k, v in vals.items()}
                items = sorted(pct.items(), key=lambda x: x[1], reverse=True)
                prediction_list = items
                prediction_breakdown_json = safe_json(pct)
                top_label, top_score = items[0] if items else (None, None)
            elif isinstance(role, (list, tuple)):
                labels = [str(r) for r in role]
                if labels:
                    if len(labels) == 1:
                        prediction_list = [(labels[0], None)]
                        top_label = labels[0]
                    else:
                        eq = round(100.0 / len(labels), 1)
                        prediction_list = [(lbl, eq) for lbl in labels]
                        prediction_breakdown_json = safe_json({lbl: eq for lbl, _ in prediction_list})
                        top_label, top_score = prediction_list[0]
            elif role is not None:
                prediction_list = [(str(role), float(conf) if conf is not None else None)]
                top_label = str(role)
                top_score = float(conf) if conf is not None else None

            if conf is not None and prediction_confidence is None:
                try:
                    prediction_confidence = int(round(float(conf)))
                except Exception:
                    prediction_confidence = None

            return (
                prediction_list,
                prediction_confidence,
                prediction_breakdown_json,
                top_label,
                top_score,
            )

        # list/tuple forms
        if isinstance(raw_pred, (list, tuple)):
            # list of pairs [(label, score), ...]
            if raw_pred and isinstance(raw_pred[0], (list, tuple)) and len(raw_pred[0]) >= 2:
                parsed = []
                for it in raw_pred:
                    lbl = str(it[0])
                    val = None if it[1] is None else float(it[1])
                    parsed.append((lbl, val))
                if any(v is not None for _, v in parsed):
                    scores = [(lbl, 0.0 if v is None else float(v)) for lbl, v in parsed]
                    total = sum(v for _, v in scores) or 1.0
                    pct = {lbl: round((v / total) * 100, 1) for lbl, v in scores}
                    items = sorted(pct.items(), key=lambda x: x[1], reverse=True)
                    return (
                        items,
                        (int(round(items[0][1])) if items else None),
                        safe_json(pct),
                        (items[0][0] if items else None),
                        (items[0][1] if items else None),
                    )
                else:
                    items = [(lbl, None) for lbl, _ in parsed]
                    return items, None, None, (items[0][0] if items else None), None

            # plain list of labels
            labels = [str(x) for x in raw_pred]
            if labels:
                if len(labels) == 1:
                    return [(labels[0], None)], None, None, labels[0], None
                eq = round(100.0 / len(labels), 1)
                items = [(lbl, eq) for lbl in labels]
                return items, int(round(items[0][1])), safe_json({lbl: eq for lbl in labels}), items[0][0], items[0][1]

        # fallback single value
        s = str(raw_pred)
        return [(s, None)], None, None, s, None

    except Exception:
        try:
            s = str(raw_pred)
            return [(s, None)], None, None, s, None
        except Exception:
            return [], None, None, None, None


# -------------------- Authentication --------------------
def register_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")
        confirm = request.POST.get("confirm_password", "")
        if not username or not password:
            messages.error(request, "Username and password required.")
            return redirect("users:register")
        if password != confirm:
            messages.error(request, "Passwords do not match.")
            return redirect("users:register")
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists.")
            return redirect("users:register")
        User.objects.create_user(username=username, email=email, password=password)
        messages.success(request, "Registration successful. Please log in.")
        return redirect("users:login")
    return render(request, "users/register.html")


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect("users:dashboard")
        messages.error(request, "Invalid username or password.")
        return redirect("users:login")
    return render(request, "users/login.html")


def logout_view(request):
    logout(request)
    return redirect("users:login")


# -------------------- Dashboard / profile --------------------
@login_required
def dashboard(request):
    profile = Profile.objects.filter(user=request.user).first()
    form = PredictForm()
    sample_jobs = [
        {"title": "Software Developer", "company": "TechNova", "location": "Bangalore", "type": "Full-Time"},
        {"title": "Data Analyst", "company": "DataWorks", "location": "Hyderabad", "type": "Internship"},
        {"title": "Frontend Engineer", "company": "WebCraft", "location": "Chennai", "type": "Remote"},
    ]
    return render(request, "users/dashboard.html", {"form": form, "job_recommendations": sample_jobs, "profile": profile})


def _read_docx_text_from_path_or_filelike(fp) -> str:
    try:
        doc = docx.Document(fp)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        return ""


def _read_pdf_text_from_path_or_filelike(fp) -> str:
    try:
        reader = PyPDF2.PdfReader(fp)
        out = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                out.append(t)
        return "\n".join(out)
    except Exception:
        return ""


def _extract_text_from_filefield(filefield) -> str:
    """
    Support local file paths (development) or file-like objects opened via default_storage.
    Accepts either a Django FieldFile object, or any object with a .name attribute referencing storage path.
    """
    if not filefield:
        return ""

    # If it's a Django FieldFile (has .path on local filesystem)
    try:
        if hasattr(filefield, "path") and getattr(filefield, "path"):
            path = filefield.path
            if path.lower().endswith(".docx"):
                return _read_docx_text_from_path_or_filelike(path)
            if path.lower().endswith(".pdf"):
                return _read_pdf_text_from_path_or_filelike(path)
            try:
                with open(path, "rb") as fh:
                    return fh.read(200000).decode("utf-8", errors="ignore")
            except Exception:
                return ""
    except Exception:
        pass

    # If it has a .read()/file-like interface already
    try:
        if hasattr(filefield, "read") and hasattr(filefield, "seek"):
            filefield.seek(0)
            name = getattr(filefield, "name", "").lower()
            if name.endswith(".docx"):
                return _read_docx_text_from_path_or_filelike(filefield)
            if name.endswith(".pdf"):
                return _read_pdf_text_from_path_or_filelike(filefield)
            try:
                filefield.seek(0)
                return filefield.read(200000).decode("utf-8", errors="ignore")
            except Exception:
                return ""
    except Exception:
        pass

    # Fallback: try to open via default_storage using the .name attribute
    try:
        name = getattr(filefield, "name", None)
        if name:
            media_url = getattr(settings, "MEDIA_URL", "/media/")
            if name.startswith(media_url):
                name = name[len(media_url) :]
            with default_storage.open(name, "rb") as fh:
                lower = name.lower()
                if lower.endswith(".docx"):
                    return _read_docx_text_from_path_or_filelike(fh)
                if lower.endswith(".pdf"):
                    return _read_pdf_text_from_path_or_filelike(fh)
                try:
                    fh.seek(0)
                    return fh.read(200000).decode("utf-8", errors="ignore")
                except Exception:
                    return ""
    except Exception:
        logger.debug("fallback storage read failed", exc_info=True)

    return ""


@login_required
def profile_view(request):
    """
    Renders user profile and also serves preview JSON when ?preview=1 is present.
    This view builds `resume_list` and returns it in context for templates.
    """
    profile, _ = Profile.objects.get_or_create(user=request.user)

    # Normalize resume list: try common places (single FileField, related managers)
    resume_list: List[Dict[str, Any]] = []

    # 1) single FileField on profile named `resume`
    try:
        f = getattr(profile, "resume", None)
        if f and getattr(f, "name", None):
            resume_list.append(
                {
                    "id": getattr(f, "name", ""),
                    "name": os.path.basename(getattr(f, "name", "")) or "Resume",
                    "url": getattr(f, "url", None),
                    "filefield": f,
                    "updated_at": getattr(profile, "updated_at", None),
                }
            )
    except Exception:
        logger.debug("profile.resume access failed", exc_info=True)

    # 2) try reverse relations / collections (resume_history, resumes, uploads, resume_set)
    for attr in ("resume_history", "resumes", "uploads", "resume_set"):
        if resume_list:
            break
        col = getattr(profile, attr, None)
        if not col:
            continue
        try:
            if hasattr(col, "all"):
                iterable = col.all()
            else:
                iterable = col
            for r in iterable:
                filefield = getattr(r, "file", None) or getattr(r, "resume", None) or getattr(r, "upload", None)
                url = getattr(r, "url", None) or (filefield.url if getattr(filefield, "url", None) else None)
                resume_list.append(
                    {
                        "id": getattr(r, "id", None) or getattr(r, "pk", None) or getattr(r, "name", None),
                        "name": getattr(r, "name", None) or getattr(r, "filename", None) or os.path.basename(getattr(filefield, "name", "") or "") or "Resume",
                        "url": url,
                        "filefield": filefield,
                        "updated_at": getattr(r, "updated_at", None) or getattr(r, "created_at", None),
                        "obj": r,
                    }
                )
            if resume_list:
                break
        except Exception:
            logger.debug("iterating resume collection failed for attr=%s", attr, exc_info=True)
            continue

    # If preview requested (support both profile/?preview=1 and dedicated preview route), return JSON
    if request.GET.get("preview") == "1" or request.path.endswith("/preview/"):
        resume_id = request.GET.get("resume_id")
        resume_url = request.GET.get("resume_url")
        chosen = None

        if resume_id:
            for r in resume_list:
                if str(r.get("id")) == str(resume_id) or str(r.get("name")) == str(resume_id):
                    chosen = r
                    break
        if not chosen and resume_url:
            for r in resume_list:
                if r.get("url") and r.get("url") == resume_url:
                    chosen = r
                    break
        if not chosen and resume_list:
            chosen = resume_list[0]
        if not chosen:
            return JsonResponse({"resume_text": ""})

        filefield = chosen.get("filefield")
        if not filefield and chosen.get("url"):
            url = chosen.get("url")
            media_url = getattr(settings, "MEDIA_URL", "/media/")
            name = url[len(media_url) :] if url.startswith(media_url) else os.path.basename(url)
            class Tmp:
                def __init__(self, n):
                    self.name = n
            filefield = Tmp(name)

        try:
            text = _extract_text_from_filefield(filefield) or ""
        except Exception:
            logger.exception("preview extraction failed", exc_info=True)
            text = ""
        return JsonResponse({"resume_text": text.strip()})

    # normal render
    return render(request, "users/profile.html", {"profile": profile, "resume_list": resume_list})


@login_required
def edit_profile(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = ProfileForm(request.POST, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Profile updated.")
            return redirect("users:profile")
        messages.error(request, "Please correct the errors.")
    else:
        form = ProfileForm(instance=profile)
    return render(request, "users/edit_profile.html", {"form": form})


# -------------------- Resume parsing & predict --------------------
def _extract(pattern: str, text: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def _parse_resume_text(text: str) -> Dict[str, str]:
    skills = sorted(
        {
            s.strip()
            for s in re.findall(
                r"(Python|Java|C\+\+|SQL|Django|ML|Machine Learning|HTML|CSS|JavaScript|AWS|Docker|Kubernetes)",
                text,
                re.IGNORECASE,
            )
        },
        key=lambda s: s.lower(),
    )
    return {
        "Degree": _extract(r"(B\.Tech|BTech|B\.E|BE|M\.Tech|MTech|MCA|BCA)", text, ""),
        "Major": _extract(r"(Computer Science|Information Technology|Electronics|Mechanical|Electrical)", text, ""),
        "Specialization": "",
        "CGPA": "",
        "Skills": ", ".join(skills),
        "Certification": "",
        "Years of Experience": _extract(r"(\d+)\s+years? of experience", text, "0"),
        "Preferred Industry": _extract(r"(Software|IT|Manufacturing|Finance|Healthcare)", text, ""),
    }


@login_required
def upload_resume(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = ResumeForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            profile = form.save()
            try:
                resume_path = getattr(profile.resume, "path", None)
                text = ""
                if resume_path and resume_path.lower().endswith(".pdf"):
                    reader = PyPDF2.PdfReader(resume_path)
                    for p in reader.pages:
                        text += (p.extract_text() or "") + "\n"
                elif resume_path and resume_path.lower().endswith(".docx"):
                    doc = docx.Document(resume_path)
                    text = "\n".join([p.text for p in doc.paragraphs])
                else:
                    # attempt file-like read
                    text = _extract_text_from_filefield(getattr(profile, "resume", None))
                auto = _parse_resume_text(text)
                pred = ml_helper.predict_from_dict(auto)
                pl, pconf, pjson, top_label, top_score = normalize_prediction(pred)
                if top_label:
                    profile.predicted_role = str(top_label)
                profile.predicted_confidence = float(pconf) if pconf is not None else None
                profile.save()
            except Exception:
                logger.exception("resume processing failed")
            messages.success(request, "Resume uploaded & processed.")
            return redirect("users:profile")
        messages.error(request, "Fix errors.")
    else:
        form = ResumeForm(instance=profile)
    return render(request, "users/upload_resume.html", {"form": form})


@login_required
@require_http_methods(["POST"])
def resume_predict(request):
    resume = request.FILES.get("resume")
    if not resume:
        return HttpResponseBadRequest("No resume uploaded.")
    try:
        buf = BytesIO(resume.read())
        buf.seek(0)
        text = ""
        if resume.name.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(buf)
            for p in reader.pages:
                text += (p.extract_text() or "") + "\n"
        elif resume.name.lower().endswith(".docx"):
            doc = docx.Document(buf)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return HttpResponseBadRequest("Only PDF or DOCX allowed.")
    except Exception as exc:
        return HttpResponseServerError(f"Failed to read resume: {exc}")

    auto = _parse_resume_text(text)
    try:
        prediction = ml_helper.predict_from_dict(auto)
    except Exception as exc:
        return JsonResponse({"error": f"Prediction failed: {exc}"}, status=500)

    try:
        profile, _ = Profile.objects.get_or_create(user=request.user)
        profile.resume.save(resume.name, resume, save=True)
        pl, pconf, pjson, top_label, top_score = normalize_prediction(prediction)
        if top_label:
            profile.predicted_role = str(top_label)
        profile.predicted_confidence = float(pconf) if pconf is not None else None
        profile.save()
    except Exception:
        logger.exception("resume predict save failed")

    return JsonResponse({"auto_fill": auto, "prediction": prediction, "resume_text": text[:10000]})


# -------------------- Resume management helpers --------------------
@login_required
@require_POST
def delete_resume(request, pk):
    """
    Delete a resume by pk. If a Resume model exists it will be used.
    Otherwise falls back to clearing Profile.resume when it matches.
    """
    profile = Profile.objects.filter(user=request.user).first()
    if not profile:
        return JsonResponse({"error": "Profile not found"}, status=404)

    # Try Resume model first (if present)
    try:
        from .models import Resume  # optional
        r = Resume.objects.filter(pk=pk).first()
        if r:
            owner = getattr(r, "profile", None) or getattr(r, "user", None)
            if owner and getattr(owner, "user", None) != request.user and getattr(owner, "user", owner) != request.user:
                return ("Not allowed")
            filefield = getattr(r, "file", None) or getattr(r, "resume", None) or getattr(r, "upload", None)
            try:
                if filefield and getattr(filefield, "name", None):
                    if default_storage.exists(filefield.name):
                        default_storage.delete(filefield.name)
            except Exception:
                logger.debug("failed to delete storage file for Resume", exc_info=True)
            r.delete()
            return JsonResponse({"status": "ok"})
    except Exception:
        # Resume model not present; continue to profile fallback
        pass

    # Fallback for single profile.resume
    try:
        f = getattr(profile, "resume", None)
        if f and (str(pk) in (str(getattr(f, "name", "")) or "") or str(pk) == str(getattr(f, "name", ""))):
            try:
                if getattr(f, "name", None) and default_storage.exists(f.name):
                    default_storage.delete(f.name)
            except Exception:
                logger.debug("failed to delete profile resume file", exc_info=True)
            profile.resume = None
            profile.save()
            return JsonResponse({"status": "ok"})
    except Exception:
        pass

    return JsonResponse({"error": "Resume not found"}, status=404)


@login_required
@require_POST
def rename_resume(request, pk):
    """
    Rename resume metadata. Supports JSON or form POST with 'name'.
    Does not rename storage blobs (storage move is backend-specific).
    """
    profile = Profile.objects.filter(user=request.user).first()
    if not profile:
        return JsonResponse({"error": "Profile not found"}, status=404)

    new_name = ""
    try:
        data = json.loads(request.body.decode("utf-8") or "{}")
        new_name = data.get("name", "") or request.POST.get("name", "") or ""
    except Exception:
        new_name = request.POST.get("name", "") or ""

    if not new_name:
        return JsonResponse({"error": "Missing name"}, status=400)

    try:
        from .models import Resume  # optional
        r = Resume.objects.filter(pk=pk).first()
        if r:
            owner = getattr(r, "profile", None) or getattr(r, "user", None)
            if owner and getattr(owner, "user", None) != request.user and getattr(owner, "user", owner) != request.user:
                return ("Not allowed")
            r.name = new_name
            r.save()
            return JsonResponse({"status": "ok", "name": new_name})
    except Exception:
        pass

    # Fallback: update profile metadata if supported
    try:
        f = getattr(profile, "resume", None)
        if f and (str(pk) in (str(getattr(f, "name", "")) or "") or str(pk) == str(getattr(f, "name", ""))):
            if hasattr(profile, "resume_name"):
                profile.resume_name = new_name
                profile.save()
                return JsonResponse({"status": "ok", "name": new_name})
            return JsonResponse({"error": "Rename not supported for this storage/file type"}, status=501)
    except Exception:
        pass

    return JsonResponse({"error": "Resume not found"}, status=404)


# -------------------- Dataset upload & training --------------------
@login_required
@require_http_methods(["POST"])
def upload_dataset(request):
    csv_file = request.FILES.get("file")
    label_col = request.POST.get("label_col", "Job Role")
    if not csv_file:
        return HttpResponseBadRequest("No file uploaded.")
    fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(csv_file.name)[1] or ".csv")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            for chunk in csv_file.chunks():
                f.write(chunk)
        model_path = ml_helper.train_model_from_file(tmp_path, label_col=label_col)
        try:
            ml_helper.reload_model(path=model_path)
            logger.info("Reloaded ML artifact after training: %s", model_path)
        except Exception:
            logger.exception("Failed to reload ML artifact after training")
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return JsonResponse({"error": str(exc)}, status=400)
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return JsonResponse({"status": "ok", "model_path": model_path})


# -------------------- JSON prediction API --------------------
@login_required
@require_http_methods(["POST"])
def predict_api(request):
    try:
        data = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON.")
    try:
        pred = ml_helper.predict_from_dict(data)
    except Exception as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    return JsonResponse({"prediction": pred})


# -------------------- Predict debug --------------------
@csrf_exempt
def predict_debug(request):
    try:
        print("===== PREDICT DEBUG =====", file=sys.stdout)
        print("Method:", request.method, file=sys.stdout)
        try:
            print("POST keys:", list(request.POST.keys()), file=sys.stdout)
            if request.POST:
                print("POST data (truncated):", dict(request.POST) if len(request.POST) < 50 else "(too many keys)", file=sys.stdout)
        except Exception:
            pass
        raw = request.body.decode("utf-8") if request.body else ""
        if raw:
            print("Raw (trunc):", raw[:4000], file=sys.stdout)
        if request.FILES:
            print("FILES:", list(request.FILES.keys()), file=sys.stdout)
        print("===== END DEBUG =====", file=sys.stdout)
    except Exception as exc:
        print("Debug logging failed:", exc, file=sys.stdout)
    return JsonResponse({"role": "Debug Role", "confidence": 100})


# -------------------- Predict (main) --------------------
@login_required
def predict_from_details(request):
    profile, _ = Profile.objects.get_or_create(user=request.user)
    prediction = None
    prediction_list = None
    prediction_confidence = None
    prediction_breakdown_json = None

    # chart_url will be added to context if plotting succeeds
    chart_url = None

    if request.method == "POST":
        form = PredictForm(request.POST)
        if form.is_valid():
            # Build input data; these keys must match feature names used during training
            input_data = {
                "Degree": form.cleaned_data.get("degree", "") or "",
                "Major": form.cleaned_data.get("major", "") or "",
                "Specialization": form.cleaned_data.get("specialization", "") or "",
                "CGPA": form.cleaned_data.get("cgpa", "") or "",
                "Skills": form.cleaned_data.get("skills", "") or "",
                "Certification": form.cleaned_data.get("certification", "") or "",
                "Years of Experience": form.cleaned_data.get("experience", "") or "",
                "Preferred Industry": form.cleaned_data.get("industry", "") or "",
            }

            try:
                raw = ml_helper.predict_from_dict(input_data)
            except Exception as exc:
                messages.error(request, f"Prediction failed: {exc}")
                raw = None
            else:
                prediction = raw
                prediction_list, prediction_confidence, prediction_breakdown_json, top_label, top_score = normalize_prediction(raw)

                # fallback: try predict_topk_percent or predict_proba if breakdown missing
                if (not prediction_list or len(prediction_list) <= 1):
                    tried = False
                    if hasattr(ml_helper, "predict_topk_percent"):
                        try:
                            topk = ml_helper.predict_topk_percent(input_data, k=5)
                            prediction_list = [(lbl, prob) for lbl, prob in topk]
                            prediction_breakdown_json = mark_safe(json.dumps({lbl: prob for lbl, prob in topk}))
                            prediction_confidence = int(round(prediction_list[0][1])) if prediction_list and prediction_list[0][1] is not None else prediction_confidence
                            tried = True
                        except Exception:
                            logger.debug("predict_topk_percent failed", exc_info=True)
                    if not tried and hasattr(ml_helper, "predict_proba"):
                        try:
                            probs = ml_helper.predict_proba(input_data)
                            if isinstance(probs, dict) and probs:
                                total = sum(float(v) for v in probs.values()) or 1.0
                                pct = {str(k): round((float(v) / total) * 100, 1) for k, v in probs.items()}
                                prediction_breakdown_json = mark_safe(json.dumps(pct))
                                prediction_list = sorted(pct.items(), key=lambda x: x[1], reverse=True)
                                top_label, top_score = prediction_list[0]
                                if prediction_confidence is None and top_score is not None:
                                    try:
                                        prediction_confidence = int(round(float(top_score)))
                                    except Exception:
                                        pass
                        except Exception:
                            logger.debug("predict_proba failed", exc_info=True)

                # persist best top_label into profile
                if prediction_list:
                    try:
                        profile.predicted_role = str(prediction_list[0][0])
                        profile.predicted_confidence = float(prediction_confidence) if prediction_confidence is not None else None
                        profile.save()
                    except Exception:
                        logger.exception("Failed to persist profile prediction")

                # --- NEW: persist prediction history (defensive) ---
                try:
                    # compact summary of inputs
                    skills_summary = input_data.get("Skills", "") or ""
                    compact_input = {
                        "Skills": skills_summary,
                        "Experience": input_data.get("Years of Experience", ""),
                        "Degree": input_data.get("Degree", ""),
                    }

                    # probability: prefer top_score if available, else prediction_confidence
                    prob_val = None
                    try:
                        prob_val = float(top_score) if top_score is not None else (float(prediction_confidence) if prediction_confidence is not None else None)
                    except Exception:
                        prob_val = None

                    # breakdown JSON as string (if present)
                    breakdown_json_str = None
                    try:
                        if prediction_breakdown_json:
                            # prediction_breakdown_json may be mark_safe or JSON string
                            # ensure it's a JSON string
                            if isinstance(prediction_breakdown_json, str):
                                # check if it's safe-marked JSON string or already JSON string
                                try:
                                    # attempt load -> dump to normalize
                                    breakdown_obj = json.loads(prediction_breakdown_json)
                                    breakdown_json_str = json.dumps(breakdown_obj)
                                except Exception:
                                    # if not JSON, attempt to coerce
                                    breakdown_json_str = json.dumps({"detail": str(prediction_breakdown_json)})
                            else:
                                try:
                                    breakdown_obj = json.loads(str(prediction_breakdown_json))
                                    breakdown_json_str = json.dumps(breakdown_obj)
                                except Exception:
                                    breakdown_json_str = json.dumps({"detail": str(prediction_breakdown_json)})
                    except Exception:
                        breakdown_json_str = None

                    # Create kwargs and try to persist robustly (support models without metadata/breakdown fields)
                    create_kwargs = {
                        "user": request.user,
                        "skills": skills_summary,
                        "predicted_role": str(profile.predicted_role) if profile.predicted_role else (str(top_label) if top_label else ""),
                        "probability": prob_val,
                    }
                    # add optional fields
                    if compact_input:
                        try:
                            create_kwargs["metadata"] = json.dumps(compact_input)
                        except Exception:
                            create_kwargs["metadata"] = None
                    if breakdown_json_str:
                        create_kwargs["breakdown"] = breakdown_json_str

                    try:
                        PredictionHistory.objects.create(**create_kwargs)
                    except TypeError:
                        # model doesn't accept metadata/breakdown -> fallback minimal
                        minimal = {
                            "user": request.user,
                            "skills": skills_summary,
                            "predicted_role": create_kwargs.get("predicted_role"),
                            "probability": prob_val,
                        }
                        PredictionHistory.objects.create(**minimal)
                except Exception:
                    logger.exception("Failed to save PredictionHistory (non-fatal)")

                messages.success(request, f"Predicted and saved: {profile.predicted_role if profile.predicted_role else 'result'}")

                # --- NEW: generate server-side PNG of top-5 and include chart_url in context ---
                try:
                    # Build a normalized top5 dict: label -> percent (0..100)
                    top5_dict: Dict[str, float] = {}
                    # prefer prediction_breakdown_json if present
                    if prediction_breakdown_json:
                        try:
                            pb = json.loads(prediction_breakdown_json) if isinstance(prediction_breakdown_json, str) else prediction_breakdown_json
                            if isinstance(pb, dict):
                                for k, v in pb.items():
                                    try:
                                        n = float(v or 0.0)
                                    except Exception:
                                        n = 0.0
                                    if n <= 1.01:
                                        n = n * 100.0
                                    top5_dict[str(k)] = round(n, 1)
                        except Exception:
                            logger.debug("Invalid prediction_breakdown_json, falling back to prediction_list", exc_info=True)
                            pb = None
                    if not top5_dict and prediction_list:
                        for lbl, val in (prediction_list[:5] if isinstance(prediction_list, (list, tuple)) else []):
                            try:
                                n = float(val) if val is not None else 0.0
                            except Exception:
                                n = 0.0
                            if n <= 1.01:
                                n = n * 100.0
                            top5_dict[str(lbl)] = round(n, 1)

                    # Only attempt plotting if we have at least one numeric value and plotting util is present
                    if top5_dict and callable(plot_prediction_bars):
                        # unique filename per user+timestamp to avoid collisions & caching issues
                        unique = f"{getattr(request.user, 'id', 'anon')}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
                        filename = f"prediction_chart_{unique}.png"
                        media_root = getattr(settings, "MEDIA_ROOT", None) or os.path.join(getattr(settings, "BASE_DIR", "."), "media")
                        abs_path = os.path.join(media_root, filename)
                        try:
                            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                        except Exception:
                            pass
                        ok = False
                        try:
                            ok = plot_prediction_bars(top5_dict, abs_path)
                        except Exception:
                            logger.exception("plot_prediction_bars raised", exc_info=True)
                            ok = False
                        if ok and os.path.exists(abs_path):
                            ts = int(time.time())
                            media_url = getattr(settings, "MEDIA_URL", "/media/")
                            if not media_url.endswith("/"):
                                media_url += "/"
                            # ensure correct URL formation (avoid os.path.join for URL)
                            chart_url = f"{media_url}{filename}?v={ts}"
                            logger.info("Plot created at %s -> %s", abs_path, chart_url)
                        else:
                            logger.debug("Plot not created or missing: %s", abs_path)
                except Exception:
                    logger.exception("Failed while attempting to generate chart image", exc_info=True)

            # If AJAX request, return partial HTML (result box) to update chart + history
            is_ajax = request.META.get("HTTP_X_REQUESTED_WITH") == "XMLHttpRequest" or request.headers.get("x-requested-with") == "XMLHttpRequest"
            if is_ajax:
                # load recent history for the AJAX response
                try:
                    history = list(PredictionHistory.objects.filter(user=request.user).order_by('-created_at')[:10])
                except Exception:
                    history = []
                ctx = {
                    "prediction": prediction,
                    "prediction_confidence": prediction_confidence,
                    "prediction_list": prediction_list,
                    "prediction_breakdown_json": prediction_breakdown_json,
                    "history": history,
                    "chart_url": chart_url,
                }
                try:
                    html = render_to_string("users/result_box.html", ctx, request=request)
                    return HttpResponse(html)
                except Exception:
                    logger.exception("Failed to render result partial", exc_info=True)
                    return HttpResponseServerError("Failed to render result partial")
        else:
            messages.error(request, "Correct the form errors.")
            is_ajax = request.META.get("HTTP_X_REQUESTED_WITH") == "XMLHttpRequest" or request.headers.get("x-requested-with") == "XMLHttpRequest"
            if is_ajax:
                try:
                    html = render_to_string("users/predict_form.html", {"form": form, "profile": profile}, request=request)
                    return HttpResponse(html, status=400)
                except Exception:
                    logger.exception("Failed to render form partial", exc_info=True)
                    return HttpResponseServerError("Failed to render error partial")
    else:
        form = PredictForm(initial={"experience": profile.experience_years or ""})

    # normal render: include history
    try:
        history = list(PredictionHistory.objects.filter(user=request.user).order_by('-created_at')[:10])
    except Exception:
        history = []

    # If we computed a chart_url earlier (from POST), include it; otherwise None
    context = {"form": form, "prediction": None, "profile": profile, "history": history}
    if chart_url:
        context["chart_url"] = chart_url

    return render(request, "users/predict_form.html", context)


# --- Compatibility / small wrapper replaced old standalone predict_view ---
@login_required
def predict_view(request):
    # reuse main predict logic (keeps single source of truth)
    return predict_from_details(request)


# compatibility wrappers
@login_required
def user_predict(request):
    return predict_from_details(request)


@login_required
def predict(request):
    return predict_from_details(request)


@login_required
def predict_form(request):
    return predict_from_details(request)


# EOF
