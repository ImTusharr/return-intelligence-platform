from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import click
from PIL import Image, UnidentifiedImageError
from flask import (
    Flask,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from flask_login import (
    current_user,
    login_required,
    login_user,
    logout_user,
)
from sqlalchemy import case, func, or_, select
from supabase import create_client
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from ai_analysis import analyze_return_request
from config import Config
from extensions import csrf, db, login_manager
from models import ReturnImage, ReturnRequest, User


# ---------------------------------------------------------
# SUPABASE STORAGE
# ---------------------------------------------------------

_supabase_client = None


def get_supabase_client():
    """Create the Supabase client only when it is needed."""
    global _supabase_client

    if _supabase_client is None:
        supabase_url = os.environ.get("SUPABASE_URL")
        service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not service_key:
            raise RuntimeError(
                "Supabase Storage configuration is missing. "
                "Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )

        _supabase_client = create_client(
            supabase_url,
            service_key,
        )

    return _supabase_client


def get_storage_bucket():
    bucket = os.environ.get(
        "SUPABASE_STORAGE_BUCKET",
        "return-images",
    ).strip()

    if not bucket:
        raise RuntimeError("SUPABASE_STORAGE_BUCKET is missing.")

    return bucket


def upload_to_supabase(storage_name, file_bytes, mime_type):
    """Upload a new image to the private Supabase bucket."""
    return (
        get_supabase_client()
        .storage
        .from_(get_storage_bucket())
        .upload(
            storage_name,
            file_bytes,
            {
                "content-type": mime_type,
                "upsert": "false",
            },
        )
    )


def delete_from_supabase(storage_name):
    """Delete an uploaded object when a database transaction fails."""
    return (
        get_supabase_client()
        .storage
        .from_(get_storage_bucket())
        .remove([storage_name])
    )


def download_from_supabase(storage_name):
    """Download an image from the private bucket."""
    return (
        get_supabase_client()
        .storage
        .from_(get_storage_bucket())
        .download(storage_name)
    )


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

RETURN_REASONS = [
    "Transit Damage",
    "Manufacturing Defect",
    "Wrong Item Received",
    "Size Issue",
    "Customer Expectation",
    "Not Working",
    "Missing Parts",
    "Other",
]

ANALYSIS_STATUSES = [
    "Pending",
    "Processing",
    "Analyzed",
    "Failed",
]

EVIDENCE_LEVELS = [
    "Strong",
    "Weak",
    "Insufficient",
]

REVIEW_STATUSES = [
    "Pending",
    "Reviewed",
    "Corrected",
]

IMAGE_FORMATS = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
}

# One AI job at a time for local development.
# Multi-worker production deployments should use a task queue.
ai_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------
# AI ANALYSIS DATABASE MODEL
# ---------------------------------------------------------

class ReturnAnalysis(db.Model):
    __tablename__ = "return_analyses"

    id = db.Column(db.Integer, primary_key=True)

    return_request_id = db.Column(
        db.Integer,
        db.ForeignKey("return_requests.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    # AI-generated fields
    predicted_reason = db.Column(db.String(100), nullable=True)
    evidence_strength = db.Column(db.String(30), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    supporting_evidence = db.Column(db.Text, nullable=True)
    conflicting_evidence = db.Column(db.Text, nullable=True)
    analysis_notes = db.Column(db.Text, nullable=True)
    model_version = db.Column(db.String(100), nullable=True)

    # Staff-reviewed fields
    verified_reason = db.Column(db.String(100), nullable=True)

    review_status = db.Column(
        db.String(30),
        nullable=False,
        default="Pending",
    )

    reviewer_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=True,
    )

    reviewer_notes = db.Column(db.Text, nullable=True)

    reviewed_at = db.Column(
        db.DateTime(timezone=True),
        nullable=True,
    )

    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    return_request = db.relationship(
        "ReturnRequest",
        backref=db.backref("analysis", uselist=False),
    )

    reviewer = db.relationship(
        "User",
        foreign_keys=[reviewer_id],
    )


# ---------------------------------------------------------
# APPLICATION FACTORY
# ---------------------------------------------------------

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    if not app.config.get("SECRET_KEY"):
        raise RuntimeError(
            "SECRET_KEY is missing. Set it in your .env file."
        )

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    login_manager.login_view = "login"
    login_manager.login_message_category = "warning"

    # -----------------------------------------------------
    # LOGIN USER LOADER
    # -----------------------------------------------------

    @login_manager.user_loader
    def load_user(user_id):
        try:
            return db.session.get(User, int(user_id))
        except (ValueError, TypeError):
            return None

    # -----------------------------------------------------
    # ROLE PROTECTION
    # -----------------------------------------------------

    def role_required(role):
        def decorator(view_function):
            @wraps(view_function)
            @login_required
            def wrapped_view(*args, **kwargs):
                if current_user.role != role:
                    flash(
                        "You are not authorized to access that page.",
                        "danger",
                    )

                    if current_user.role == "admin":
                        return redirect(url_for("admin_dashboard"))

                    return redirect(url_for("user_home"))

                return view_function(*args, **kwargs)

            return wrapped_view

        return decorator

    # -----------------------------------------------------
    # COMMON REDIRECT
    # -----------------------------------------------------

    def dashboard_redirect():
        if current_user.is_authenticated:
            if current_user.role == "admin":
                return redirect(url_for("admin_dashboard"))

            return redirect(url_for("user_home"))

        return redirect(url_for("login"))

    # -----------------------------------------------------
    # IMAGE VALIDATION AND STORAGE
    # -----------------------------------------------------

    def save_image(upload, image_type):
        """
        Validate the uploaded image, save a local cache copy,
        upload it to Supabase, and create its database record.

        The caller commits the database transaction.
        """
        if not upload or not upload.filename:
            raise ValueError("One of the required images is missing.")

        original_name = secure_filename(upload.filename)

        if not original_name:
            raise ValueError("The uploaded filename is invalid.")

        extension = Path(original_name).suffix.lower().lstrip(".")
        allowed_extensions = current_app.config[
            "ALLOWED_IMAGE_EXTENSIONS"
        ]

        if extension not in allowed_extensions:
            raise ValueError(
                "Unsupported image extension. Use JPG, PNG, or WEBP."
            )

        try:
            upload.stream.seek(0, os.SEEK_END)
            file_size = upload.stream.tell()
            upload.stream.seek(0)
        except (OSError, ValueError):
            raise ValueError(
                "The uploaded image could not be read."
            ) from None

        if file_size == 0:
            raise ValueError("An uploaded image is empty.")

        if file_size > current_app.config["MAX_IMAGE_SIZE"]:
            raise ValueError("Each image must be 8 MB or smaller.")

        try:
            with Image.open(upload.stream) as image:
                image.verify()

            upload.stream.seek(0)

            with Image.open(upload.stream) as image:
                image_format = image.format

        except (UnidentifiedImageError, OSError, ValueError):
            raise ValueError(
                "An uploaded file is not a valid image."
            ) from None

        if image_format not in IMAGE_FORMATS:
            raise ValueError(
                "Only JPEG, PNG, and WEBP images are accepted."
            )

        extension, mime_type = IMAGE_FORMATS[image_format]
        storage_name = f"{uuid.uuid4().hex}.{extension}"

        upload_folder = current_app.config["UPLOAD_FOLDER"]
        destination = os.path.join(upload_folder, storage_name)

        try:
            # Save a local copy first. The AI model can use this copy
            # immediately without needing to download it again.
            upload.stream.seek(0)
            upload.save(destination)

            with open(destination, "rb") as image_file:
                image_bytes = image_file.read()

            # Persistent storage in Supabase.
            upload_to_supabase(
                storage_name,
                image_bytes,
                mime_type,
            )

        except Exception as error:
            # If storage upload failed, do not leave an incomplete
            # image that looks like a successfully saved submission.
            try:
                if os.path.exists(destination):
                    os.remove(destination)
            except OSError:
                current_app.logger.warning(
                    "Could not remove incomplete image: %s",
                    destination,
                )

            current_app.logger.exception(
                "Could not save image %s to Supabase Storage.",
                storage_name,
            )

            raise ValueError(
                "The image could not be saved to cloud storage. "
                "Please try again."
            ) from error

        image_record = ReturnImage(
            image_type=image_type,
            storage_name=storage_name,
            original_filename=original_name[:255],
            mime_type=mime_type,
        )

        return image_record, destination

    def remove_saved_files(paths):
        """
        Roll back local cache files and Supabase objects when
        a return request's database transaction fails.
        """
        for path in paths:
            storage_name = os.path.basename(path)

            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                current_app.logger.warning(
                    "Could not remove local image: %s",
                    path,
                )

            try:
                delete_from_supabase(storage_name)
            except Exception:
                # Do not hide the original database error.
                current_app.logger.exception(
                    "Could not roll back Supabase image %s.",
                    storage_name,
                )

    # -----------------------------------------------------
    # SIMILAR STAFF-VERIFIED CASE RETRIEVAL
    # -----------------------------------------------------

    def retrieve_similar_verified_cases(return_request, limit=3):
        """
        Retrieve earlier staff-verified cases as contextual examples.
        Exclude the current case and cases created after it.
        """
        if return_request is None:
            return []

        query = (
            select(ReturnRequest, ReturnAnalysis)
            .join(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .where(
                ReturnRequest.id != return_request.id,
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(["Reviewed", "Corrected"]),
            )
        )

        current_created_at = getattr(return_request, "created_at", None)

        if current_created_at is not None:
            query = query.where(
                ReturnRequest.created_at < current_created_at
            )

        candidates = db.session.execute(
            query.order_by(
                ReturnRequest.created_at.desc()
            ).limit(300)
        ).all()

        def tokens(value):
            return {
                word
                for word in re.findall(
                    r"[a-z0-9]+",
                    (value or "").lower(),
                )
                if len(word) > 2
            }

        current_product = tokens(
            getattr(return_request, "product_name", "")
        )
        current_comment = tokens(
            getattr(return_request, "reason_comment", "")
        )
        current_reason = (
            getattr(return_request, "selected_reason", "") or ""
        ).strip()

        ranked = []

        for candidate_request, candidate_analysis in candidates:
            past_product = tokens(candidate_request.product_name or "")
            past_comment = tokens(candidate_request.reason_comment or "")

            same_selected_reason = (
                current_reason
                and current_reason == (candidate_request.selected_reason or "")
            )

            product_union = current_product | past_product
            comment_union = current_comment | past_comment

            product_overlap = (
                len(current_product & past_product) / len(product_union)
                if product_union else 0.0
            )

            comment_overlap = (
                len(current_comment & past_comment) / len(comment_union)
                if comment_union else 0.0
            )

            score = (
                product_overlap * 0.50
                + comment_overlap * 0.35
            )

            if same_selected_reason:
                score += 0.15

            if score <= 0:
                continue

            ranked.append(
                (score, candidate_request, candidate_analysis)
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].created_at,
            ),
            reverse=True,
        )

        similar_cases = []

        for score, past_request, past_analysis in ranked[:limit]:
            similar_cases.append({
                "case_id": past_request.id,
                "product_name": (past_request.product_name or "")[:200],
                "customer_selected_reason": (
                    past_request.selected_reason or ""
                ),
                "customer_comment": (
                    past_request.reason_comment or ""
                )[:700],
                "staff_verified_reason": (
                    past_analysis.verified_reason or ""
                ),
                "review_status": past_analysis.review_status or "",
                "reviewer_notes": (
                    past_analysis.reviewer_notes or ""
                )[:500],
                "similarity_score": round(score, 3),
            })

        return similar_cases

    # -----------------------------------------------------
    # BACKGROUND AI PROCESSING
    # -----------------------------------------------------

    def process_return_analysis(return_id):
        """Run AI analysis in a separate application context."""

        with app.app_context():
            try:
                return_request = db.session.get(
                    ReturnRequest,
                    return_id,
                )

                if return_request is None:
                    app.logger.error(
                        "Return case %s was not found.",
                        return_id,
                    )
                    return

                if return_request.analysis_status != "Processing":
                    return

                analysis = db.session.scalar(
                    select(ReturnAnalysis).where(
                        ReturnAnalysis.return_request_id == return_id
                    )
                )

                if analysis is None:
                    analysis = ReturnAnalysis(
                        return_request_id=return_id,
                    )
                    db.session.add(analysis)
                    db.session.commit()

                similar_cases = retrieve_similar_verified_cases(
                    return_request
                )

                result = analyze_return_request(
                    return_request=return_request,
                    upload_folder=app.config["UPLOAD_FOLDER"],
                    return_reasons=RETURN_REASONS,
                    similar_cases=similar_cases,
                )

                predicted_reason = result.get("predicted_reason")
                evidence_strength = result.get("evidence_strength")

                if predicted_reason not in RETURN_REASONS:
                    raise ValueError(
                        "AI returned an unsupported predicted reason."
                    )

                if evidence_strength not in EVIDENCE_LEVELS:
                    raise ValueError(
                        "AI returned an invalid evidence level."
                    )

                confidence = float(result.get("confidence"))

                if not 0 <= confidence <= 100:
                    raise ValueError(
                        "AI confidence must be between 0 and 100."
                    )

                analysis.predicted_reason = predicted_reason
                analysis.evidence_strength = evidence_strength
                analysis.confidence = confidence
                analysis.supporting_evidence = result.get(
                    "supporting_evidence", ""
                )
                analysis.conflicting_evidence = result.get(
                    "conflicting_evidence", ""
                )
                analysis.analysis_notes = result.get(
                    "analysis_notes", ""
                )
                analysis.model_version = result.get(
                    "model_version", "Unknown"
                )

                # Do not overwrite the customer's selected reason
                # or the staff's verified reason.
                return_request.analysis_status = "Analyzed"
                db.session.commit()

                app.logger.info(
                    "AI analysis completed for return case %s.",
                    return_id,
                )

            except Exception:
                db.session.rollback()

                app.logger.exception(
                    "AI analysis failed for return case %s.",
                    return_id,
                )

                try:
                    return_request = db.session.get(
                        ReturnRequest,
                        return_id,
                    )

                    if return_request is not None:
                        analysis = db.session.scalar(
                            select(ReturnAnalysis).where(
                                ReturnAnalysis.return_request_id
                                == return_id
                            )
                        )

                        if analysis is None:
                            analysis = ReturnAnalysis(
                                return_request_id=return_id,
                            )
                            db.session.add(analysis)

                        return_request.analysis_status = "Failed"

                        analysis.predicted_reason = None
                        analysis.evidence_strength = None
                        analysis.confidence = None
                        analysis.supporting_evidence = None
                        analysis.conflicting_evidence = None
                        analysis.model_version = None
                        analysis.analysis_notes = (
                            "AI analysis failed. No successful result "
                            "is available for this attempt. Check the "
                            "server terminal and retry the analysis."
                        )

                        db.session.commit()

                except Exception:
                    db.session.rollback()

                    app.logger.exception(
                        "Could not save the failure status for case %s.",
                        return_id,
                    )

            finally:
                db.session.remove()

    def queue_return_analysis(return_id):
        """Mark a case as processing and submit its AI job."""

        return_request = db.session.get(ReturnRequest, return_id)

        if return_request is None:
            return False

        if return_request.analysis_status == "Processing":
            return False

        try:
            return_request.analysis_status = "Processing"
            db.session.commit()

            ai_executor.submit(
                process_return_analysis,
                return_id,
            )

            app.logger.info(
                "AI analysis queued for return case %s.",
                return_id,
            )

            return True

        except Exception:
            db.session.rollback()

            app.logger.exception(
                "Could not queue AI analysis for case %s.",
                return_id,
            )

            try:
                return_request = db.session.get(
                    ReturnRequest,
                    return_id,
                )

                if return_request is not None:
                    return_request.analysis_status = "Failed"

                    analysis = db.session.scalar(
                        select(ReturnAnalysis).where(
                            ReturnAnalysis.return_request_id == return_id
                        )
                    )

                    if analysis is None:
                        analysis = ReturnAnalysis(
                            return_request_id=return_id,
                        )
                        db.session.add(analysis)

                    analysis.analysis_notes = (
                        "AI analysis could not be queued. "
                        "An administrator can retry the analysis."
                    )

                    db.session.commit()

            except Exception:
                db.session.rollback()

                app.logger.exception(
                    "Could not save queue failure for case %s.",
                    return_id,
                )

            return False

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_request(error):
        flash(
            "The total upload is too large. Please reduce the image "
            "sizes and try again.",
            "danger",
        )
        return dashboard_redirect()

    # -----------------------------------------------------
    # HOME
    # -----------------------------------------------------

    @app.route("/")
    def index():
        return dashboard_redirect()

    # -----------------------------------------------------
    # LOGIN
    # -----------------------------------------------------

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return dashboard_redirect()

        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")

            selected_role = request.form.get(
                "login_role",
                request.form.get("role", ""),
            )

            if selected_role not in ("admin", "user"):
                flash(
                    "Please select Login as User or Login as Admin.",
                    "danger",
                )
                return render_template("login.html")

            if not username or not password:
                flash(
                    "Please enter both username and password.",
                    "danger",
                )
                return render_template("login.html")

            user = db.session.scalar(
                select(User).where(User.username == username)
            )

            if (
                user is not None
                and user.check_password(password)
                and user.role == selected_role
            ):
                login_user(user)
                flash("Login successful.", "success")
                return dashboard_redirect()

            flash(
                "Invalid username, password, or selected account type.",
                "danger",
            )

        return render_template("login.html")

    # -----------------------------------------------------
    # ADMIN DASHBOARD
    # -----------------------------------------------------

    @app.route("/admin")
    @role_required("admin")
    def admin_dashboard():
        search_text = request.args.get("q", "").strip()[:200]

        selected_reason = request.args.get("reason", "").strip()
        selected_status = request.args.get("status", "").strip()
        selected_evidence = request.args.get("evidence", "").strip()
        selected_review = request.args.get("review", "").strip()
        selected_mismatch = request.args.get("mismatch", "").strip()

        total_requests = db.session.scalar(
            select(func.count()).select_from(ReturnRequest)
        ) or 0

        status_counts = {}

        for status in ANALYSIS_STATUSES:
            count_key = status.lower() + "_count"

            status_counts[count_key] = db.session.scalar(
                select(func.count())
                .select_from(ReturnRequest)
                .where(ReturnRequest.analysis_status == status)
            ) or 0

        customer_count = db.session.scalar(
            select(func.count(func.distinct(ReturnRequest.user_id)))
        ) or 0

        product_count = db.session.scalar(
            select(func.count(func.distinct(ReturnRequest.product_id)))
        ) or 0

        # Customer-selected return reason counts.
        customer_reason_counts = {
            reason: 0 for reason in RETURN_REASONS
        }

        reason_rows = db.session.execute(
            select(
                ReturnRequest.selected_reason,
                func.count(ReturnRequest.id),
            )
            .group_by(ReturnRequest.selected_reason)
        ).all()

        for reason, count in reason_rows:
            if reason in customer_reason_counts:
                customer_reason_counts[reason] = count

        reason_labels = list(RETURN_REASONS)
        reason_values = [
            customer_reason_counts[reason]
            for reason in reason_labels
        ]

        # Evidence strength counts.
        evidence_counts = {}

        for level in EVIDENCE_LEVELS:
            count_key = level.lower() + "_evidence_count"

            evidence_counts[count_key] = db.session.scalar(
                select(func.count())
                .select_from(ReturnAnalysis)
                .where(ReturnAnalysis.evidence_strength == level)
            ) or 0

        # Staff review counts.
        reviewed_count = db.session.scalar(
            select(func.count())
            .select_from(ReturnAnalysis)
            .where(
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                )
            )
        ) or 0

        corrected_count = db.session.scalar(
            select(func.count())
            .select_from(ReturnAnalysis)
            .where(ReturnAnalysis.review_status == "Corrected")
        ) or 0

        # Customer versus staff reason comparison.
        comparison_labels = list(RETURN_REASONS)

        staff_reason_counts = {
            reason: 0 for reason in comparison_labels
        }

        confirmed_match_rows = db.session.execute(
            select(
                ReturnRequest.selected_reason,
                func.count(ReturnRequest.id),
            )
            .join(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .where(
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnRequest.selected_reason
                == ReturnAnalysis.verified_reason,
            )
            .group_by(ReturnRequest.selected_reason)
        ).all()

        for reason, count in confirmed_match_rows:
            if reason in staff_reason_counts:
                staff_reason_counts[reason] = count

        customer_comparison_values = list(reason_values)

        staff_comparison_values = [
            staff_reason_counts[reason]
            for reason in comparison_labels
        ]

        # Customer and staff reason mismatch count.
        mismatch_count = db.session.scalar(
            select(func.count())
            .select_from(ReturnRequest)
            .join(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .where(
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnRequest.selected_reason
                != ReturnAnalysis.verified_reason,
            )
        ) or 0

        # Return cases, ordered to prioritize weak evidence.
        cases_query = (
            select(ReturnRequest)
            .outerjoin(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .order_by(
                case(
                    (ReturnAnalysis.evidence_strength == "Weak", 0),
                    (
                        ReturnAnalysis.evidence_strength == "Insufficient",
                        1,
                    ),
                    else_=2,
                ),
                ReturnRequest.created_at.desc(),
                ReturnRequest.id.desc(),
            )
        )

        if search_text:
            pattern = f"%{search_text}%"

            cases_query = cases_query.where(
                or_(
                    ReturnRequest.product_id.ilike(pattern),
                    ReturnRequest.product_name.ilike(pattern),
                    ReturnRequest.reason_comment.ilike(pattern),
                )
            )

        if selected_reason in RETURN_REASONS:
            cases_query = cases_query.where(
                ReturnRequest.selected_reason == selected_reason
            )
        else:
            selected_reason = ""

        if selected_status in ANALYSIS_STATUSES:
            cases_query = cases_query.where(
                ReturnRequest.analysis_status == selected_status
            )
        else:
            selected_status = ""

        if selected_evidence in EVIDENCE_LEVELS:
            cases_query = cases_query.where(
                ReturnAnalysis.evidence_strength == selected_evidence
            )
        else:
            selected_evidence = ""

        if selected_review in REVIEW_STATUSES:
            cases_query = cases_query.where(
                ReturnAnalysis.review_status == selected_review
            )
        else:
            selected_review = ""

        if selected_mismatch == "yes":
            cases_query = cases_query.where(
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnRequest.selected_reason
                != ReturnAnalysis.verified_reason,
            )

        elif selected_mismatch == "no":
            cases_query = cases_query.where(
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnRequest.selected_reason
                == ReturnAnalysis.verified_reason,
            )
        else:
            selected_mismatch = ""

        cases = db.session.scalars(
            cases_query.limit(200)
        ).all()

        return render_template(
            "admin_dashboard.html",
            total_requests=total_requests,
            pending_count=status_counts["pending_count"],
            processing_count=status_counts["processing_count"],
            analyzed_count=status_counts["analyzed_count"],
            failed_count=status_counts["failed_count"],
            customer_count=customer_count,
            product_count=product_count,
            reason_labels=reason_labels,
            reason_values=reason_values,
            reason_counts=reason_values,
            comparison_labels=comparison_labels,
            customer_comparison_values=customer_comparison_values,
            staff_comparison_values=staff_comparison_values,
            cases=cases,
            return_reasons=RETURN_REASONS,
            analysis_statuses=ANALYSIS_STATUSES,
            search_text=search_text,
            selected_reason=selected_reason,
            selected_status=selected_status,
            evidence_levels=EVIDENCE_LEVELS,
            review_statuses=REVIEW_STATUSES,
            selected_evidence=selected_evidence,
            selected_review=selected_review,
            selected_mismatch=selected_mismatch,
            strong_evidence_count=evidence_counts[
                "strong_evidence_count"
            ],
            weak_evidence_count=evidence_counts[
                "weak_evidence_count"
            ],
            insufficient_evidence_count=evidence_counts[
                "insufficient_evidence_count"
            ],
            reviewed_count=reviewed_count,
            corrected_count=corrected_count,
            mismatch_count=mismatch_count,
        )

    # -----------------------------------------------------
    # QUALITY REPORT
    # -----------------------------------------------------

    @app.route("/admin/quality-report")
    @role_required("admin")
    def quality_report():
        total_requests = db.session.scalar(
            select(func.count()).select_from(ReturnRequest)
        ) or 0

        analyzed_requests = db.session.scalar(
            select(func.count())
            .select_from(ReturnRequest)
            .where(ReturnRequest.analysis_status == "Analyzed")
        ) or 0

        pending_requests = db.session.scalar(
            select(func.count())
            .select_from(ReturnRequest)
            .where(ReturnRequest.analysis_status == "Pending")
        ) or 0

        failed_requests = db.session.scalar(
            select(func.count())
            .select_from(ReturnRequest)
            .where(ReturnRequest.analysis_status == "Failed")
        ) or 0

        evidence_rows = db.session.execute(
            select(
                ReturnAnalysis.evidence_strength,
                func.count(ReturnAnalysis.id),
            )
            .group_by(ReturnAnalysis.evidence_strength)
        ).all()

        evidence_counts = {
            "Strong": 0,
            "Weak": 0,
            "Insufficient": 0,
            "Not assessed": 0,
        }

        for evidence, count in evidence_rows:
            if evidence in evidence_counts:
                evidence_counts[evidence] = count
            else:
                evidence_counts["Not assessed"] += count

        reviewed_count = db.session.scalar(
            select(func.count())
            .select_from(ReturnAnalysis)
            .where(
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnAnalysis.verified_reason.is_not(None),
            )
        ) or 0

        corrected_count = db.session.scalar(
            select(func.count())
            .select_from(ReturnAnalysis)
            .where(
                ReturnAnalysis.review_status == "Corrected",
                ReturnAnalysis.verified_reason.is_not(None),
            )
        ) or 0

        verified_records = db.session.execute(
            select(
                ReturnRequest.selected_reason,
                ReturnAnalysis.predicted_reason,
                ReturnAnalysis.verified_reason,
            )
            .join(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .where(
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
            )
        ).all()

        comparable_records = [
            record
            for record in verified_records
            if record.predicted_reason
        ]

        prediction_matches = sum(
            1 for record in comparable_records
            if record.predicted_reason == record.verified_reason
        )

        prediction_mismatches = (
            len(comparable_records) - prediction_matches
        )

        prediction_accuracy = (
            round(
                prediction_matches / len(comparable_records) * 100,
                1,
            )
            if comparable_records else None
        )

        customer_matches = sum(
            1 for record in verified_records
            if record.selected_reason == record.verified_reason
        )

        customer_mismatches = len(verified_records) - customer_matches

        customer_match_rate = (
            round(
                customer_matches / len(verified_records) * 100,
                1,
            )
            if verified_records else None
        )

        reason_rows = db.session.execute(
            select(
                ReturnRequest.selected_reason,
                func.count(ReturnRequest.id),
            )
            .group_by(ReturnRequest.selected_reason)
            .order_by(func.count(ReturnRequest.id).desc())
        ).all()

        reason_labels = [
            reason or "Not specified"
            for reason, count in reason_rows
        ]

        reason_values = [count for reason, count in reason_rows]

        status_rows = db.session.execute(
            select(
                ReturnRequest.analysis_status,
                func.count(ReturnRequest.id),
            )
            .group_by(ReturnRequest.analysis_status)
        ).all()

        status_labels = [
            status or "Unknown"
            for status, count in status_rows
        ]

        status_values = [count for status, count in status_rows]

        mismatch_cases = db.session.scalars(
            select(ReturnRequest)
            .join(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .where(
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnRequest.selected_reason
                != ReturnAnalysis.verified_reason,
            )
            .order_by(
                ReturnAnalysis.reviewed_at.desc(),
                ReturnRequest.created_at.desc(),
            )
            .limit(20)
        ).all()

        ai_mismatch_cases = db.session.scalars(
            select(ReturnRequest)
            .join(
                ReturnAnalysis,
                ReturnAnalysis.return_request_id == ReturnRequest.id,
            )
            .where(
                ReturnAnalysis.predicted_reason.is_not(None),
                ReturnAnalysis.verified_reason.is_not(None),
                ReturnAnalysis.review_status.in_(
                    ["Reviewed", "Corrected"]
                ),
                ReturnAnalysis.predicted_reason
                != ReturnAnalysis.verified_reason,
            )
            .order_by(
                ReturnAnalysis.reviewed_at.desc(),
                ReturnRequest.created_at.desc(),
            )
            .limit(20)
        ).all()

        return render_template(
            "quality_report.html",
            total_requests=total_requests,
            analyzed_requests=analyzed_requests,
            pending_requests=pending_requests,
            failed_requests=failed_requests,
            evidence_counts=evidence_counts,
            reviewed_count=reviewed_count,
            corrected_count=corrected_count,
            verified_count=len(verified_records),
            comparable_count=len(comparable_records),
            prediction_matches=prediction_matches,
            prediction_mismatches=prediction_mismatches,
            prediction_accuracy=prediction_accuracy,
            customer_matches=customer_matches,
            customer_mismatches=customer_mismatches,
            customer_match_rate=customer_match_rate,
            reason_labels=reason_labels,
            reason_values=reason_values,
            status_labels=status_labels,
            status_values=status_values,
            mismatch_cases=mismatch_cases,
            ai_mismatch_cases=ai_mismatch_cases,
        )

    # -----------------------------------------------------
    # RETURN CASE DETAILS
    # -----------------------------------------------------

    @app.route("/admin/returns/<int:return_id>")
    @role_required("admin")
    def admin_return_details(return_id):
        return_request = db.session.get(ReturnRequest, return_id)

        if return_request is None:
            abort(404)

        analysis = db.session.scalar(
            select(ReturnAnalysis).where(
                ReturnAnalysis.return_request_id == return_id
            )
        )

        listing_images = [
            image for image in return_request.images
            if image.image_type == "listing"
        ]

        received_images = [
            image for image in return_request.images
            if image.image_type == "received"
        ]

        return render_template(
            "admin_return_details.html",
            return_request=return_request,
            listing_images=listing_images,
            received_images=received_images,
            analysis=analysis,
            return_reasons=RETURN_REASONS,
            evidence_levels=EVIDENCE_LEVELS,
        )

    # -----------------------------------------------------
    # DELETE RETURN CASE
    # -----------------------------------------------------

    @app.route(
        "/admin/returns/<int:return_id>/delete",
        methods=["POST"],
    )
    @role_required("admin")
    def delete_return_case(return_id):
        """Delete one return case and clean up its associated images."""

        return_request = db.session.get(ReturnRequest, return_id)

        if return_request is None:
            abort(404)

        # Never delete a case while its background AI worker is using it.
        # if return_request.analysis_status == "Processing":
        #     flash(
        #         "This case cannot be deleted while AI analysis is processing. "
        #         "Wait for processing to finish, then try again.",
        #         "warning",
        #     )
        #     return redirect(
        #         url_for("admin_return_details", return_id=return_id)
        #     )

        # Capture storage names before deleting the related image rows.
        image_records = list(return_request.images)
        storage_names = []

        upload_folder = os.path.abspath(
            current_app.config["UPLOAD_FOLDER"]
        )
        local_paths = []

        for image_record in image_records:
            storage_name = (image_record.storage_name or "").strip()

            # Storage names are generated by this application. Validate them
            # before using them for either local paths or cloud deletion.
            if (
                not storage_name
                or storage_name != os.path.basename(storage_name)
                or storage_name in (".", "..")
            ):
                current_app.logger.warning(
                    "Skipping invalid storage name on image record %s "
                    "while deleting return case %s.",
                    image_record.id,
                    return_id,
                )
                continue

            storage_names.append(storage_name)
            local_path = os.path.abspath(
                os.path.join(upload_folder, storage_name)
            )

            if os.path.commonpath([upload_folder, local_path]) == upload_folder:
                local_paths.append(local_path)

        try:
            # ReturnAnalysis has its own FK and no delete cascade, so remove
            # it explicitly before deleting the parent return request.
            analysis = db.session.scalar(
                select(ReturnAnalysis).where(
                    ReturnAnalysis.return_request_id == return_id
                )
            )

            if analysis is not None:
                db.session.delete(analysis)

            # ReturnRequest.images is configured with delete-orphan cascade.
            db.session.delete(return_request)
            db.session.commit()

        except Exception:
            db.session.rollback()
            current_app.logger.exception(
                "Could not delete return case %s from the database.",
                return_id,
            )
            flash(
                "The return case could not be deleted. No stored images "
                "were intentionally removed; please try again.",
                "danger",
            )
            return redirect(
                url_for("admin_return_details", return_id=return_id)
            )

        # The database deletion is committed. Clean up local cache files and
        # cloud objects separately; storage failures must not undo the DB work.
        cleanup_failed = False

        for local_path in local_paths:
            try:
                if os.path.isfile(local_path):
                    os.remove(local_path)
            except OSError:
                cleanup_failed = True
                current_app.logger.exception(
                    "Could not remove local image cache file for deleted "
                    "return case %s: %s",
                    return_id,
                    local_path,
                )

        for storage_name in storage_names:
            try:
                delete_from_supabase(storage_name)
            except Exception:
                cleanup_failed = True
                current_app.logger.exception(
                    "Could not delete Supabase object %s for return case %s.",
                    storage_name,
                    return_id,
                )

        if cleanup_failed:
            flash(
                f"Return case #{return_id} and its database records were "
                "deleted, but one or more stored images could not be cleaned "
                "up. Check the server logs and Supabase Storage.",
                "warning",
            )
        else:
            flash(
                f"Return case #{return_id}, its AI/staff analysis, and "
                "associated images were deleted successfully.",
                "success",
            )

        return redirect(url_for("admin_dashboard"))

    # -----------------------------------------------------
    # STAFF REVIEW
    # -----------------------------------------------------

    @app.route(
        "/admin/returns/<int:return_id>/review",
        methods=["POST"],
    )
    @role_required("admin")
    def review_return(return_id):
        return_request = db.session.get(ReturnRequest, return_id)

        if return_request is None:
            abort(404)

        verified_reason = request.form.get(
            "verified_reason", ""
        ).strip()

        reviewer_notes = request.form.get(
            "reviewer_notes", ""
        ).strip()

        if verified_reason not in RETURN_REASONS:
            flash(
                "Select a valid staff-confirmed reason.",
                "danger",
            )
            return redirect(
                url_for(
                    "admin_return_details",
                    return_id=return_id,
                )
            )

        if len(reviewer_notes) > 3000:
            flash(
                "Review notes must be 3000 characters or fewer.",
                "danger",
            )
            return redirect(
                url_for(
                    "admin_return_details",
                    return_id=return_id,
                )
            )

        try:
            analysis = db.session.scalar(
                select(ReturnAnalysis).where(
                    ReturnAnalysis.return_request_id == return_id
                )
            )

            if analysis is None:
                analysis = ReturnAnalysis(
                    return_request_id=return_id,
                )
                db.session.add(analysis)

            analysis.verified_reason = verified_reason
            analysis.reviewer_notes = reviewer_notes
            analysis.reviewer_id = current_user.id
            analysis.reviewed_at = datetime.now(timezone.utc)

            analysis.review_status = (
                "Corrected"
                if verified_reason != return_request.selected_reason
                else "Reviewed"
            )

            db.session.commit()
            flash("Staff review saved successfully.", "success")

        except Exception:
            db.session.rollback()

            current_app.logger.exception(
                "Could not save staff review for case %s.",
                return_id,
            )

            flash(
                "The review could not be saved. Please try again.",
                "danger",
            )

        return redirect(
            url_for(
                "admin_return_details",
                return_id=return_id,
            )
        )

    # -----------------------------------------------------
    # MANUAL AI RETRY
    # -----------------------------------------------------

    @app.route(
        "/admin/returns/<int:return_id>/analyze",
        methods=["POST"],
    )
    @role_required("admin")
    def analyze_return(return_id):
        return_request = db.session.get(ReturnRequest, return_id)

        if return_request is None:
            abort(404)

        if return_request.analysis_status == "Processing":
            flash(
                "This return is already being processed.",
                "warning",
            )
            return redirect(
                url_for(
                    "admin_return_details",
                    return_id=return_id,
                )
            )

        queued = queue_return_analysis(return_id)

        if queued:
            flash(
                "AI analysis has been queued. Results will appear "
                "when background processing finishes.",
                "success",
            )
        else:
            flash(
                "AI analysis could not be queued. Check the server "
                "terminal for details.",
                "danger",
            )

        return redirect(
            url_for(
                "admin_return_details",
                return_id=return_id,
            )
        )

    # -----------------------------------------------------
    # CUSTOMER RETURN PORTAL
    # -----------------------------------------------------

    @app.route("/user", methods=["GET", "POST"])
    @role_required("user")
    def user_home():
        if request.method == "POST":
            product_id = request.form.get(
                "product_id", ""
            ).strip()

            product_name = request.form.get(
                "product_name", ""
            ).strip()

            selected_reason = request.form.get(
                "selected_reason", ""
            ).strip()

            reason_comment = request.form.get(
                "reason_comment", ""
            ).strip()

            listing_image = request.files.get("listing_image")

            received_images = [
                image
                for image in request.files.getlist("received_images")
                if image and image.filename
            ]

            if (
                not product_id
                or not product_name
                or len(product_id) > 100
                or len(product_name) > 200
            ):
                flash(
                    "Enter a valid product ID and product name.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            if selected_reason not in RETURN_REASONS:
                flash(
                    "Please select a valid return reason.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            if not reason_comment or len(reason_comment) > 3000:
                flash(
                    "Enter a comment of up to 3000 characters.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            if not listing_image or not listing_image.filename:
                flash(
                    "Please upload the product listing image.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            if not received_images:
                flash(
                    "Upload at least one received-product image.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            max_received_images = current_app.config[
                "MAX_RECEIVED_IMAGES"
            ]

            if len(received_images) > max_received_images:
                flash(
                    f"Upload no more than {max_received_images} "
                    "received-product images.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            saved_paths = []

            try:
                return_request = ReturnRequest(
                    user_id=current_user.id,
                    product_id=product_id,
                    product_name=product_name,
                    selected_reason=selected_reason,
                    reason_comment=reason_comment,
                    analysis_status="Pending",
                )

                db.session.add(return_request)
                db.session.flush()

                all_images = [(listing_image, "listing")] + [
                    (image, "received")
                    for image in received_images
                ]

                for upload, image_type in all_images:
                    image_record, saved_path = save_image(
                        upload,
                        image_type,
                    )

                    saved_paths.append(saved_path)
                    image_record.return_request = return_request
                    db.session.add(image_record)

                db.session.commit()

            except ValueError as error:
                db.session.rollback()
                remove_saved_files(saved_paths)

                flash(str(error), "danger")
                return redirect(url_for("user_home"))

            except Exception:
                db.session.rollback()
                remove_saved_files(saved_paths)

                current_app.logger.exception(
                    "Failed to save a return request."
                )

                flash(
                    "The request could not be saved. Please try again.",
                    "danger",
                )
                return redirect(url_for("user_home"))

            return_id = return_request.id
            queued = queue_return_analysis(return_id)

            if queued:
                flash(
                    f"Return request #{return_id} was submitted. "
                    "AI evidence analysis has started in the background.",
                    "success",
                )
            else:
                flash(
                    f"Return request #{return_id} was saved, but AI "
                    "analysis could not be queued. An administrator "
                    "can retry it.",
                    "warning",
                )

            return redirect(url_for("user_home"))

        my_requests = db.session.scalars(
            select(ReturnRequest)
            .where(ReturnRequest.user_id == current_user.id)
            .order_by(
                ReturnRequest.created_at.desc(),
                ReturnRequest.id.desc(),
            )
        ).all()

        return render_template(
            "user_home.html",
            return_reasons=RETURN_REASONS,
            my_requests=my_requests,
        )

    # -----------------------------------------------------
    # SECURE IMAGE VIEWING
    # -----------------------------------------------------

    @app.route("/return-images/<int:image_id>")
    @login_required
    def view_return_image(image_id):
        image_record = db.session.get(ReturnImage, image_id)

        if image_record is None:
            abort(404)

        # Customers may only access images belonging to their own cases.
        if (
            current_user.role != "admin"
            and image_record.return_request.user_id != current_user.id
        ):
            abort(403)

        storage_name = image_record.storage_name or ""

        # Do not allow paths to be supplied through the database field.
        if (
            not storage_name
            or storage_name != os.path.basename(storage_name)
            or storage_name in (".", "..")
        ):
            abort(404)

        upload_folder = os.path.abspath(
            current_app.config["UPLOAD_FOLDER"]
        )

        local_path = os.path.abspath(
            os.path.join(upload_folder, storage_name)
        )

        if os.path.commonpath([upload_folder, local_path]) != upload_folder:
            abort(404)

        try:
            if os.path.isfile(local_path):
                response = send_from_directory(
                    upload_folder,
                    storage_name,
                    mimetype=image_record.mime_type,
                    as_attachment=False,
                )
            else:
                # The image may be missing locally on another machine
                # or after a deployment. Retrieve it from cloud storage.
                image_bytes = download_from_supabase(storage_name)

                if not image_bytes:
                    abort(404)

                response = send_file(
                    BytesIO(image_bytes),
                    mimetype=image_record.mime_type or "application/octet-stream",
                    download_name=storage_name,
                    as_attachment=False,
                    max_age=0,
                )

        except Exception:
            current_app.logger.exception(
                "Could not retrieve image %s.",
                image_id,
            )
            abort(404)

        response.headers["Cache-Control"] = "private, no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"

        return response

    # -----------------------------------------------------
    # LOGOUT
    # -----------------------------------------------------

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()

        flash(
            "You have logged out successfully.",
            "success",
        )

        return redirect(url_for("login"))

    # -----------------------------------------------------
    # DATABASE COMMANDS
    # -----------------------------------------------------

    @app.cli.command("init-db")
    def init_db():
        """Create missing database tables."""
        db.create_all()
        click.echo("Database tables created.")

    @app.cli.command("seed-users")
    def seed_users():
        """Create local development accounts if missing."""

        accounts = [
            ("admin", "Admin@12345", "admin"),
            ("tester", "User@12345", "user"),
        ]

        try:
            for username, password, role in accounts:
                existing = db.session.scalar(
                    select(User).where(User.username == username)
                )

                if existing:
                    click.echo(
                        f"Account '{username}' already exists."
                    )
                    continue

                user = User(
                    username=username,
                    role=role,
                )

                user.set_password(password)
                db.session.add(user)

            db.session.commit()
            click.echo("Development accounts are ready.")

        except Exception:
            db.session.rollback()

            app.logger.exception(
                "Could not seed development accounts."
            )

            raise click.ClickException(
                "Could not create development accounts. "
                "Check the server error log."
            )

    return app


# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------

app = create_app()


if __name__ == "__main__":
    app.run(debug=True)