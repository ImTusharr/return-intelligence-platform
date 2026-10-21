
import logging
import os
import tempfile
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from gemini_reasoner import (
    GEMINI_MODEL_NAME,
    generate_case_explanation,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SUPABASE_BUCKET_DEFAULT = "return_image"

DEFAULT_RETURN_REASONS = [
    "Transit Damage",
    "Manufacturing Defect",
    "Wrong Item Received",
    "Size Issue",
    "Customer Expectation",
    "Not Working",
    "Missing Parts",
    "Other",
]

ALLOWED_EVIDENCE_LEVELS = {
    "Strong",
    "Weak",
    "Insufficient",
}

_supabase_client = None


# ---------------------------------------------------------
# SUPABASE STORAGE HELPERS
# ---------------------------------------------------------

def get_supabase_client():
    """Initialize Supabase only when a cloud image is needed."""

    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    supabase_url = os.environ.get("SUPABASE_URL", "").strip()
    supabase_key = os.environ.get(
        "SUPABASE_SERVICE_ROLE_KEY", ""
    ).strip()

    if not supabase_url or not supabase_key:
        logger.warning(
            "Supabase image recovery is unavailable because "
            "required environment variables are missing."
        )
        return None

    try:
        from supabase import create_client

        _supabase_client = create_client(
            supabase_url,
            supabase_key,
        )

        return _supabase_client

    except Exception:
        logger.exception("Could not initialize Supabase.")
        return None


def get_supabase_bucket_name():
    """Get the configured Supabase Storage bucket."""

    return (
        os.environ.get(
            "SUPABASE_STORAGE_BUCKET",
            SUPABASE_BUCKET_DEFAULT,
        ).strip()
        or SUPABASE_BUCKET_DEFAULT
    )


def _safe_storage_name(image_record):
    """Accept only a safe filename as the Storage object key."""

    if not image_record:
        return None

    storage_name = getattr(
        image_record,
        "storage_name",
        None,
    )

    if not isinstance(storage_name, str):
        return None

    storage_name = storage_name.strip()

    if (
        not storage_name
        or "\x00" in storage_name
        or "/" in storage_name
        or "\\" in storage_name
        or storage_name in {".", ".."}
        or os.path.basename(storage_name) != storage_name
    ):
        logger.warning("Rejected unsafe image storage name.")
        return None

    return storage_name


def _download_image_from_supabase(storage_name):
    """Download image bytes from the configured private bucket."""

    client = get_supabase_client()

    if client is None or not storage_name:
        return None

    try:
        bucket = client.storage.from_(
            get_supabase_bucket_name()
        )

        image_bytes = bucket.download(storage_name)

        if not image_bytes:
            logger.warning(
                "Supabase returned an empty image: %s",
                storage_name,
            )
            return None

        if not isinstance(image_bytes, bytes):
            image_bytes = bytes(image_bytes)

        return image_bytes

    except Exception:
        logger.warning(
            "Could not download image from Supabase: %s",
            storage_name,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------
# IMAGE VALIDATION AND CACHING
# ---------------------------------------------------------

def _is_valid_image_file(image_path):
    """Check whether a local file contains a readable image."""

    if not image_path or not os.path.isfile(image_path):
        return False

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.convert("RGB").load()

        return True

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ):
        return False


def _validate_image_bytes(image_bytes):
    """Validate downloaded bytes before caching them."""

    if not image_bytes:
        return False

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            image.convert("RGB").load()

        return True

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ):
        return False


def _cache_supabase_image(image_bytes, destination_path):
    """Validate and atomically cache an image downloaded from Supabase."""

    if not image_bytes or not destination_path:
        return None

    if not _validate_image_bytes(image_bytes):
        logger.warning("Downloaded content is not a valid image.")
        return None

    destination_path = os.path.realpath(destination_path)
    destination_folder = os.path.dirname(destination_path)

    temporary_path = None

    try:
        os.makedirs(destination_folder, exist_ok=True)

        file_descriptor, temporary_path = tempfile.mkstemp(
            prefix=".image-cache-",
            suffix=".tmp",
            dir=destination_folder,
        )

        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(image_bytes)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        if not _is_valid_image_file(temporary_path):
            logger.warning("Cached image failed validation.")
            return None

        os.replace(temporary_path, destination_path)
        temporary_path = None

        logger.info(
            "Cached Supabase image: %s",
            os.path.basename(destination_path),
        )

        return destination_path

    except Exception:
        logger.warning(
            "Could not cache downloaded image.",
            exc_info=True,
        )
        return None

    finally:
        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                logger.warning(
                    "Could not remove temporary image cache."
                )


def get_image_path(image_record, upload_folder):
    """
    Resolve an image using local storage first, then Supabase.

    Only safe filenames inside the configured upload folder are accepted.
    """

    if not image_record or not upload_folder:
        return None

    storage_name = _safe_storage_name(image_record)

    if not storage_name:
        return None

    try:
        upload_root = os.path.realpath(upload_folder)
        os.makedirs(upload_root, exist_ok=True)

        image_path = os.path.realpath(
            os.path.join(upload_root, storage_name)
        )

        if os.path.commonpath(
            [upload_root, image_path]
        ) != upload_root:
            logger.warning("Rejected path outside upload folder.")
            return None

        if _is_valid_image_file(image_path):
            return image_path

        logger.info(
            "Local image unavailable; checking Supabase for %s.",
            storage_name,
        )

        image_bytes = _download_image_from_supabase(storage_name)

        if image_bytes is None:
            return None

        cached_path = _cache_supabase_image(
            image_bytes,
            image_path,
        )

        if cached_path and _is_valid_image_file(cached_path):
            return cached_path

    except (OSError, ValueError, TypeError):
        logger.warning(
            "Could not resolve image %s.",
            storage_name,
            exc_info=True,
        )

    return None


# ---------------------------------------------------------
# RETURN-REASON HELPERS
# ---------------------------------------------------------

def _get_valid_reasons(return_reasons):
    """Keep supported return reasons in the supplied order."""

    source = (
        return_reasons
        if return_reasons is not None
        else DEFAULT_RETURN_REASONS
    )

    if isinstance(source, str):
        source = [source]

    valid_reasons = []
    seen = set()

    for reason in source:
        if not isinstance(reason, str):
            continue

        reason = reason.strip()

        if reason in DEFAULT_RETURN_REASONS and reason not in seen:
            valid_reasons.append(reason)
            seen.add(reason)

    return valid_reasons or DEFAULT_RETURN_REASONS.copy()


def _as_text(value):
    """Convert AI output into text for database Text columns."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple, set)):
        return "\n".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )

    return str(value)


def _safe_historical_cases(similar_cases):
    """Prepare historical case context without trusting arbitrary values."""

    if not isinstance(similar_cases, (list, tuple)):
        return []

    prepared = []

    for case in similar_cases[:3]:
        if not isinstance(case, dict):
            continue

        prepared.append({
            "case_id": case.get("case_id"),
            "product_name": str(
                case.get("product_name", "")
            )[:200],
            "customer_selected_reason": str(
                case.get("customer_selected_reason", "")
            )[:100],
            "customer_comment": str(
                case.get("customer_comment", "")
            )[:700],
            "staff_verified_reason": str(
                case.get("staff_verified_reason", "")
            )[:100],
            "review_status": str(
                case.get("review_status", "")
            )[:30],
            "reviewer_notes": str(
                case.get("reviewer_notes", "")
            )[:500],
            "similarity_score": case.get("similarity_score", 0),
        })

    return prepared


# ---------------------------------------------------------
# MAIN RETURN ANALYSIS
# ---------------------------------------------------------

def analyze_return_request(
    return_request,
    upload_folder,
    return_reasons,
    similar_cases=None,
):
    """
    Analyze a return case using Gemini's text and image capabilities.

    The public function signature and returned dictionary remain
    compatible with the existing app.py workflow.

    No local PyTorch, Transformers, or CLIP model is loaded.
    """

    valid_reasons = _get_valid_reasons(return_reasons)
    similar_cases = _safe_historical_cases(similar_cases or [])

    analysis_notes = []
    supporting_evidence = []
    conflicting_evidence = []

    if similar_cases:
        analysis_notes.append(
            f"{len(similar_cases)} historical staff-reviewed case(s) "
            "were supplied as context. They are not proof of the current "
            "return cause."
        )
    else:
        analysis_notes.append(
            "No historical staff-reviewed examples were available."
        )

    selected_reason = (
        getattr(return_request, "selected_reason", None)
        or getattr(return_request, "return_reason", None)
        or getattr(return_request, "reason", None)
        or ""
    )

    customer_comment = (
        getattr(return_request, "reason_comment", None)
        or getattr(return_request, "customer_comment", None)
        or getattr(return_request, "comment", None)
        or ""
    )

    selected_reason = str(selected_reason).strip()
    customer_comment = str(customer_comment).strip()

    listing_paths = []
    received_paths = []
    unavailable_image_count = 0

    images = list(
        getattr(return_request, "images", []) or []
    )

    for image_record in images:
        image_type = str(
            getattr(image_record, "image_type", "") or ""
        ).strip().lower()

        image_path = get_image_path(
            image_record,
            upload_folder,
        )

        if not image_path:
            unavailable_image_count += 1
            continue

        if image_type == "listing":
            listing_paths.append(image_path)

        elif image_type == "received":
            received_paths.append(image_path)

    if listing_paths:
        analysis_notes.append(
            f"{len(listing_paths)} listing image(s) were retrieved."
        )

    if received_paths:
        analysis_notes.append(
            f"{len(received_paths)} received-product image(s) were retrieved."
        )

    if unavailable_image_count:
        analysis_notes.append(
            f"{unavailable_image_count} image(s) could not be retrieved. "
            "The available evidence may be incomplete."
        )

    if selected_reason:
        analysis_notes.append(
            f"Customer-selected reason: {selected_reason}. "
            "This is a customer claim, not a verified conclusion."
        )

    if customer_comment:
        supporting_evidence.append(
            "A customer comment was provided for analysis."
        )
    else:
        analysis_notes.append(
            "No customer comment was provided."
        )

    if not listing_paths:
        analysis_notes.append(
            "No listing/reference image was available."
        )

    if not received_paths:
        analysis_notes.append(
            "No received-product image was available."
        )

    # -----------------------------------------------------
    # GEMINI EVIDENCE ASSESSMENT
    # -----------------------------------------------------

    evidence = {
        "customer_selected_reason": selected_reason or None,
        "customer_comment": customer_comment or None,
        "allowed_return_reasons": valid_reasons,
        "listing_image_available": bool(listing_paths),
        "received_image_count": len(received_paths),
        "unavailable_image_count": unavailable_image_count,
        "supporting_evidence": supporting_evidence,
        "conflicting_evidence": conflicting_evidence,
        "historical_staff_verified_cases": similar_cases,
        "important_note": (
            "Assess listing and received images directly when available. "
            "Distinguish visible observations from assumptions. Do not "
            "invent measurements, product identity, functionality, damage "
            "causes, or inspection results. Historical cases and customer "
            "claims are context, not ground truth. Weak, contradictory, "
            "or insufficient evidence requires human review."
        ),
    }

    explanation = None

    try:
        explanation = generate_case_explanation(
            evidence,
            valid_reasons=valid_reasons,
            image_paths={
                "listing": listing_paths,
                "received": received_paths,
            },
            similar_cases=similar_cases,
        )

    except Exception as exc:
        logger.warning(
            "Gemini assessment failed for return request %s.",
            getattr(return_request, "id", "unknown"),
            exc_info=True,
        )

        analysis_notes.append(
            f"Gemini assessment failed ({type(exc).__name__}). "
            "A conservative fallback will be returned."
        )

    # -----------------------------------------------------
    # VALIDATE GEMINI RESULT
    # -----------------------------------------------------

    if explanation:
        predicted_reason = explanation.get("predicted_reason")
        evidence_strength = explanation.get(
            "evidence_strength",
            "Insufficient",
        )

        for item in explanation.get("evidence_for", []):
            if item:
                supporting_evidence.append(str(item))

        for item in explanation.get("evidence_against", []):
            if item:
                conflicting_evidence.append(str(item))

        if explanation.get("case_summary"):
            analysis_notes.append(
                "Gemini case summary: "
                + _as_text(explanation["case_summary"])
            )

        alternatives = explanation.get("alternative_explanations", [])

        if alternatives:
            analysis_notes.append(
                "Alternative explanations: "
                + " | ".join(str(item) for item in alternatives)
            )

        missing = explanation.get("missing_information", [])

        if missing:
            analysis_notes.append(
                "Information to verify: "
                + " | ".join(str(item) for item in missing)
            )

        if explanation.get("final_assessment"):
            analysis_notes.append(
                "Gemini assessment: "
                + _as_text(explanation["final_assessment"])
            )

        if explanation.get("recommended_next_action"):
            analysis_notes.append(
                "Recommended staff action: "
                + _as_text(explanation["recommended_next_action"])
            )

        requires_human_review = explanation.get(
            "requires_human_review",
            True,
        )

    else:
        # Do not present the customer's own selection as an AI prediction.
        predicted_reason = (
            "Other" if "Other" in valid_reasons else valid_reasons[0]
        )

        evidence_strength = "Insufficient"
        requires_human_review = True

        analysis_notes.append(
            "No verified AI prediction is available. The fallback category "
            "is only a placeholder and must not be treated as an assessment."
        )

    # -----------------------------------------------------
    # SAFETY AND APP COMPATIBILITY
    # -----------------------------------------------------

    if predicted_reason not in valid_reasons:
        predicted_reason = (
            "Other" if "Other" in valid_reasons else valid_reasons[0]
        )

        evidence_strength = "Insufficient"
        requires_human_review = True

        analysis_notes.append(
            "An unsupported category was returned; a permitted fallback "
            "was selected."
        )

    if evidence_strength not in ALLOWED_EVIDENCE_LEVELS:
        evidence_strength = "Insufficient"
        requires_human_review = True

    if (
        conflicting_evidence
        and evidence_strength == "Strong"
    ):
        evidence_strength = "Weak"
        requires_human_review = True

    if unavailable_image_count:
        if evidence_strength == "Strong":
            evidence_strength = "Weak"

        requires_human_review = True

    if not customer_comment and not received_paths:
        evidence_strength = "Insufficient"
        requires_human_review = True

    if requires_human_review:
        analysis_notes.append(
            "Human review is required before the AI suggestion can be "
            "treated as a verified return reason."
        )

    # Gemini does not provide a calibrated probability in this workflow.
    # Keep this numeric field for compatibility with the existing dashboard.
    confidence = 0.0

    analysis_notes.append(
        "Dashboard confidence score is unavailable (0.0). This version "
        "does not use a local classifier or claim a calibrated probability."
    )

    model_version = (
        f"Reasoning: {GEMINI_MODEL_NAME}; "
        "Image analysis: Gemini vision; "
        f"Historical examples: {len(similar_cases)}; "
        "Image storage: local/Supabase fallback; "
        "Local models: disabled"
    )

    # Do not modify customer-selected or staff-verified database fields.
    return {
        "predicted_reason": predicted_reason,
        "evidence_strength": evidence_strength,
        "confidence": confidence,
        "supporting_evidence": _as_text(supporting_evidence),
        "conflicting_evidence": _as_text(conflicting_evidence),
        "analysis_notes": _as_text(analysis_notes),
        "model_version": model_version,
    }