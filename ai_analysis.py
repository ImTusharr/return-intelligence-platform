
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
    return (
        os.environ.get(
            "SUPABASE_STORAGE_BUCKET",
            SUPABASE_BUCKET_DEFAULT,
        ).strip()
        or SUPABASE_BUCKET_DEFAULT
    )


def _safe_storage_name(image_record):
    """Accept only a filename as the Storage object key."""

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
    """Validate and atomically cache a downloaded image."""

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
    """Resolve an image locally first, then recover it from Supabase."""

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
# RETURN-REASON AND TEXT HELPERS
# ---------------------------------------------------------

def _get_valid_reasons(return_reasons):
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

        if (
            reason in DEFAULT_RETURN_REASONS
            and reason not in seen
        ):
            valid_reasons.append(reason)
            seen.add(reason)

    return valid_reasons or DEFAULT_RETURN_REASONS.copy()


def _as_text(value):
    """Convert AI output to readable text for database Text columns."""

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (list, tuple, set)):
        return "\n".join(
            text for item in value
            if (text := _as_text(item))
        )

    return str(value).strip()


def _as_list(value):
    """Avoid iterating over individual characters in string results."""

    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        lines = [
            line.strip().lstrip("-• ").strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return lines or [text]

    if isinstance(value, (list, tuple)):
        return [
            text for item in value
            if (text := _as_text(item))
        ]

    return [_as_text(value)]


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
    Analyse a return case with Gemini.

    Important:
    - Gemini/API failures are raised to app.py.
    - No placeholder prediction is returned after a failure.
    - Unsupported model output is rejected rather than presented
      as a valid prediction.
    - Customer and staff reason fields are never modified.
    """

    valid_reasons = _get_valid_reasons(return_reasons)

    similar_cases = _safe_historical_cases(
        similar_cases or []
    )

    analysis_notes = []
    supporting_evidence = []
    conflicting_evidence = []

    if similar_cases:
        analysis_notes.append(
            f"Historical context: {len(similar_cases)} staff-reviewed "
            "case(s) were available. These examples are context, "
            "not proof of the current return cause."
        )
    else:
        analysis_notes.append(
            "Historical context: no staff-verified comparison cases "
            "were available."
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

        if image_type not in {"listing", "received"}:
            logger.warning(
                "Ignoring image with unsupported type %r on case %s.",
                image_type,
                getattr(return_request, "id", "unknown"),
            )
            continue

        image_path = get_image_path(
            image_record,
            upload_folder,
        )

        if not image_path:
            unavailable_image_count += 1
            continue

        if image_type == "listing":
            listing_paths.append(image_path)
        else:
            received_paths.append(image_path)

    if listing_paths:
        analysis_notes.append(
            f"Image availability: {len(listing_paths)} listing/reference "
            "image(s) were retrieved."
        )
    else:
        analysis_notes.append(
            "Image availability: no listing/reference image was retrieved."
        )

    if received_paths:
        analysis_notes.append(
            f"Image availability: {len(received_paths)} received-product "
            "image(s) were retrieved."
        )
    else:
        analysis_notes.append(
            "Image availability: no received-product image was retrieved."
        )

    if unavailable_image_count:
        analysis_notes.append(
            f"Image retrieval warning: {unavailable_image_count} image(s) "
            "could not be retrieved. The assessment may be incomplete."
        )

    if selected_reason:
        analysis_notes.append(
            f"Customer-reported reason: {selected_reason}. "
            "This is the customer's selection, not an independently "
            "verified conclusion."
        )

    if customer_comment:
        analysis_notes.append(
            "Customer explanation: " + customer_comment
        )
    else:
        analysis_notes.append(
            "Customer explanation: no additional comment was provided."
        )

    evidence = {
        "customer_selected_reason": selected_reason or None,
        "customer_comment": customer_comment or None,
        "allowed_return_reasons": valid_reasons,
        "listing_image_available": bool(listing_paths),
        "listing_image_count": len(listing_paths),
        "received_image_count": len(received_paths),
        "unavailable_image_count": unavailable_image_count,
        "historical_staff_verified_cases": similar_cases,
        "important_note": (
            "Assess current images directly. Distinguish visible "
            "observations from assumptions. Do not invent measurements, "
            "product identity, functionality, damage causes, or inspection "
            "results. Missing evidence is not proof that a claim is false. "
            "Keep customer-reason agreement separate from evidence strength. "
            "A visible defect does not by itself prove what caused it."
        ),
    }

    # -----------------------------------------------------
    # GEMINI CALL
    # -----------------------------------------------------
    # Do not catch the exception and continue with a placeholder.
    # The app.py worker must receive failures so it can mark the
    # case Failed rather than Analyzed.

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

    except Exception:
        logger.exception(
            "Gemini assessment failed for return request %s.",
            getattr(return_request, "id", "unknown"),
        )
        raise

    # -----------------------------------------------------
    # VALIDATE GEMINI RESULT
    # -----------------------------------------------------

    if not isinstance(explanation, dict) or not explanation:
        raise ValueError(
            "Gemini returned no valid structured analysis. "
            "The case must not be marked Analyzed."
        )

    predicted_reason = explanation.get("predicted_reason")

    if (
        not isinstance(predicted_reason, str)
        or predicted_reason.strip() not in valid_reasons
    ):
        raise ValueError(
            "Gemini returned a missing or unsupported predicted reason. "
            "The case must not be marked Analyzed."
        )

    predicted_reason = predicted_reason.strip()

    evidence_strength = explanation.get(
        "evidence_strength",
        "Insufficient",
    )

    if (
        not isinstance(evidence_strength, str)
        or evidence_strength.strip() not in ALLOWED_EVIDENCE_LEVELS
    ):
        raise ValueError(
            "Gemini returned an invalid evidence-strength value. "
            "The case must not be marked Analyzed."
        )

    evidence_strength = evidence_strength.strip()

    # -----------------------------------------------------
    # PROCESS GEMINI RESULT
    # -----------------------------------------------------

    observations = _as_list(
        explanation.get("observations")
    )

    if observations:
        analysis_notes.append(
            "Image and evidence observations:\n- "
            + "\n- ".join(observations)
        )

    supporting_evidence.extend(
        _as_list(explanation.get("evidence_for"))
    )

    conflicting_evidence.extend(
        _as_list(explanation.get("evidence_against"))
    )

    case_summary = _as_text(
        explanation.get("case_summary")
    )

    if case_summary:
        analysis_notes.append(
            "Case summary: " + case_summary
        )

    alternatives = _as_list(
        explanation.get("alternative_explanations")
    )

    if alternatives:
        analysis_notes.append(
            "Alternative explanations:\n- "
            + "\n- ".join(alternatives)
        )

    missing = _as_list(
        explanation.get("missing_information")
    )

    if missing:
        analysis_notes.append(
            "Information to verify:\n- "
            + "\n- ".join(missing)
        )

    final_assessment = _as_text(
        explanation.get("final_assessment")
    )

    if final_assessment:
        analysis_notes.append(
            "Evidence assessment: " + final_assessment
        )

    next_action = _as_text(
        explanation.get("recommended_next_action")
    )

    if next_action:
        analysis_notes.append(
            "Recommended staff action: " + next_action
        )

    requires_human_review = explanation.get(
        "requires_human_review",
        True,
    )

    # Do not interpret arbitrary strings such as "false" as False.
    if not isinstance(requires_human_review, bool):
        requires_human_review = True

    # -----------------------------------------------------
    # CONSERVATIVE EVIDENCE VALIDATION
    # -----------------------------------------------------

    if not supporting_evidence:
        evidence_strength = "Insufficient"
        requires_human_review = True

        analysis_notes.append(
            "Evidence limitation: no specific supporting evidence was "
            "returned, so the prediction cannot be treated as substantiated."
        )

    # Material contradictions prevent a Strong rating.
    if conflicting_evidence and evidence_strength == "Strong":
        evidence_strength = "Weak"
        requires_human_review = True

        analysis_notes.append(
            "Evidence-strength adjustment: conflicting evidence prevents "
            "a Strong rating until the contradiction is resolved."
        )

    # Missing images must not be treated as proof for or against the claim.
    if unavailable_image_count:
        if evidence_strength == "Strong":
            evidence_strength = "Weak"

        requires_human_review = True

        analysis_notes.append(
            "Evidence-strength adjustment: one or more images were "
            "unavailable, so staff should review the case before relying "
            "on the AI assessment."
        )

    # Without a received-product image, Strong visual verification is
    # not justified by this workflow.
    if not received_paths and evidence_strength == "Strong":
        evidence_strength = "Weak"
        requires_human_review = True

        analysis_notes.append(
            "Evidence-strength adjustment: no received-product image was "
            "available. A Strong rating is not justified by this workflow "
            "without direct evidence of the returned item."
        )

    if not customer_comment and not received_paths:
        evidence_strength = "Insufficient"
        requires_human_review = True

        analysis_notes.append(
            "Evidence limitation: neither a customer comment nor a "
            "retrieved received-product image was available."
        )

    if evidence_strength == "Insufficient":
        requires_human_review = True

    if requires_human_review:
        analysis_notes.append(
            "Staff decision required: verify the relevant evidence before "
            "treating the AI suggestion as the confirmed return reason."
        )
    else:
        analysis_notes.append(
            "Review note: the available evidence was assessed as Strong, "
            "but the AI recommendation is still not a staff-verified result."
        )

    # This workflow does not calculate a calibrated probability.
    confidence = 0.0

    model_version = (
        f"Reasoning: {GEMINI_MODEL_NAME}; "
        "Image analysis: Gemini vision; "
        f"Historical examples: {len(similar_cases)}; "
        "Image storage: local/Supabase fallback; "
        "Local models: disabled; "
        "Probability calibration: unavailable"
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
