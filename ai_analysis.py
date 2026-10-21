
import logging
import os

import torch
from PIL import Image, UnidentifiedImageError
from transformers import pipeline

from gemini_reasoner import (
    GEMINI_MODEL_NAME,
    generate_case_explanation,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# MODEL CONFIGURATION
# ---------------------------------------------------------

TEXT_MODEL_NAME = "valhalla/distilbart-mnli-12-1"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

_text_model = None
_image_classifier = None
_clip_model = None
_clip_processor = None


# ---------------------------------------------------------
# RETURN REASON DEFINITIONS
# ---------------------------------------------------------

RETURN_REASON_DESCRIPTIONS = {
    "Transit Damage": (
        "Physical product damage that may have occurred during transportation "
        "or delivery. Packaging and dispatch evidence are useful to confirm cause."
    ),
    "Manufacturing Defect": (
        "A defect related to manufacturing, construction, assembly, or material quality."
    ),
    "Wrong Item Received": (
        "The delivered product differs from the product, model, design, or item ordered."
    ),
    "Size Issue": (
        "The product size, fit, or dimensions do not meet the customer's requirements."
    ),
    "Customer Expectation": (
        "The product does not meet expectations regarding appearance, features, or description."
    ),
    "Not Working": (
        "The product does not function or operate as expected."
    ),
    "Missing Parts": (
        "One or more expected product components or accessories may be missing."
    ),
    "Other": (
        "The evidence does not clearly fit another available return category."
    ),
}

DEFAULT_RETURN_REASONS = list(RETURN_REASON_DESCRIPTIONS.keys())

ALLOWED_EVIDENCE_LEVELS = {
    "Strong",
    "Weak",
    "Insufficient",
}


VISUAL_OBSERVATION_LABELS = [
    "product with visible cracks, dents, scratches, or broken parts",
    "product with a visibly different design, shape, or logo",
    "product with a visibly different color",
    "product with visibly missing components or accessories",
    "product that appears visually intact with no obvious damage",
    "image where relevant product details are unclear",
]


# Visual observations are clues, not proof of a return cause.
VISUAL_REASON_MAPPING = {
    VISUAL_OBSERVATION_LABELS[0]: [
        "Transit Damage",
        "Manufacturing Defect",
    ],
    VISUAL_OBSERVATION_LABELS[1]: [
        "Wrong Item Received",
    ],
    VISUAL_OBSERVATION_LABELS[2]: [
        "Wrong Item Received",
        "Customer Expectation",
    ],
    VISUAL_OBSERVATION_LABELS[3]: [
        "Missing Parts",
    ],
    VISUAL_OBSERVATION_LABELS[4]: [],
    VISUAL_OBSERVATION_LABELS[5]: [],
}


# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def get_text_model():
    """Load the text classification model only when required."""

    global _text_model

    if _text_model is None:
        _text_model = pipeline(
            "zero-shot-classification",
            model=TEXT_MODEL_NAME,
            device=-1,
        )

    return _text_model


def get_image_classifier():
    """Load the CLIP image observation classifier only when required."""

    global _image_classifier

    if _image_classifier is None:
        _image_classifier = pipeline(
            "zero-shot-image-classification",
            model=IMAGE_MODEL_NAME,
            device=-1,
        )

    return _image_classifier


def get_clip_components():
    """Load CLIP components for comparing listing and received images."""

    global _clip_model, _clip_processor

    if _clip_model is None or _clip_processor is None:
        from transformers import CLIPModel, CLIPProcessor

        _clip_model = CLIPModel.from_pretrained(
            IMAGE_MODEL_NAME
        )

        _clip_processor = CLIPProcessor.from_pretrained(
            IMAGE_MODEL_NAME
        )

        _clip_model.eval()

    return _clip_model, _clip_processor


# ---------------------------------------------------------
# IMAGE HELPERS
# ---------------------------------------------------------

def get_image_path(image_record, upload_folder):
    """
    Resolve an image record's storage_name inside the upload folder.

    Returns None for missing, invalid, or unsafe paths.
    """

    if not image_record or not upload_folder:
        return None

    storage_name = getattr(image_record, "storage_name", None)

    if not storage_name:
        return None

    try:
        upload_root = os.path.realpath(upload_folder)

        if os.path.isabs(storage_name):
            image_path = os.path.realpath(storage_name)
        else:
            image_path = os.path.realpath(
                os.path.join(upload_root, storage_name)
            )

        # Prevent stored filenames from escaping the upload folder.
        if os.path.commonpath(
            [upload_root, image_path]
        ) != upload_root:
            logger.warning(
                "Rejected image path outside upload folder."
            )
            return None

        if os.path.isfile(image_path):
            return image_path

    except (OSError, ValueError, TypeError):
        logger.warning(
            "Could not resolve image path.",
            exc_info=True,
        )

    return None


def load_rgb_image(image_path):
    """Validate an image and return an independent RGB copy."""

    if not image_path or not os.path.isfile(image_path):
        return None

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            return image.convert("RGB").copy()

    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
    ):
        logger.warning(
            "Unable to load image: %s",
            os.path.basename(image_path),
        )
        return None


# ---------------------------------------------------------
# LISTING / RECEIVED IMAGE COMPARISON
# ---------------------------------------------------------

def compare_listing_and_received_images(
    listing_path,
    received_paths,
):
    """
    Calculate the highest CLIP cosine similarity between the listing
    image and valid received images.

    Similarity indicates visual resemblance only. It does not prove
    product identity or establish the cause of damage.

    Returns a float or None.
    """

    listing_image = load_rgb_image(listing_path)

    if listing_image is None or not received_paths:
        return None

    try:
        model, processor = get_clip_components()

        listing_inputs = processor(
            images=listing_image,
            return_tensors="pt",
        )

        with torch.inference_mode():
            listing_output = model.vision_model(
                pixel_values=listing_inputs["pixel_values"]
            )

            listing_features = model.visual_projection(
                listing_output.pooler_output
            )

            listing_features = torch.nn.functional.normalize(
                listing_features,
                p=2,
                dim=-1,
            )

        similarities = []

        for received_path in received_paths:
            received_image = load_rgb_image(received_path)

            if received_image is None:
                continue

            received_inputs = processor(
                images=received_image,
                return_tensors="pt",
            )

            with torch.inference_mode():
                received_output = model.vision_model(
                    pixel_values=received_inputs["pixel_values"]
                )

                received_features = model.visual_projection(
                    received_output.pooler_output
                )

                received_features = torch.nn.functional.normalize(
                    received_features,
                    p=2,
                    dim=-1,
                )

                similarity = torch.sum(
                    listing_features * received_features,
                    dim=-1,
                ).item()

            similarities.append(float(similarity))

        if not similarities:
            return None

        return max(similarities)

    except Exception:
        logger.warning(
            "Listing/received image comparison failed.",
            exc_info=True,
        )
        return None


# ---------------------------------------------------------
# CLASSIFICATION HELPERS
# ---------------------------------------------------------

def _get_valid_reasons(return_reasons):
    """Keep only supported return-reason names."""

    source_reasons = (
        return_reasons
        if return_reasons is not None
        else DEFAULT_RETURN_REASONS
    )

    valid_reasons = []
    seen = set()

    for reason in source_reasons:
        if not isinstance(reason, str):
            continue

        reason = reason.strip()

        if (
            reason in RETURN_REASON_DESCRIPTIONS
            and reason not in seen
        ):
            valid_reasons.append(reason)
            seen.add(reason)

    return valid_reasons or DEFAULT_RETURN_REASONS.copy()


def _classify_comment(comment, valid_reasons):
    """
    Rank return reasons using the DistilBART zero-shot classifier.

    Scores are relative model scores, not calibrated probabilities.
    """

    if (
        not isinstance(comment, str)
        or not comment.strip()
        or not valid_reasons
    ):
        return {}

    candidate_labels = [
        (
            f"{reason}. "
            f"{RETURN_REASON_DESCRIPTIONS[reason]}"
        )
        for reason in valid_reasons
    ]

    classifier = get_text_model()

    result = classifier(
        comment.strip(),
        candidate_labels=candidate_labels,
        multi_label=False,
    )

    scores = {}
    label_to_reason = dict(
        zip(candidate_labels, valid_reasons)
    )

    for label, score in zip(
        result.get("labels", []),
        result.get("scores", []),
    ):
        reason = label_to_reason.get(label)

        if reason:
            scores[reason] = float(score)

    return scores


def _classify_received_images(received_paths):
    """
    Classify visual observations for received images.

    This does not establish product functionality, completeness,
    or the cause of damage.
    """

    if not received_paths:
        return {}

    classifier = get_image_classifier()
    image_results = []

    for image_path in received_paths:
        image = load_rgb_image(image_path)

        if image is None:
            continue

        try:
            results = classifier(
                image,
                candidate_labels=VISUAL_OBSERVATION_LABELS,
            )

            image_scores = {
                item["label"]: float(item["score"])
                for item in results
                if item.get("label") in VISUAL_OBSERVATION_LABELS
            }

            if image_scores:
                image_results.append(image_scores)

        except Exception:
            logger.warning(
                "Image observation classification failed.",
                exc_info=True,
            )

    if not image_results:
        return {}

    average_scores = {}

    for label in VISUAL_OBSERVATION_LABELS:
        values = [
            scores[label]
            for scores in image_results
            if label in scores
        ]

        if values:
            average_scores[label] = (
                sum(values) / len(values)
            )

    return average_scores


def _format_score(score):
    """Format a model score for staff-facing notes."""

    if score is None:
        return "unavailable"

    return f"{score * 100:.1f}/100"


def _assess_fallback_evidence(
    text_scores,
    predicted_reason,
    conflicting_evidence,
):
    """
    Conservative fallback if Gemini is unavailable.

    This is a review indicator, not a validated accuracy measurement.
    """

    if (
        not text_scores
        or predicted_reason not in text_scores
    ):
        return "Insufficient"

    ranked_scores = sorted(
        text_scores.values(),
        reverse=True,
    )

    top_score = ranked_scores[0]
    second_score = (
        ranked_scores[1]
        if len(ranked_scores) > 1
        else 0.0
    )

    score_gap = top_score - second_score

    if conflicting_evidence:
        return "Weak"

    # Even a high text score is not independent proof of the cause.
    if top_score >= 0.60 and score_gap >= 0.15:
        return "Weak"

    if top_score >= 0.45 and score_gap >= 0.08:
        return "Weak"

    return "Weak"


def _as_text(items):
    """Convert evidence values into strings for SQLAlchemy Text columns."""

    if items is None:
        return ""

    if isinstance(items, str):
        return items.strip()

    if isinstance(items, (list, tuple, set)):
        return "\n".join(
            str(item).strip()
            for item in items
            if str(item).strip()
        )

    return str(items)


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
    Analyze a return request while preserving the existing app.py interface.

    Gemini assesses combined evidence when available. DistilBART and CLIP
    provide supporting signals. The prediction is not a staff-verified result.
    """

    valid_reasons = _get_valid_reasons(return_reasons)
    similar_cases = list(similar_cases or [])[:3]

    if similar_cases:
        analysis_notes_for_cases = (
            f"{len(similar_cases)} earlier staff-verified similar case(s) "
            "were retrieved as context. These examples are not ground truth "
            "for the current return and do not replace image/text assessment."
        )
    else:
        analysis_notes_for_cases = (
            "No earlier similar staff-verified cases were available; "
            "analysis relies on this case's evidence."
        )

    # These match the fields in the supplied Flask application's model.
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

    supporting_evidence = []
    conflicting_evidence = []
    analysis_notes = [analysis_notes_for_cases]

    images = list(
        getattr(return_request, "images", []) or []
    )

    listing_paths = []
    received_paths = []

    for image_record in images:
        image_type = (
            getattr(image_record, "image_type", "") or ""
        ).strip().lower()

        image_path = get_image_path(
            image_record,
            upload_folder,
        )

        if not image_path:
            continue

        if image_type == "listing":
            listing_paths.append(image_path)

        elif image_type == "received":
            received_paths.append(image_path)

    listing_path = (
        listing_paths[0]
        if listing_paths
        else None
    )

    # -----------------------------------------------------
    # INPUT NOTES
    # -----------------------------------------------------

    if selected_reason:
        analysis_notes.append(
            f"Customer-selected reason: {selected_reason}. "
            "This is customer-provided information, not verified ground truth."
        )

    if customer_comment:
        supporting_evidence.append(
            "A customer comment was provided for text analysis."
        )
    else:
        analysis_notes.append(
            "No customer comment was provided. "
            "Text classification is unavailable."
        )

    if not received_paths:
        analysis_notes.append(
            "No valid received-product images were available "
            "for visual analysis."
        )

    # -----------------------------------------------------
    # STEP 1: DISTILBART TEXT CLASSIFICATION
    # -----------------------------------------------------

    text_scores = {}

    try:
        if customer_comment and valid_reasons:
            text_scores = _classify_comment(
                customer_comment,
                valid_reasons,
            )

    except Exception:
        logger.warning(
            "Text classification failed for return request %s.",
            getattr(return_request, "id", "unknown"),
            exc_info=True,
        )

        analysis_notes.append(
            "DistilBART text classification failed."
        )

    ranked_text = sorted(
        text_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    text_suggestion = (
        ranked_text[0][0]
        if ranked_text
        else None
    )

    top_score = (
        ranked_text[0][1]
        if ranked_text
        else 0.0
    )

    second_score = (
        ranked_text[1][1]
        if len(ranked_text) > 1
        else 0.0
    )

    score_gap = top_score - second_score

    if text_suggestion:
        analysis_notes.append(
            "DistilBART's leading suggestion: "
            f"{text_suggestion} "
            f"(raw model score: {_format_score(top_score)})."
        )

        analysis_notes.append(
            "DistilBART scores are relative model scores, "
            "not calibrated probabilities of correctness."
        )

        if selected_reason in valid_reasons:
            if selected_reason == text_suggestion:
                supporting_evidence.append(
                    "DistilBART agrees with the customer's selected reason. "
                    "Agreement does not independently verify the reason."
                )
            else:
                conflicting_evidence.append(
                    "The customer's selected reason differs from "
                    "DistilBART's leading suggestion."
                )

    # -----------------------------------------------------
    # STEP 2: CLIP VISUAL OBSERVATIONS
    # -----------------------------------------------------

    visual_scores = {}

    try:
        if received_paths:
            visual_scores = _classify_received_images(
                received_paths
            )

    except Exception:
        logger.warning(
            "Visual analysis failed for return request %s.",
            getattr(return_request, "id", "unknown"),
            exc_info=True,
        )

        analysis_notes.append(
            "CLIP visual observation classification failed."
        )

    ranked_visual = sorted(
        visual_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    leading_visual_observation = (
        ranked_visual[0][0]
        if ranked_visual
        else None
    )

    leading_visual_score = (
        ranked_visual[0][1]
        if ranked_visual
        else None
    )

    if leading_visual_observation:
        analysis_notes.append(
            "CLIP's leading visual observation: "
            f"{leading_visual_observation} "
            f"(raw image-model score: "
            f"{_format_score(leading_visual_score)})."
        )

        analysis_notes.append(
            "CLIP scores are visual clues, not proof of product condition "
            "or the cause of damage."
        )

        compatible_reasons = VISUAL_REASON_MAPPING.get(
            leading_visual_observation,
            [],
        )

        if compatible_reasons:
            supporting_evidence.append(
                "The leading visual observation may be relevant to: "
                + ", ".join(compatible_reasons)
                + ". Staff must verify this clue."
            )

        if (
            text_suggestion
            and compatible_reasons
            and text_suggestion not in compatible_reasons
        ):
            conflicting_evidence.append(
                "CLIP's leading visual observation may not align with "
                "DistilBART's leading text suggestion."
            )

        if (
            leading_visual_observation
            == VISUAL_OBSERVATION_LABELS[0]
        ):
            analysis_notes.append(
                "Visible damage alone cannot distinguish transit damage "
                "from a manufacturing defect. Check packaging, dispatch "
                "records, and inspection evidence."
            )

        elif (
            leading_visual_observation
            == VISUAL_OBSERVATION_LABELS[3]
        ):
            analysis_notes.append(
                "An image cannot reliably prove that a component is missing. "
                "Check expected product contents and inspect the package."
            )

        elif (
            leading_visual_observation
            == VISUAL_OBSERVATION_LABELS[4]
        ):
            analysis_notes.append(
                "An apparently intact image does not rule out hidden damage "
                "or a functional defect."
            )

    # -----------------------------------------------------
    # STEP 3: LISTING / RECEIVED IMAGE SIMILARITY
    # -----------------------------------------------------

    image_similarity = compare_listing_and_received_images(
        listing_path,
        received_paths,
    )

    if image_similarity is not None:
        analysis_notes.append(
            "Listing/received image cosine similarity: "
            f"{image_similarity:.3f}. "
            "This indicates visual resemblance only, not product identity."
        )

    else:
        analysis_notes.append(
            "Listing/received image similarity could not be calculated."
        )

    # -----------------------------------------------------
    # STEP 4: GEMINI EVIDENCE ASSESSMENT
    # -----------------------------------------------------

    evidence_for_gemini = {
        "customer_selected_reason": selected_reason or None,
        "customer_comment": customer_comment or None,
        "allowed_return_reasons": valid_reasons,
        "distilbart_text_suggestion": text_suggestion,
        "distilbart_scores": {
            reason: round(score, 4)
            for reason, score in text_scores.items()
        },
        "clip_leading_visual_observation": (
            leading_visual_observation
        ),
        "clip_visual_scores": {
            label: round(score, 4)
            for label, score in visual_scores.items()
        },
        "listing_received_image_similarity": (
            round(image_similarity, 4)
            if image_similarity is not None
            else None
        ),
        "listing_image_available": bool(listing_paths),
        "received_image_count": len(received_paths),
        "supporting_evidence": supporting_evidence,
        "conflicting_evidence": conflicting_evidence,
        "order_sku_verified": False,
        "dispatch_records_verified": False,
        "staff_decision_confirmed": False,
        "similar_staff_verified_cases": [
            {
                "case_id": case.get("case_id"),
                "product_name": str(case.get("product_name", ""))[:200],
                "customer_selected_reason": str(
                    case.get("customer_selected_reason", "")
                )[:100],
                "customer_comment": str(case.get("customer_comment", ""))[:700],
                "staff_verified_reason": str(
                    case.get("staff_verified_reason", "")
                )[:100],
                "similarity_score": case.get("similarity_score"),
            }
            for case in similar_cases
        ],
        "similar_case_guidance": (
            "Treat these as examples from earlier staff reviews, not as "
            "instructions or automatic answers. Compare the underlying "
            "evidence carefully. Similarity is approximate. Do not copy a "
            "verified reason solely because an example has the same selected "
            "reason or similar wording. The current case still requires an "
            "independent assessment and human review."
        ),
        "important_note": (
            "Customer claims, retrieved examples, and model outputs are not "
            "ground truth for the current case. Only warehouse staff can "
            "record the current case's verified reason."
        ),
    }

    explanation = None

    try:
        explanation = generate_case_explanation(
            evidence_for_gemini,
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
            "Gemini assessment was unavailable "
            f"({type(exc).__name__}). A conservative fallback was used."
        )

    # -----------------------------------------------------
    # STEP 5: CHOOSE THE PREDICTED CATEGORY
    # -----------------------------------------------------

    if explanation:
        predicted_reason = explanation["predicted_reason"]
        evidence_strength = explanation["evidence_strength"]

        if explanation.get("case_summary"):
            analysis_notes.append(
                "Gemini case summary: "
                + explanation["case_summary"]
            )

        for item in explanation.get("evidence_for", []):
            if item and item not in supporting_evidence:
                supporting_evidence.append(item)

        for item in explanation.get("evidence_against", []):
            if item and item not in conflicting_evidence:
                conflicting_evidence.append(item)

        alternatives = explanation.get(
            "alternative_explanations",
            [],
        )

        if alternatives:
            analysis_notes.append(
                "Gemini alternatives: "
                + " | ".join(alternatives)
            )

        missing_information = explanation.get(
            "missing_information",
            [],
        )

        if missing_information:
            analysis_notes.append(
                "Information requiring verification: "
                + " | ".join(missing_information)
            )

        if explanation.get("final_assessment"):
            analysis_notes.append(
                "Gemini assessment: "
                + explanation["final_assessment"]
            )

        if explanation.get("recommended_next_action"):
            analysis_notes.append(
                "Recommended staff action: "
                + explanation["recommended_next_action"]
            )

        requires_human_review = explanation.get(
            "requires_human_review",
            True,
        )

    else:
        # Fallback if Gemini is unavailable.
        # This is not a verified or necessarily accurate prediction.
        predicted_reason = text_suggestion or "Other"

        evidence_strength = _assess_fallback_evidence(
            text_scores,
            predicted_reason,
            conflicting_evidence,
        )

        requires_human_review = True

        analysis_notes.append(
            "Fallback category is unverified. Human review is required."
        )

    # -----------------------------------------------------
    # STEP 6: ENFORCE APP.PY COMPATIBILITY
    # -----------------------------------------------------

    if predicted_reason not in valid_reasons:
        predicted_reason = (
            "Other"
            if "Other" in valid_reasons
            else valid_reasons[0]
        )

        evidence_strength = "Insufficient"
        requires_human_review = True

        analysis_notes.append(
            "An unsupported model category was returned. "
            "A permitted fallback category was selected."
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

    if not customer_comment and not received_paths:
        evidence_strength = "Insufficient"
        requires_human_review = True

    if requires_human_review:
        analysis_notes.append(
            "Human review is required before the AI suggestion "
            "can be treated as a verified return reason."
        )

    # -----------------------------------------------------
    # STEP 7: DASHBOARD SCORE AND FINAL RESULT
    # -----------------------------------------------------

    # Preserve the existing numeric field for dashboard compatibility.
    # This is not calibrated confidence or measured accuracy.
    confidence = round(
        text_scores.get(predicted_reason, 0.0) * 100,
        2,
    )

    if predicted_reason in text_scores:
        analysis_notes.append(
            "Dashboard score is the raw DistilBART score for the selected "
            "category, not a calibrated probability of correctness."
        )
    else:
        analysis_notes.append(
            "No DistilBART score was available for the selected category; "
            "dashboard score is 0."
        )

    model_version = (
        f"Text: {TEXT_MODEL_NAME}; "
        f"Image: {IMAGE_MODEL_NAME}; "
        f"Reasoning: {GEMINI_MODEL_NAME}; "
        f"Verified examples: {len(similar_cases)}"
    )

    # The supplied app.py uses SQLAlchemy Text columns for these fields.
    return {
        "predicted_reason": predicted_reason,
        "evidence_strength": evidence_strength,
        "confidence": confidence,
        "supporting_evidence": _as_text(supporting_evidence),
        "conflicting_evidence": _as_text(conflicting_evidence),
        "analysis_notes": _as_text(analysis_notes),
        "model_version": model_version,
    }
