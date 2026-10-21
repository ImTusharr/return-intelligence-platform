
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from transformers import pipeline


# These are the return reasons already used in the Flask project.
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

TEXT_MODEL_NAME = "valhalla/distilbart-mnli-12-1"
IMAGE_MODEL_NAME = "openai/clip-vit-base-patch32"

_text_model = None
_image_model = None


def get_text_model():
    global _text_model

    if _text_model is None:
        _text_model = pipeline(
            "zero-shot-classification",
            model=TEXT_MODEL_NAME,
            device=-1,
        )

    return _text_model


def get_image_model():
    global _image_model

    if _image_model is None:
        _image_model = pipeline(
            "zero-shot-image-classification",
            model=IMAGE_MODEL_NAME,
            device=-1,
        )

    return _image_model


def analyze_return_evidence(
    selected_reason,
    reason_comment,
    received_image_paths,
):
    """
    Analyze a return using the customer's text and received-item images.

    The result is an AI suggestion, not a staff-confirmed reason.
    Model scores are not calibrated probabilities.
    """

    text = (reason_comment or "").strip()
    selected_reason = (selected_reason or "").strip()

    valid_images = []

    for image_path in received_image_paths or []:
        path = Path(image_path)

        if not path.is_file():
            continue

        try:
            with Image.open(path) as image:
                image.verify()

            valid_images.append(str(path))

        except (OSError, UnidentifiedImageError):
            continue

    # Do not claim that AI analyzed evidence if there is none.
    if not text and not valid_images:
        raise ValueError(
            "There is no usable text or received-item image to analyze."
        )

    text_scores = {}
    image_scores = {}

    # Analyze the customer's written explanation.
    if text:
        text_result = get_text_model()(
            text,
            candidate_labels=RETURN_REASONS,
            multi_label=False,
        )

        text_scores = dict(
            zip(text_result["labels"], text_result["scores"])
        )

    # Analyze each received-item image.
    if valid_images:
        image_model = get_image_model()

        for image_path in valid_images:
            image_result = image_model(
                image_path,
                candidate_labels=RETURN_REASONS,
            )

            for result in image_result:
                reason = result["label"]
                score = float(result["score"])

                image_scores.setdefault(reason, []).append(score)

        # Average image scores for each reason.
        image_scores = {
            reason: sum(scores) / len(scores)
            for reason, scores in image_scores.items()
        }

    # Combine only the modalities that actually ran.
    combined_scores = {}

    for reason in RETURN_REASONS:
        scores = []
        weights = []

        if reason in text_scores:
            scores.append(float(text_scores[reason]))
            weights.append(0.55)

        if reason in image_scores:
            scores.append(float(image_scores[reason]))
            weights.append(0.45)

        if scores:
            combined_scores[reason] = sum(
                score * weight
                for score, weight in zip(scores, weights)
            ) / sum(weights)

    if not combined_scores:
        raise ValueError("The AI models did not return usable scores.")

    ranked_reasons = sorted(
        combined_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    predicted_reason, top_score = ranked_reasons[0]

    second_score = (
        ranked_reasons[1][1]
        if len(ranked_reasons) > 1
        else 0.0
    )

    # Check whether text and image models agree on the leading reason.
    modalities_agree = (
        bool(text_scores)
        and bool(image_scores)
        and max(text_scores, key=text_scores.get)
        == max(image_scores, key=image_scores.get)
    )

    # Evidence strength is a triage signal, not a return decision.
    if (
        len(text_scores) > 0
        and len(image_scores) > 0
        and modalities_agree
        and top_score >= 0.45
        and (top_score - second_score) >= 0.08
    ):
        evidence_strength = "Strong"

    elif (
        top_score >= 0.30
        and (top_score - second_score) >= 0.03
    ):
        evidence_strength = "Weak"

    else:
        evidence_strength = "Insufficient"

    supporting_evidence = []

    if text_scores:
        text_reason = max(text_scores, key=text_scores.get)
        supporting_evidence.append(
            f"Text model's leading suggestion: {text_reason}."
        )

    if image_scores:
        image_reason = max(image_scores, key=image_scores.get)
        supporting_evidence.append(
            f"Image model's leading suggestion: {image_reason}."
        )

    conflicting_evidence = []

    if text_scores and image_scores and not modalities_agree:
        conflicting_evidence.append(
            "The leading text and image model suggestions differ."
        )

    if selected_reason and predicted_reason != selected_reason:
        conflicting_evidence.append(
            "The AI suggestion differs from the customer's selected reason."
        )

    if not text_scores:
        conflicting_evidence.append(
            "No usable text was available for text-model analysis."
        )

    if not image_scores:
        conflicting_evidence.append(
            "No usable received-item image was available for image-model analysis."
        )

    return {
        "predicted_reason": predicted_reason,
        "evidence_strength": evidence_strength,

        # This is a combined model score, NOT a calibrated probability.
        "confidence": round(float(top_score) * 100, 2),

        "supporting_evidence": "\n".join(supporting_evidence),
        "conflicting_evidence": "\n".join(conflicting_evidence),

        "analysis_notes": (
            "Pre-trained text and image models were used. "
            "Their suggestions are not verified facts. "
            "A staff member must inspect the evidence and confirm "
            "the actual return reason."
        ),

        "model_version": (
            f"text:{TEXT_MODEL_NAME}; image:{IMAGE_MODEL_NAME}"
        ),
    }
