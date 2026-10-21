
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY")

    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "sqlite:///return_insight.db"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    MAX_CONTENT_LENGTH = 40 * 1024 * 1024  # 40 MB per HTTP request

    UPLOAD_FOLDER = str(BASE_DIR / "instance" / "uploads")

    MAX_IMAGE_SIZE = 8 * 1024 * 1024  # 8 MB per image
    MAX_RECEIVED_IMAGES = 5

    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
