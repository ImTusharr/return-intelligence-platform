
import os
import mimetypes
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_STORAGE_BUCKET", "return-images")

UPLOAD_FOLDER = Path("instance/uploads")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials are missing from .env")

if not UPLOAD_FOLDER.is_dir():
    raise RuntimeError(f"Upload folder not found: {UPLOAD_FOLDER}")

client = create_client(SUPABASE_URL, SUPABASE_KEY)
storage = client.storage.from_(BUCKET_NAME)

files = [path for path in UPLOAD_FOLDER.iterdir() if path.is_file()]

print(f"Local files found: {len(files)}")
print(f"Target bucket: {BUCKET_NAME}")

try:
    existing_objects = storage.list()
    existing_names = {
        item.name if hasattr(item, "name") else item.get("name")
        for item in existing_objects
    }
except Exception as error:
    raise RuntimeError(
        f"Could not list bucket objects. No files were uploaded. {error}"
    ) from error

uploaded = 0
skipped = 0
failed = []

for path in files:
    filename = path.name

    if filename in existing_names:
        print(f"SKIPPED (already exists): {filename}")
        skipped += 1
        continue

    content_type, _ = mimetypes.guess_type(filename)
    content_type = content_type or "application/octet-stream"

    try:
        with path.open("rb") as file:
            storage.upload(
                filename,
                file.read(),
                file_options={
                    "content-type": content_type,
                    "upsert": "false",
                },
            )

        print(f"UPLOADED: {filename}")
        uploaded += 1

    except Exception as error:
        print(f"FAILED: {filename} - {error}")
        failed.append(filename)

print("\nUpload summary")
print(f"Uploaded: {uploaded}")
print(f"Already existed: {skipped}")
print(f"Failed: {len(failed)}")

if failed:
    print("Files needing attention:")
    for filename in failed:
        print(f"- {filename}")
else:
    print("All local files are present in Supabase Storage.")
