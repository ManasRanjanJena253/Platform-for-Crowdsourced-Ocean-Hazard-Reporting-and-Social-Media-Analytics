import os
import shutil
import tempfile
import uuid
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt

load_dotenv()
app = FastAPI()

client = AsyncIOMotorClient(host = "localhost", port = 27017)
db = client["Crowd_Sourced_Ocean_Hazard_Reporting"]
user_collection = db["user"]
hotspot_collection = db["hotspot"]
report_collection = db["reports"]

# Configuring cloudinary storage for image and report storing
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

async def upload_file(file: UploadFile, folder: str = "hazard_reports_by_user"):
    """
    Uploads a file (from FastAPI UploadFile) to Cloudinary and returns its secure URL.
    """
    try:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete = False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Upload to Cloudinary
        result = cloudinary.uploader.upload(
            tmp_path,
            folder = folder,
            resource_type = "auto"  # auto-detects image/video/pdf
        )
        return result["secure_url"]

    except Exception as e:
        print(f"Cloudinary upload failed: {e}")
        return None


async def authorize_user(user_name: str, password):
    """
    Function to authorize the user
    :param user_name: The name of the user
    :param password: The password of the user
    :return: Boolean, whether the credentials match or not.
    """
    user = await user_collection.find_one({"user_name": user_name}, {"_id": 0, "user_id": 1, "hashed_pwd": 1})
    if user and bcrypt.verify(password, user["hashed_pwd"]):
        return user["user_id"]
    return False

async def generate_uuid():

    user_id = uuid.uuid4()
    uuid_check = await user_collection.find_one({"user_id": str(user_id)})

    if uuid_check:
        while uuid_check:
            user_id = uuid.uuid4()
            uuid_check = await user_collection.find_one({"user_id": str(user_id)})

        return str(user_id)

    else:
        return str(user_id)

async def generate_report_id():

    report_id = uuid.uuid4()
    uuid_check = await report_collection.find_one({"report_id": str(report_id)})

    if uuid_check:
        while uuid_check:
            report_id = uuid.uuid4()
            uuid_check = await report_collection.find_one({"report_id": str(report_id)})

        return str(report_id)

    else:
        return str(report_id)
