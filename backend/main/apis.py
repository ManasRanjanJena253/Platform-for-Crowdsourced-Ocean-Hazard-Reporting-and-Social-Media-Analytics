import os
from datetime import datetime, UTC
from fastapi import FastAPI, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
from fastapi.exceptions import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
import uuid
from dotenv import load_dotenv
import uvicorn

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

def upload_file(file_path: str, folder: str = "hazard_reports_by_user"):
    """
    Uploads a file to Cloudinary and returns its secure URL.

    :param file_path: Local path to the file to upload
    :param folder: Cloudinary folder name to store the file
    :return: Secure URL of the uploaded file
    """
    try:
        result = cloudinary.uploader.upload(
            file_path,
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
    check = await user_collection.find_one({"user_name": user_name, "hashed_pwd": password}, {"_id": 0, "user_id": 1})
    if check:
        return check["user_id"]
    else:
        return False

async def generate_uuid():
    """
    Used to generate the user_id
    :return: user_id
    """
    user_id = uuid.uuid4()
    uuid_check = await user_collection.find_one({"user_id": user_id})

    if uuid_check:
        while uuid_check:
            user_id = uuid.uuid4()
            uuid_check = await user_collection.find_one({"user_id": user_id})

        return user_id

    else:
        return user_id

async def generate_report_id():
    """
    Used to generate the report_id
    :return: report_id
    """
    report_id = uuid.uuid4()
    uuid_check = await report_collection.find_one({"report_id": report_id})

    if uuid_check:
        while uuid_check:
            report_id = uuid.uuid4()
            uuid_check = await report_collection.find_one({"report_id": report_id})

        return report_id

    else:
        return report_id


@app.post("/sign_in")
async def sign_in(user_name: str = Form(...), password: str = Form(...), role: str = Form(...)):
    """
    API endpoint for a new user to sign in.
    :param user_name: The name of the user
    :param password: The password of the user
    :param role: The role of the user i.e. either official or citizen
    :return: Confirmation
    """
    hashed_pwd = bcrypt(password.encode())
    user_id = generate_uuid()
    try:
        await user_collection.insert_one({"user_name": user_name, "user_id": user_id, "hashed_pwd": hashed_pwd, "role": role.lower()})
        return {"Status": "Successful"}
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

@app.post("/login")
async def login(user_name: str = Form(...), password: str = Form(...), role: str = Form(...)):
    """
    API endpoint for logging in pre-registered users
    :param user_name: The name of the user
    :param password: The password of the user
    :param role: The role of the user i.e. either official or citizen
    :return: Confirmation
    """
    user_id = await authorize_user(user_name, password)
    if user_id:
        return user_id
    else:
        raise HTTPException(status_code = 400, detail = "Invalid Credentials")

@app.post("/{user_id}/upload_report")
async def upload_report(user_id, location, file = UploadFile(...)):
    report_id = generate_report_id()
    time = datetime.now(UTC)

    # Currently haven't added the ai tags as the models are not trained yet.
    try:
        report_url = upload_file(file)
        await report_collection.insert_one({"user_id": user_id, "report_id": report_id, "location": location, "timestamp": time, "report_url": report_url})
        return {"Status": "Successful"}
    except Exception as e:
        raise HTTPException(status_code = 304, detail = str(e))



if __name__ == "__main__":
    uvicorn.run(app = app, reload = True, port = 8001)