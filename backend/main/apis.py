import os
from datetime import datetime, UTC
import requests
from fastapi import FastAPI, Form, UploadFile, Request, Response
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


@app.post("/sign_in")
async def sign_in(user_name: str = Form(...), password: str = Form(...), role: str = Form(...)):
    """
    API endpoint for a new user to sign in.
    :param user_name: The name of the user
    :param password: The password of the user
    :param role: The role of the user i.e. either official or citizen
    :return: Confirmation
    """
    hashed_pwd = bcrypt.hash(password)
    user_id = await generate_uuid()
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
    report_id = await generate_report_id()
    time = datetime.now(UTC)

    # Currently haven't added the ai tags as the models are not trained yet.
    try:
        report_url = upload_file(file)
        await report_collection.insert_one({"user_id": user_id, "report_id": report_id, "location": location, "timestamp": time, "report_url": report_url})
        return {"Status": "Successful"}
    except Exception as e:
        raise HTTPException(status_code = 304, detail = str(e))

@app.post("/{user_id}/{report_id}")
async def get_report(user_id, report_id):   # Currently it supports only image reports submitted by the citizen.
    """
    API endpoint to get the uploaded image report.
    :param user_id: The unique id of the user
    :param report_id: The unique id of the report.
    :return: JSON
    """
    role_check = await user_collection.find_one({"user_id": user_id})
    report_data = await report_collection.find_one({"report_id": report_id})
    if role_check["role"] == "official":
        try:
            url = report_data["report_url"]
            r = requests.get(url, stream = True)
            return Response(content = r.content, media_type = "image/jpeg")
        except Exception as e:
            return {"ERROR": str(e)}

    else:
        raise HTTPException(status_code = 401, detail = "Unauthorized User, Data can be accessed only by officials.")

@app.post("/{user_id}/{report_id}/details")
async def get_report_details(user_id, report_id):     # Currently ai models are not trained so, only these details provided, later multiple more details regarding
                                                      # the report will also be provided.
    """
    API endpoint for getting all the details associated to the report_id
    :param user_id: The unique id of the user
    :param report_id: The unique id of the report
    :return: All the details in a JSON format
    """
    role_check = await user_collection.find_one({"user_id": user_id})
    report_data = await report_collection.find_one({"report_id": report_id})
    if role_check["role"] == "official":
        try:
            ai_tags = report_data["ai_tags"]
            urgency_level = ai_tags["urgency"]
            calamity_type = ai_tags["classification"]
            return {"urgency_level": urgency_level, "calamity_type": calamity_type}
        except Exception as e:
            return {"ERROR": str(e)}

    else:
        raise HTTPException(status_code = 401, detail = "Unauthorized User, Data can be accessed only by officials.")

if __name__ == "__main__":
    uvicorn.run(app = app, port = 8001)