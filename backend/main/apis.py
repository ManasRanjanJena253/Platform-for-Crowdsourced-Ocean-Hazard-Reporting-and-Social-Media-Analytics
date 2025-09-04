import os
from datetime import datetime, UTC
import requests
from fastapi import FastAPI, Form, UploadFile, Request, Response, File
from fastapi.middleware.cors import CORSMiddleware
import cloudinary
import cloudinary.uploader
from fastapi.exceptions import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
import uuid
from dotenv import load_dotenv
import uvicorn
import tempfile
import shutil
from geopy.geocoders import Nominatim

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
        return {"Status": "Successful", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))

@app.post("/login")
async def login(user_name: str = Form(...), password: str = Form(...)):
    """
    API endpoint for logging in pre-registered users
    :param user_name: The name of the user
    :param password: The password of the user
    :return: Confirmation
    """
    user_id = await authorize_user(user_name, password)
    if user_id:
        return {"user_id": user_id}
    else:
        raise HTTPException(status_code = 400, detail = "Invalid Credentials")

@app.post("/{user_id}/upload_report")
async def upload_report(user_id, latitude, longitude, file: UploadFile = File(...)):
    report_id = await generate_report_id()
    time = datetime.now(UTC)
    geolocator = Nominatim(user_agent="geoapi")      # Used to get the area name where the latitude and longitude lies.
    location = geolocator.reverse((latitude, longitude)).raw["address"]
    print(location)
    # Using so, many or statements to handle fallbacks as the return of geolocator is inconsistent and sometime may not have some keys.
    city = location.get("city") or location.get("town") or location.get("village") or location.get("municipality")
    suburb = location.get("suburb") or location.get("neighbourhood") or location.get("hamlet") or location.get("county")
    state = location.get("state") or location.get("state_district")

    # Currently haven't added the ai tags as the models are not trained yet.
    try:
        report_url = await upload_file(file)
        await report_collection.insert_one({"user_id": user_id, "report_id": report_id,
                                            "location": {
                                                         "latitude": float(latitude),
                                                         "longitude": float(longitude),
                                                         "state": state,
                                                         "city": city,
                                                         "suburb": suburb
                                                         },
                                            "timestamp": time, "report_url": report_url})
        return {"Status": "Successful"}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

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
    if role_check:
        if role_check["role"] == "official":
            try:
                ai_tags = report_data["ai_tags"]
                urgency_level = ai_tags["urgency"]
                calamity_type = ai_tags["classification"]
                url = report_data["report_url"]
                return {"urgency_level": urgency_level, "calamity_type": calamity_type, "report_url": url}
            except Exception as e:
                return {"ERROR": str(e)}

        else:
            raise HTTPException(status_code = 401, detail = "Unauthorized User, Data can be accessed only by officials.")
    else:
        raise HTTPException(status_code = 500, detail = "No report with specified credentials found.")

@app.post("/list_reports/{user_id}")
async def list_reports_by_user_id(user_id):
    """
    API endpoint to get all the reports posted by a particular user_id
    :param user_id: The unique id of the user
    :return: Details about the user reports
    """
    all_reports = await report_collection.find({"user_id": user_id})
    reports_list = all_reports.to_list()
    if reports_list:
        return {"reports_list": reports_list}
    else:
        raise HTTPException(status_code = 400, detail = "No reports submitted yet.")


if __name__ == "__main__":
    uvicorn.run(app = app, port = 8001)