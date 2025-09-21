import asyncio
import json
import os
import pickle
from datetime import datetime, UTC, timezone, timedelta
import requests
import torch
import torchvision
from fastapi import FastAPI, Form, UploadFile, Request, Response, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cloudinary
import cloudinary.uploader
from fastapi.exceptions import HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
import uuid
from dotenv import load_dotenv
import uvicorn
from geopy.geocoders import Nominatim
from helper_func import upload_file, authorize_user, generate_uuid, generate_report_id
from helper_func import sentence_embeddings, cluster_data
import numpy as np

load_dotenv()
app = FastAPI()

client = AsyncIOMotorClient(host = "localhost", port = 27017)
db = client["Crowd_Sourced_Ocean_Hazard_Reporting"]
user_collection = db["user"]
hotspot_collection = db["hotspot"]
report_collection = db["reports"]
twitter_stream_collection = db["twitter_stream"]

with open("models/lgbm_classifier_model.pkl", mode="rb+") as f:
    panic_classifier_model = pickle.load(f)
    print("model_loaded")

with open("models/svd.pkl", "rb") as f:
    svd = pickle.load(f)

with open("models/tf_idf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Configuring cloudinary storage for image and report storing
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

@app.post("/sign_in")
async def sign_in(user_name: str = Form(...), password: str = Form(...), role: str = Form(...), official_pwd: str = Form(None)):
    """
    API endpoint for a new user to sign in.
    :param user_name: The name of the user
    :param password: The password of the user
    :param role: The role of the user i.e. either official or citizen
    :param official_pwd: Extra password required only if the role is 'official'
    :return: Confirmation
    """
    hashed_pwd = bcrypt.hash(password)
    user_id = await generate_uuid()

    # Check if username already exists
    existing_user = await user_collection.find_one({"user_name": user_name})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists.")

    # If role is official, verify official password
    if role.lower() == "official":
        env_official_pwd = os.getenv("OFFICIAL_PWD")
        if not official_pwd or official_pwd != env_official_pwd:
            raise HTTPException(status_code=403, detail="Invalid official password. Unauthorized to register as official.")

    try:
        await user_collection.insert_one({
            "user_name": user_name,
            "user_id": user_id,
            "hashed_pwd": hashed_pwd,
            "role": role.lower()
        })
        return {"Status": "Successful", "user_id": user_id, "role": role.lower()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


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
async def upload_report(user_id, latitude: float, longitude: float, file: UploadFile = File(...)):
    """
    API endpoint for uploading a new hazard report by a user.
    :param user_id: The ID of the user submitting the report
    :param latitude: Latitude of the location
    :param longitude: Longitude of the location
    :param file: Uploaded media file
    :return: Status message
    """
    report_id = await generate_report_id()
    time = datetime.now(UTC)
    geolocator = Nominatim(user_agent="geoapi")  # Used to get the area name where the latitude and longitude lies.
    location = geolocator.reverse((latitude, longitude)).raw["address"]

    # Using so many fallbacks because geopy is inconsistent sometimes.
    city = location.get("city") or location.get("town") or location.get("village") or location.get("municipality")
    suburb = location.get("suburb") or location.get("neighbourhood") or location.get("hamlet") or location.get("county")
    state = location.get("state") or location.get("state_district")

    try:
        report_url = await upload_file(file)

        state_dict = torch.load("models/panic_meter_mobilenet_model.pth")
        model = torchvision.models.mobilenet_v2()
        model.load_state_dict(state_dict)
        with torch.inference_mode():
            model.eval()
            urgency = model.forward()

        # --------- AI MODEL PLACEHOLDER ----------
        # Replace the following stub with your actual AI tagging
        ai_tags = {
            "classification": "flood",   # e.g. "Flood", "Tsunami" etc.
            "urgency": "Medium"             # e.g. "low", "medium", "high"
        }
        # ai_tags = await run_ai_tagging_model(report_url, ...)  # <--- your model hook

        await report_collection.insert_one({
            "user_id": user_id,
            "report_id": report_id,
            "location": {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "state": state,
                "city": city,
                "suburb": suburb
            },
            "ai_tags": ai_tags,
            "timestamp": time,
            "report_url": report_url
        })

        return {"Status": "Successful", "report_id": report_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    try:
        all_reports = report_collection.find({"user_id": user_id})
        reports_list = await all_reports.to_list(length=1000)   # Added await + length cap
        if reports_list:
            return {"reports_list": reports_list}
        else:
            return {"reports_list": []}   # returning empty list instead of error
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_tweet_data")
async def get_tweets(latitude: float, longitude: float, radius: int,
                     days: int = 0, hours: int = 0, minutes: int = 30, time_limit: bool = True):
    """
    Function to get the recent tweets (Citizen Mode - on-demand fetch)
    """
    bearer_token = os.getenv("BEARER_TOKEN")
    hazard_keywords = "tsunami OR flood OR cyclone OR storm OR surge OR बारिश OR बाढ़ OR తుఫాను OR वादळ"

    query = f"({hazard_keywords}) point_radius:[{longitude} {latitude} {radius}km] -is:retweet"
    if time_limit:
        start_time = (datetime.now(timezone.utc) - timedelta(minutes=minutes, days=days, hours=hours)).isoformat()

    url = "https://api.twitter.com/2/tweets/search/recent"

    params = {
        "query": query,
        "max_results": 50,
        "tweet.fields": "id,text,created_at,lang,geo",
        "expansions": "geo.place_id",
        "place.fields": "full_name,country,geo",
    }
    if time_limit:
        params["start_time"] = start_time

    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        try:
            data = response.json()
            tweets_out = []
            for tweet in data.get("data", []):
                tweet_doc = {
                    "tweet_id": tweet["id"],
                    "created_at": tweet["created_at"],
                    "lang": tweet.get("lang"),
                    "text": tweet.get("text"),
                    "ai_tags": panic_classifier_model.predict(sentence_embeddings(text = tweet["text"], svd = svd, vectorizer = vectorizer))
                }
                tweets_out.append(tweet_doc)

            return {"tweets": tweets_out}
        except Exception as e:
            raise HTTPException(status_code=501, detail=str(e))
    else:
        raise HTTPException(status_code=501, detail="Unable to fetch twitter data.")

@app.post("/{official_id}/start_stream")
async def start_twitter_stream(official_id: str, duration_minutes: int = 30):
    """
    API endpoint to start streaming twitter data (Official Mode).
    Marks a stream session as active in the DB.
    Only allowed for registered officials.
    """
    # Verify official exists in DB
    official = await user_collection.find_one({"user_id": official_id})
    if not official:
        raise HTTPException(status_code=404, detail="User not found.")
    if official.get("role") != "official":
        raise HTTPException(status_code=403, detail="Only officials are authorized to start streams.")

    # Check if another stream is already active
    existing_stream = await report_collection.find_one({"created_stream": True})
    if existing_stream:
        raise HTTPException(status_code=400, detail="Another stream is already active.")

    start_time = datetime.now(UTC)
    end_time = start_time + timedelta(minutes=duration_minutes)

    try:
        await report_collection.insert_one({
            "user_id": official_id,
            "report_id": f"stream-{uuid.uuid4()}",
            "location": {"latitude": 0, "longitude": 0},  # not relevant for stream doc
            "timestamp": start_time,
            "report_url": "twitter data",
            "source": "social",
            "created_stream": True,
            "stream_details": {
                "official_id": official_id,
                "start_time": start_time,
                "end_time": end_time
            }
        })
        return {"Status": "Stream Started", "start_time": start_time, "end_time": end_time}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream_status")
async def get_stream_status():
    """
    API endpoint to check if a twitter stream is currently active.
    """
    active_stream = await report_collection.find_one({"created_stream": True})
    if active_stream:
        return {"active": True, "details": active_stream.get("stream_details")}
    else:
        return {"active": False}

@app.post("/{official_id}/stop_stream")
async def stop_twitter_stream(official_id: str):
    """
    API endpoint to stop the active twitter stream (Official Mode).
    Only the official who started the stream can stop it.
    """
    # Verify official exists in DB
    official = await user_collection.find_one({"user_id": official_id})
    if not official:
        raise HTTPException(status_code=404, detail="User not found.")
    if official.get("role") != "official":
        raise HTTPException(status_code=403, detail="Only officials are authorized to stop streams.")

    # Find active stream
    active_stream = await report_collection.find_one({"created_stream": True})
    if not active_stream:
        raise HTTPException(status_code=400, detail="No active stream found.")

    # Check if the same official started it
    stream_details = active_stream.get("stream_details", {})
    if stream_details.get("official_id") != official_id:
        raise HTTPException(status_code=403, detail="Only the official who started the stream can stop it.")

    try:
        await report_collection.update_one(
            {"_id": active_stream["_id"]},
            {"$set": {"created_stream": False, "stream_details": {}}}
        )
        return {"Status": "Stream Stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_social_reports/{official_id}")
async def list_social_reports(official_id: str, limit: int = 100):
    """
    API endpoint to list reports ingested from Twitter (source = 'social').
    Accessible only to officials.
    :param official_id: The unique id of the official
    :param limit: Max number of reports to return
    :return: List of social reports
    """
    # Verify official exists in DB
    official = await user_collection.find_one({"user_id": official_id})
    if not official:
        raise HTTPException(status_code=404, detail="User not found.")
    if official.get("role") != "official":
        raise HTTPException(status_code=403, detail="Only officials are authorized to view social reports.")

    try:
        cursor = report_collection.find({"source": "social"}).sort("timestamp", -1).limit(limit)
        reports_list = await cursor.to_list(length=limit)
        return {"social_reports": reports_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_citizen_reports/{user_id}")
async def list_citizen_reports(user_id: str, limit: int = 100):
    """
    API endpoint to list all reports submitted by a citizen (source = 'citizen').
    :param user_id: The unique id of the citizen
    :param limit: Max number of reports to return
    :return: List of citizen reports
    """
    # Verify user exists in DB
    user = await user_collection.find_one({"user_id": user_id})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.get("role") != "citizen":
        raise HTTPException(status_code=403, detail="Only citizens can access their own reports.")

    try:
        cursor = report_collection.find({"user_id": user_id, "source": "citizen"}).sort("timestamp", -1).limit(limit)
        reports_list = await cursor.to_list(length=limit)
        return {"citizen_reports": reports_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/{official_id}/hotspots/run_clustering")
async def run_hotspot_clustering(official_id: str, method: str = "auto", min_samples: int = 3):
    """
    Run DBSCAN/HDBSCAN clustering on reports and generate hotspots.
    Officials only.
    """
    # Verify official
    official = await user_collection.find_one({"user_id": official_id})
    if not official:
        raise HTTPException(status_code=404, detail="User not found.")
    if official.get("role") != "official":
        raise HTTPException(status_code=403, detail="Only officials are authorized.")

    try:
        # Fetch report locations
        cursor = report_collection.find({}, {"location.latitude": 1, "location.longitude": 1, "ai_tags": 1, "report_id": 1})
        reports = await cursor.to_list(length=5000)

        if not reports:
            return {"Status": "No reports available for clustering."}

        coords = np.array([[r["location"]["latitude"], r["location"]["longitude"]] for r in reports])

        # Run clustering
        results = cluster_data(coords, method=method)

        labels = results["labels"]
        n_clusters = results["n_clusters"]

        # Clear previous hotspots
        await hotspot_collection.delete_many({})

        # Insert new hotspots
        hotspot_docs = []
        for cluster_id in range(n_clusters):
            cluster_points = [reports[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not cluster_points:
                continue

            lats = [p["location"]["latitude"] for p in cluster_points]
            lons = [p["location"]["longitude"] for p in cluster_points]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)

            hotspot_docs.append({
                "hotspot_id": str(uuid.uuid4()),
                "location_center": f"{center_lat},{center_lon}",
                "report_count": len(cluster_points),
                "urgency_level": np.mean([1 if p["ai_tags"]["urgency"] == "high" else 0.5 if p["ai_tags"]["urgency"] == "medium" else 0 for p in cluster_points]),
                "reports": [p["report_id"] for p in cluster_points]
            })

        if hotspot_docs:
            await hotspot_collection.insert_many(hotspot_docs)

        return {
            "Status": "Clustering complete",
            "hotspots_created": len(hotspot_docs),
            "method_used": results["method"].upper()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/hotspots/list")
async def list_hotspots(min_urgency: float = None, limit: int = 100):
    """
    API endpoint to list hotspots detected in the system.
    :param min_urgency: Filter to only show hotspots above this urgency
    :param limit: Max number of hotspots to return
    :return: List of hotspots
    """
    try:
        query = {}
        if min_urgency is not None:
            query["urgency_level"] = {"$gte": float(min_urgency)}

        cursor = hotspot_collection.find(query).sort("urgency_level", -1).limit(limit)
        hotspots = await cursor.to_list(length=limit)
        return {"hotspots": hotspots}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hotspots/{official_id}/{hotspot_id}/details")
async def hotspot_details(official_id: str, hotspot_id: str):
    """
    API endpoint to fetch details of a specific hotspot.
    Only officials are allowed.
    :param official_id: The id of the official
    :param hotspot_id: The id of the hotspot
    :return:  details including report_ids
    """
    # Verify official
    official = await user_collection.find_one({"user_id": official_id})
    if not official:
        raise HTTPException(status_code=404, detail="User not found.")
    if official.get("role") != "official":
        raise HTTPException(status_code=403, detail="Only officials are authorized to access hotspot details.")

    try:
        hotspot = await hotspot_collection.find_one({"hotspot_id": hotspot_id})
        if not hotspot:
            raise HTTPException(status_code=404, detail="Hotspot not found.")
        return {"hotspot": hotspot, "linked_reports": hotspot.get("reports", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/list_all")   # For frontend engineers only, so, they can fetch all the details without the need of user login.
async def list_all_reports(limit: int = 500):
    cursor = report_collection.find({}).sort("timestamp", -1).limit(limit)
    reports = await cursor.to_list(length=limit)
    return {"reports": reports}


# Fake apis for prototype showcase
async def event_generator():
    cursor = twitter_stream_collection.find({}, {"_id": 0})    # No need to await the find query because it by default returns a cursor object.
    async for doc in cursor:
        # Convert datetime to string for JSON
        doc["Time"] = doc["Time"].isoformat()
        await asyncio.sleep(2)  # simulating delay
        yield f"data: {json.dumps(doc)}\n\n"

@app.get("/fake_stream")
async def fake_stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app = app, port = 8001)