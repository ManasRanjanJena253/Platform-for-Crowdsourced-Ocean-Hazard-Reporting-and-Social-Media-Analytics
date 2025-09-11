"""
Fetch tweets using Twitter (X) API v2 with a point_radius filter.
This example searches within a circle around a given latitude & longitude.

Steps:
1. Install deps: `pip install requests python-dotenv`
2. Put your BEARER_TOKEN in a `.env` file
3. Run this script
"""
import os
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1) Load Bearer Token
# ---------------------------------------------------------
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# ---------------------------------------------------------
# 2) Coordinates and radius
# ---------------------------------------------------------
latitude = 17.6868    # Example: Vizag, Andhra Pradesh
longitude = 83.2185
radius_km = 25        # Radius in km

# ---------------------------------------------------------
# 3) Build query
# ---------------------------------------------------------
# Keywords for hazards (add more if needed)
hazard_keywords = "tsunami OR flood OR cyclone OR storm OR surge OR बारिश OR बाढ़ "

# NOTE: We do NOT restrict by lang:en (so it supports Hindi, Tamil, Telugu, Bengali, etc.)
# The point_radius filter requires: longitude latitude radius
query = f"({hazard_keywords}) point_radius:[{longitude} {latitude} {radius_km}km] -is:retweet"

# ---------------------------------------------------------
# 4) Endpoint and parameters
# ---------------------------------------------------------
url = "https://api.twitter.com/2/tweets/search/recent"

params = {
    "query": query,
    "max_results": 10,  # its value can be 10 to 100.
    "tweet.fields": "id,text,created_at,lang,geo",
    "expansions": "geo.place_id",
    "place.fields": "full_name,country,geo"
}

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# ---------------------------------------------------------
# 5) Make request
# ---------------------------------------------------------
response = requests.get(url, headers=headers, params=params)

# ---------------------------------------------------------
# 6) Process response
# ---------------------------------------------------------
if response.status_code == 200:
    data = response.json()
    print("✅ Tweets near your coordinates:\n")

    for tweet in data.get("data", []):
        print(f"Tweet ID: {tweet['id']}")
        print(f"Time: {tweet['created_at']}")
        print(f"Lang: {tweet['lang']}")
        print(f"Text: {tweet['text']}\n")

    if "includes" in data and "places" in data["includes"]:
        for place in data["includes"]["places"]:
            print("Place Info:", place)
else:
    print("Error:", response.status_code, response.text)