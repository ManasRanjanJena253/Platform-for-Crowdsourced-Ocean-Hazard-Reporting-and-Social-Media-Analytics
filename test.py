import tweepy
import pandas as pd
import time
from itertools import cycle
import os
from geopy.geocoders import Nominatim
import reverse_geocoder as rg
from dotenv import load_dotenv

load_dotenv()

# ======================
#  Twitter API v2 setup
# ======================
bearer_token = os.getenv("BEARER_TOKEN")

client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# ======================
# Disaster Keywords
# ======================
keywords = ["flood", "cyclone", "tsunami", "storm surge", "बारिश", "बाढ़", "सुनामी", "disaster", "SOS"]
query = " OR ".join(keywords) + " -is:retweet lang:en"

# ======================
# Coastal city lat-long (India)
# ======================
coastal_coords = [
    ("Visakhapatnam", 17.6868, 83.2185),
    ("Kochi", 9.9312, 76.2673),
    ("Goa", 15.2993, 74.1240),
    ("Pondicherry", 11.9416, 79.8083)
]

coords_cycle = cycle(coastal_coords)

# ======================
# CSV setup
# ======================
csv_file = "backend/ML_models/data/tweets.csv"
if not os.path.exists(csv_file):
    df_init = pd.DataFrame(columns=["time", "location", "city", "text", "latitude", "longitude"])
    df_init.to_csv(csv_file, index=False)

# Geopy setup
geolocator = Nominatim(user_agent="disaster_tweets_geocoder")

# ======================
# Fetch loop
# ======================
try:
    while True:
        city_name, lat, lon = next(coords_cycle)
        print(f"Fetching tweets near {city_name}...")

        # Approximate city keyword for v2 since geo search is limited
        city_query = f"{query} {city_name}"

        try:
            tweets_response = client.search_recent_tweets(
                query=city_query,
                max_results=100,
                tweet_fields=["created_at","geo","lang"],
                expansions=["geo.place_id"],
                place_fields=["full_name","geo"]
            )

            data = []

            if tweets_response.data:
                for tweet in tweets_response.data:
                    user_loc = city_name  # v2 recent search doesn't reliably provide user location
                    text = tweet.text
                    created_at = tweet.created_at

                    # Geocode city for lat-long
                    latitude, longitude = None, None
                    try:
                        location = geolocator.geocode(user_loc)
                        if location:
                            latitude, longitude = location.latitude, location.longitude
                        else:
                            results = rg.search((lat, lon))
                            latitude = results[0]['lat']
                            longitude = results[0]['lon']
                    except Exception:
                        results = rg.search((lat, lon))
                        latitude = results[0]['lat']
                        longitude = results[0]['lon']

                    data.append([created_at, user_loc, city_name, text, latitude, longitude])

            if data:
                df = pd.DataFrame(data, columns=["time", "location", "city", "text", "latitude", "longitude"])
                df.to_csv(csv_file, mode="a", header=False, index=False)
                print(f"Saved {len(data)} tweets from {city_name}.")

            time.sleep(2)

        except tweepy.TweepyException as e:
            print(f"Error fetching tweets for {city_name}: {e}")
            time.sleep(60)

except Exception as final_error:
    print(f"Stopped due to error: {final_error}")
