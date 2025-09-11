import os
import requests
from dotenv import load_dotenv

load_dotenv()

def get_tweets(latitude, longitude, radius):
    """
    Function to get the recent tweets
    :param latitude: Latitude of the place you want to retrieve
    :param longitude:
    :param radius:
    :return:
    """
    url = "https://api.twitter.com/2/tweets/search/recent"

    bearer_token = os.getenv("BEARER_TOKEN")
    hazard_keywords = "tsunami OR flood OR cyclone OR storm OR surge OR बारिश OR बाढ़ OR తుఫాను OR वादळ"

    query = f"({hazard_keywords}) point_radius:[{longitude} {latitude} {radius}km] -is:retweet"
