from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geoapi")
location = geolocator.reverse((17.6868, 83.2185))  # lat, lon for Vizag
print(location.raw['address'])
