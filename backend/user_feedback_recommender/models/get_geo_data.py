import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from tqdm import tqdm 


# Convert JSON data to DataFrame
df = pd.read_json("restaurant_pages.json")

# 🚀 Step 2: Initialize Geocoder
geolocator = Nominatim(user_agent="geoapi_explorer")


# 🚀 Step 3: Function to Get Coordinates
def get_coordinates(address):
    try:
        # print(f"🔍 Searching for: {address}")
        location = geolocator.geocode(address, timeout=10)  # Avoid timeout errors
        if location:
            return pd.Series([location.latitude, location.longitude])  # Return (lat, lon)
    except GeocoderTimedOut:
        print(f"⚠️ Timeout error for: {address}")
    return pd.Series([None, None])  # Return None if not found

# 🚀 Step 4: Apply Geocoding with Progress Bar
tqdm.pandas()  # Enable tqdm for pandas
df[['geo_lat', 'geo_long']] = df['address'].progress_apply(get_coordinates)


# 🚀 Step 5: Save Updated JSON
df.to_json("restaurant_pages_geo.json", orient="records", indent=4)

print("✅ Geocoding complete! Data saved as restaurant_pages_with_geo.json")
