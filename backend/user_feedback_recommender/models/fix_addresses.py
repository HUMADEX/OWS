import requests
import pandas as pd
from tqdm import tqdm
from fuzzywuzzy import process


# Function to fetch geocoding data and find best match
def get_best_address(address):
    url = f"https://photon.komoot.io/api/?q={address}"
    response = requests.get(url).json()

    if 'features' not in response:
        print(f"No features found for address: {address}")
        return None, None, None
    
    new_addresses = []
    coordinates_map = {}

    # Extract addresses from API response
    for res in response["features"]:
        try:
            street = res["properties"]["street"]
            house_no = res["properties"]["housenumber"]
            city = res["properties"]["city"]
            postcode = res["properties"]["postcode"]
            country = res["properties"]["country"]

            full_address = f"{street} {house_no}, {city} {postcode} {country}"
            new_addresses.append(full_address)

            # Store coordinates for each address
            coordinates_map[full_address] = res["geometry"]["coordinates"]  # [lon, lat]

        except KeyError:
            continue  # Skip if any key is missing

    # Fuzzy match to find the closest address
    if new_addresses:
        matches = process.extract(address, new_addresses, limit=3)
        best_match, best_score = max(matches, key=lambda x: x[1])  # Get best address (highest score)
        if best_score >= 80:
            print(f"Best score: {best_score}")
            best_coords = coordinates_map[best_match]  # Get coordinates (lon, lat)
            print(f"Old address: {address}")
            print(f"{best_match}, {best_coords[1]}, {best_coords[0]}")
            return [best_match, float(best_coords[1]), float(best_coords[0])] # (address, lat, lon)
    
    return [None, None, None] # Return None if no match


tqdm.pandas()

df = pd.read_json('restaurant_pages_geo_fixed.json')
# df = df[:20]

# df["best_address"] = None
# mask = df['geo_lat'].isnull()
# df.loc[mask, ["best_address", "geo_lat", "geo_long"]] = df.loc[mask, "address"].apply(get_best_address)
# print(f"len: {len(df.loc[df['geo_lat'].isnull(), ["best_address", "geo_lat", "geo_long"]])}")

# df.loc[df['geo_lat'].isnull(), ["best_address", "geo_lat", "geo_long"]] = data_list = df.loc[df['geo_lat'].isnull(), "address"].progress_apply(
#     get_best_address
# ).apply(pd.Series)
# print(f"len data list: {len(data_list)}")
# print(data_list)

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Addresses"):
    if pd.isnull(row["geo_lat"]):  # Only process missing geo locations
        best_address, lat, lon = get_best_address(row["address"])
        df.at[idx, "best_address"] = best_address
        df.at[idx, "geo_lat"] = lat
        df.at[idx, "geo_long"] = lon

df.reset_index(drop=True, inplace=True)
df.to_json("restaurant_pages_geo_fixed1.json", orient="records", indent=4)