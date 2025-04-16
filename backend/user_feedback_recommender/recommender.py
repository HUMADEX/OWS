import pandas as pd
import numpy as np
import time
import torch
import json
import ast
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from user_feedback_recommender.models.sbert_model import SBERTModel
from user_feedback_recommender.preprocess import process_sentences
from logging_config import LoggerConfig
from math import radians, cos, sin, sqrt, atan2

# Initialize logger
logger = LoggerConfig.get_logger("recommender")

# Specify the CUDA device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Ensure model is loaded at the start of the server
s_model = SBERTModel.get_model()

df = pd.read_parquet("user_feedback_recommender/restaurant_pages.parquet")
# ✅ Load the precomputed embeddings
embeddings_data = torch.load("user_feedback_recommender/restaurant_embeddings_all-mpnet-base-v2.pt", map_location=device)
#restaurant_embeddings_all-mpnet-base-v2.pt
#restaurant_embeddings_all-MiniLM-L6-v2.pt
# ✅ Convert lists to PyTorch tensors
record_ids = embeddings_data["record_ids"]
text_embeddings = embeddings_data["text_embeddings"].to(device)
bow_embeddings = embeddings_data["bow_embeddings"].to(device)

# ✅ Merge embeddings with DataFrame based on `record_id`
embedding_map = {record_id: (text, bow) for record_id, text, bow in zip(record_ids, text_embeddings, bow_embeddings)}
df["text_embedding"] = df["record_id"].map(lambda x: embedding_map[x][0] if x in embedding_map else torch.zeros(384, device=device))
df["bow_embedding"] = df["record_id"].map(lambda x: embedding_map[x][1] if x in embedding_map else torch.zeros(384, device=device))


# # Columns that might contain lists
columns_to_fix = [
    'cuisines',
    'meals',
    'price range', 
    'features', 
    'reviews', 
    'filtered_reviews', 
    'cuisines_p',
    'meals_p',
    'features_p',
    'filtered_reviews_p'
]

"""Convert JSON-encoded strings back to lists before returning the response."""
for col in columns_to_fix:
    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else x)

# # Convert embeddings back to PyTorch tensors
# df['text_embedding'] = df['text_embedding'].apply(
#     lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, (list, np.ndarray)) else x
# )
# df['bow_embedding'] = df['bow_embedding'].apply(
#     lambda x: torch.tensor(x, dtype=torch.float32) if isinstance(x, (list, np.ndarray)) else x
# )
# df = pd.read_json("user_feedback_recommender/restaurant_pages.json")


# Haversine formula to calculate distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in km


# Filter restaurants based on location type (city or coordinates)
def filter_restaurants(df, city, location, radius):
    logger.debug(f"Filtering by location: {city}, {location}, {radius}")
    if city and city.strip() != "":  # City case
        logger.debug(f"Filtering by city: {city}")
        city = city.lower()
        return df[df["address"].str.lower().str.contains(city, na=False)]  # Filter by city

    else:  # Coordinates case
        try:
            location = ast.literal_eval(location)
            logger.debug(f"Location type: {type(location)}")
            user_lat, user_lon = float(location[0]), float(location[1])
            logger.debug(f"Filtering by coordinates: {user_lat}, {user_lon}")
            if radius <= 0:
                radius = 2  # Default radius in km
            df["distance_km"] = df.apply(lambda row: haversine(user_lat, user_lon, row["geo_lat"], row["geo_long"]), axis=1)
            return df[df["distance_km"] <= radius]  # Return restaurants within radius
        except ValueError:
            logger.debug("Invalid coordinates. Returning all restaurants.")
            return df  # If invalid coordinates, return all restaurants

    return df  # Default: return all restaurants

class RestaurantRecommender:
    @staticmethod
    def recommend_with_bm25(description, df):
        """Search using BM25 for ranking restaurants based on bag-of-words similarity."""
        logger.debug(f"Recommending with bm25")
        start_time = time.time()

        # ✅ Rank results
        df = df.copy()
        
        # ✅ Preprocess & tokenize BoW for BM25
        df["bag_of_words"] = df["bag_of_words"].fillna("").str.lower().str.split()

        # ✅ Create BM25 index
        bm25 = BM25Okapi(df["bag_of_words"].tolist())
        
        # ✅ Tokenize query
        query_tokens = description.lower().split()
        logger.debug(f"BM25 Search query tokens: {query_tokens}")
        
        # ✅ Compute BM25 scores
        scores = bm25.get_scores(query_tokens)
        
        df["similarity"] = scores
        df = df.sort_values(by="similarity", ascending=False)
        df = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

        logger.debug(f"BM25 Search completed in {time.time() - start_time:.4f} seconds")
        return df
    
    @staticmethod
    def recommend_with_sbert(description, df):
        logger.debug(f"Recommending with sbert")
        # Encode descriptions with SBERT
        # restaurant_embeddings = s_model.encode(df["fullText"].tolist(), convert_to_tensor=True)
        
        # Ensure device consistency
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        query_embedding = s_model.encode(description, convert_to_tensor=True).to(device)

        # Use precomputed embeddings instead of encoding plain_text again
        restaurant_embeddings = torch.stack(df["text_embedding"].tolist()).to(device)  # Convert list of tensors to a tensor matrix
        bow_embeddings = torch.stack(df["bow_embedding"].tolist()).to(device)
        df = df.copy()
        
        # logger.debug(f"Query embedding shape:{query_embedding.shape}")
        # logger.debug(f"Restaurant embeddings shape: {restaurant_embeddings.shape}")
        
        # Compute cosine similarity
        similarities = s_model.compute_similarity(query_embedding, bow_embeddings)[0].cpu().numpy()
        
        # Compute cosine similarity
        # similarities = torch.nn.functional.cosine_similarity(query_embedding, restaurant_embeddings, dim=1).cpu().numpy()
        df.loc[:, 'similarity'] = similarities
        df = df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
        df = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
        
        # logger.debug(f"Top Similarity Scores: {sorted(similarities, reverse=True)[:10]}")  # Top 10 scores
        # logger.debug(f"Lowest Similarity Scores: {sorted(similarities)[:10]}")  # Lowest 10 scores

        return df

    @staticmethod
    def recommend_with_tfidf(description, df):
        logger.debug(f"Recommending with tfidf")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df["bag_of_words"].tolist())
        query_vector = vectorizer.transform([description])
        df = df.copy()
        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        df.loc[:, 'similarity'] = similarities
        df = df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
        df = df.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

        return df

    
    @staticmethod
    def recommend(description, extracted_entities, city, location, radius, method='bm25'):
        logger.debug(f"Ranking with method: {method}")
        start_time = time.time()
        
        description = process_sentences(description)
        logger.debug(f"Processed description: {description}")
        logger.debug(f"Processing description took {time.time() - start_time:.2f} seconds")
        
        filtered_df = filter_restaurants(df, city, location, radius)
        logger.debug(f"Filtered df: {len(filtered_df)}")
        
        if method == "sbert":
            ranked_df = RestaurantRecommender.recommend_with_sbert(description, filtered_df)
        elif method == "tfidf":
            ranked_df = RestaurantRecommender.recommend_with_tfidf(description, filtered_df)
        elif method == "bm25":
            ranked_df = RestaurantRecommender.recommend_with_bm25(description, filtered_df)
        elif method == "hybrid":
            top_df = RestaurantRecommender.recommend_with_bm25(description, filtered_df)
            logger.debug("After recommending with bm25, use sbert for semantic search.")
            logger.debug(f"Length Top df: {len(top_df)}")
            logger.debug(f"Top 5 from bm25: {top_df[['name', 'cuisines_p', 'meals_p', 'features_p','similarity']][:5]}")
            ranked_df = RestaurantRecommender.recommend_with_sbert(description, top_df)
        else:
            raise ValueError("Invalid method. Choose 'sbert', 'tfidf', or 'bm25'.")
        
        total_time = time.time() - start_time
        logger.debug(f"Total recommendation process took {total_time:.2f} seconds")
        logger.debug(f"Len df: {len(ranked_df[:20])}")
        
        return ranked_df[[
            'name', 
            'url',
            'address', 
            'phone number',
            'cuisines', 
            'meals', 
            'price range',
            'price_category',
            'features_p', 
            'filtered_reviews',
            'geo_lat', 
            'geo_long', 
            'similarity'
        ]][:20]

