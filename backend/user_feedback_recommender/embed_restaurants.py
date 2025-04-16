import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

HF_CACHE_DIR_MINI_LM_L6_V2 = "/home/user/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9"

HF_CACHE_DIR_MPNET_BASE_V2 = "/home/user/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/9a3225965996d404b775526de6dbfe85d3368642"
df = pd.read_parquet('restaurant_pages.parquet')

# Specify the CUDA device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load model
s_model = SentenceTransformer(HF_CACHE_DIR_MPNET_BASE_V2)

# ✅ Store record IDs for validation
record_ids = df['record_id'].tolist()

# ✅ Generate text embeddings
text_embeddings = torch.stack(list(df['plain_text'].apply(
    lambda x: s_model.encode(x, convert_to_tensor=True).to(device) if x else torch.zeros(768).to(device)
)))

# ✅ Generate bag-of-words embeddings
bow_embeddings = torch.stack(list(df['bag_of_words'].apply(
    lambda x: s_model.encode(x, convert_to_tensor=True).to(device) if x else torch.zeros(768).to(device)
)))


# ✅ Save embeddings and record IDs in a `.pt` file
torch.save({
    "record_ids": record_ids,
    "text_embeddings": text_embeddings,
    "bow_embeddings": bow_embeddings
}, 'restaurant_embeddings_all-mpnet-base-v2.pt')


print("✅ Embeddings generated and saved to 'restaurant_embeddings_all-mpnet-base-v2.pt'")
