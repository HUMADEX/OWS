import pandas as pd
import re
import json
import nltk
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


# Process sentences method
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df = pd.read_json("user_feedback_recommender/restaurant_pages.json")
md = pd.read_parquet('user_feedback_recommender/metadata.parquet')

# Regex pattern to check for special characters
diacritic_pattern = re.compile(r'[^\x00-\x7F]')


def process_sentences(text):
    
    temp_sent =[]

    # Tokenize words
    words = nltk.word_tokenize(text)

    # Lemmatize each of the words based on their position in the sentence
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):  # only verbs
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        
        # Remove stop words and non alphabet tokens
        if lemmatized not in stop_words and lemmatized.isalpha(): 
            temp_sent.append(lemmatized)

    # Some other clean-up
    full_sentence = ' '.join(temp_sent)
    full_sentence = full_sentence.replace("n't", " not")
    full_sentence = full_sentence.replace("'m", " am")
    full_sentence = full_sentence.replace("'s", " is")
    full_sentence = full_sentence.replace("'re", " are")
    full_sentence = full_sentence.replace("'ll", " will")
    full_sentence = full_sentence.replace("'ve", " have")
    full_sentence = full_sentence.replace("'d", " would")
    return full_sentence

# Function to check and convert if necessary
def normalize_if_needed(text):
    text = text.strip().lower()
    if diacritic_pattern.search(text):  # Check if there are special characters
        return unidecode(text)
    return  ""

def categorize_price(price):
    """
    Categorizes the price range into 'cheap', 'moderate', or 'expensive'.
    - Extracts numeric values if available.
    - Counts dollar signs if numeric values are missing.
    """
    if not isinstance(price, str) or price.strip() == '':
        return ""  # Handle missing values

    price = price.replace(" ", "").lower()  # Normalize spaces & case
    
    # Check if price contains numbers (e.g., "$10-20", "$5-$15")
    numbers = re.findall(r'\d+', price)
    
    if numbers:
        numbers = list(map(int, numbers))  # Convert to integers
        print(f"numbers: {numbers}")
        avg_price = sum(numbers) / len(numbers)  # Compute average price
        print(f"avg_price: {avg_price}")
        if avg_price <= 10:  
            return "cheap"
        elif avg_price <= 30:  
            return "moderate"
        else:
            return "expensive"
    else:
        # If no numbers, count dollar signs
        parts = price.split("-")
        print(f"parts: {parts}")
        avg_price = sum(part.count("$") for part in parts) / len(parts)
        print(f"avg_price: {avg_price}")
        # Assign categories based on price level
        if avg_price <= 1.5:  
            return "cheap"
        elif avg_price <= 2.5:  
            return "moderate"
        else:
            return "expensive"

df = df.merge(md[['record_id', 'plain_text']], on='record_id', how='left')
df['plain_text'] = df['plain_text'].apply(process_sentences)

# Preprocess textual data
df["filtered_reviews_p"] = df["filtered_reviews"].apply(lambda x: [review.lower() for review in x] if isinstance(x, list) else x)
df["cuisines_p"] = df["cuisines"].apply(lambda x: [review.lower() for review in x] if isinstance(x, list) else x)
df["meals_p"] = df["meals"].apply(lambda x: [meal.lower() for meal in x] if isinstance(x, list) else x)
df["features_p"] = df["features"].apply(lambda x: 
    [str(feature).lower() for feature in x.values()] if isinstance(x, dict) else
    [str(feature).lower() for feature in x] if isinstance(x, list) else x
)
df['address_p'] = df['address'].apply(lambda x: x.lower() if isinstance(x, str) else x)
df["price_category"] = df["price range"].apply(categorize_price)

# Apply sentence processing
df['filtered_reviews_p'] = df['filtered_reviews_p'].apply(lambda x: [process_sentences(review) for review in x] if isinstance(x, list) else x)
df['cuisines_p'] = df['cuisines_p'].apply(lambda x: [process_sentences(review) for review in x] if isinstance(x, list) else x)
df['meals_p'] = df['meals_p'].apply(lambda x: [process_sentences(review) for review in x] if isinstance(x, list) else x)
df['features_p'] = df['features_p'].apply(lambda x: [process_sentences(review) for review in x] if isinstance(x, list) else x)


# Construct bag of words
df["bag_of_words"] = df["filtered_reviews_p"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " + \
                     df["cuisines_p"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " + \
                     df["meals_p"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " + \
                     df["features_p"].apply(lambda x: " ".join(x) if isinstance(x, list) else "") + " " + \
                     df["address_p"].fillna("") + " " + \
                     df["price_category"].fillna("") + " " + df['name'].apply(normalize_if_needed)
                     
# Columns that might contain lists
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

for col in columns_to_fix:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)

df['ratings'] = df['ratings'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else ("" if pd.isna(x) else str(x)))
df['price range'] = df['price range'].str.replace(r'[\[\]]', '', regex=True)

# Save processed data
df.to_parquet('restaurant_pages1.parquet', index=False)