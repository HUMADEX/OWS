# setup.py
'''
Setup script to download neccessary NLTK packages.
'''

import nltk

# Download nltk packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')