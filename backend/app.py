import requests

# Import modules
from flask import Flask
from flask import request, render_template, jsonify
from flask_restful import Resource, Api
import numpy as np
import pandas as pd
# Import RecommendationSystem classese from recommender module
from user_feedback_recommender.recommender import RestaurantRecommender
from user_feedback_recommender.models.llama_extract_categories import extract_entities
from logging_config import LoggerConfig

# Get centralized logger
logger = LoggerConfig.get_logger(__name__)

# Flask app
app = Flask(__name__)
# Create an API instance for Flask app
api = Api(app)

# Create instance of RecommendationSystem class
restaurant_recommender = RestaurantRecommender()

logger.info("Starting Flask application...")

class Recommend(Resource):
    def get(self):
        logger.debug("GET /recommend - Test log entry")
        return {'recommend': 'ready'}

    def post(self):
        data = request.get_json(force=True)
        logger.debug(f"Incoming request to /recommend_new: {data}")  # Log the incoming request
        user_input = data.get('input')
        extracted_entities = data.get('entities')
        location = data.get('location')
        city = data.get('city').lower()
        radius = data.get('radius')

        if user_input:
            recommendations = restaurant_recommender.recommend(
                description=user_input, 
                extracted_entities=extracted_entities,
                city=city,
                location=location,
                radius=radius
            )
            # recommendations = recommendations.fillna(value=np.nan)
            recommendations = recommendations.replace({np.nan: None})

            return {'recommendations': recommendations.to_dict(orient='records')}
        else:
            return {'error': 'Invalid input'}, 400

class EntityRecognition(Resource):
    def post(self):
        data = request.get_json(force=True)
        logger.debug(f"Incoming request to 78/extract_entities: {data}")
        sentence = data.get("input")
        if not sentence:
            return jsonify({"error": "Invalid input, sentence is required"}), 400

        # **Call the model function**
        response = extract_entities(sentence)
        logger.debug(f"Outgoing response from 78/extract_entities: {response}")

        return response

# Add the RecommendNew resource to the API with the '/recommend_new' endpoint
api.add_resource(Recommend, '/recommend')
# EntityRecognition API endpoint
api.add_resource(EntityRecognition, '/extract_entities')

