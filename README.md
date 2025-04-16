# ows-restaurant-recommender

The Restaurant Recommender is a privacy-focused system helping users discover restaurants based on preferences, context (e.g., time, events), and location. It includes an Android app, backend server, and web interface, ensuring seamless cross-platform use with minimal, privacy-preserving data sharing.

The Android mobile app is the primary interface for users to interact with the restaurant recommendation engine. The app allows users to:

Discover restaurants in Slovenia based on cuisine preferences, time of day (e.g., lunch or dinner), and location. Filter restaurants using personalized recommendations based on previous interactions (e.g., clicked restaurants, user ratings, and sentiment). View restaurants on a map, using Google Maps interface. Privacy-preserving search: The app stores user preferences locally in a XML file on the phone, ensuring that minimal data is shared with the server. Only the complemented query, which includes essential information like cuisine preferences, time, and location, is sent to the backend for collaborative filtering.

The backend server handles the recommendation logic and processes queries from both the mobile app and web interface. The key features include:

Collaborative filtering engine: The server runs the collaborative filtering model to provide personalized restaurant recommendations based on user preferences and data. Minimal data exchange: The backend only receives a minimal complemented query from the mobile app or web interface, ensuring that sensitive user data remains on the user's device. Contextual post-filtering: After the server processes the query, the results are adjusted locally on the user's phone, taking into account contextual information like calendar events and sentiment analysis from user interactions. Database of restaurants: The backend stores a database of restaurants, sourced from TripAdvisorincluding information like cuisine type, ratings, and operating hours. A subset of this data is downloaded and maintained locally to improve search efficiency and privacy at the edge (on-device or on-region servers).
