package com.restaurantrecommender

import android.content.Context
import android.content.SharedPreferences
import android.provider.Settings
import android.util.Log
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.util.*

class UserPreferences(context: Context) {

    private val sharedPreferences: SharedPreferences =
        context.getSharedPreferences("user_prefs", Context.MODE_PRIVATE)
    private val gson = Gson()

    companion object {
        private const val KEY_USER_ID  = "user_id"
        private const val KEY_CITY = "city"
        private const val KEY_SEARCH_QUERY = "search_query"
        private const val KEY_TOP_RESTAURANTS = "top_restaurants"
        private const val KEY_CLICKED_RESTAURANTS = "clicked_restaurants"
        private const val KEY_CLICKED_RESTAURANT_STYLES = "clicked_restaurant_styles"
        private const val KEY_CLICKED_RESTAURANT_PRICES = "clicked_restaurant_prices"
        private const val KEY_CLICKED_RESTAURANT_MEALS = "clicked_restaurant_meals"
        private const val KEY_CLICKED_RESTAURANT_FEATURES = "clicked_restaurant_features"
        private const val KEY_CLICKED_RESTAURANT_RATINGS = "clicked_restaurant_ratings"
        private const val KEY_USER_STYLES = "user_styles"
        private const val KEY_USER_PRICES = "user_prices"
        private const val KEY_USER_MEALS = "user_meals"
        private const val KEY_USER_FEATURES = "user_features"
        private const val KEY_USER_RATINGS = "user_ratings"
        private const val MAX_ITEMS = 100
    }

    private var clickedRestaurants: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_CLICKED_RESTAURANTS, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_CLICKED_RESTAURANTS, json).apply()
        }

    // Property for price
    var restaurantStyles: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_CLICKED_RESTAURANT_STYLES, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_CLICKED_RESTAURANT_STYLES, json).apply()
        }
    // Property for price
    var restaurantPrices: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_CLICKED_RESTAURANT_PRICES, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_CLICKED_RESTAURANT_PRICES, json).apply()
        }

    var restaurantMeals: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_CLICKED_RESTAURANT_MEALS, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_CLICKED_RESTAURANT_MEALS, json).apply()
        }
    var restaurantFeatures: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_CLICKED_RESTAURANT_FEATURES, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_CLICKED_RESTAURANT_FEATURES, json).apply()
        }

    var restaurantRatings: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_CLICKED_RESTAURANT_RATINGS, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_CLICKED_RESTAURANT_RATINGS, json).apply()
        }

    // Property for price
    var userPrice: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_USER_PRICES, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_USER_PRICES, json).apply()
        }

    // Property for style
    var userStyles: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_USER_STYLES, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_USER_STYLES, json).apply()
        }

    var userMeals: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_USER_MEALS, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_USER_MEALS, json).apply()
        }
    var userFeatures: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_USER_FEATURES, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_USER_FEATURES, json).apply()
        }

    var userRatings: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_USER_RATINGS, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_USER_RATINGS, json).apply()
        }

    var userId: String?
        get() {
            // If the user ID doesn't exist, generate one and save it
            var id = sharedPreferences.getString(KEY_USER_ID, null)
            if (id.isNullOrEmpty()) {
                id = UUID.randomUUID().toString()
                sharedPreferences.edit().putString(KEY_USER_ID, id).apply()
            }
            return id
        }
        set(value) = sharedPreferences.edit().putString(KEY_USER_ID, value).apply()

    var searchQuery: List<String>
        get() {
            val json = sharedPreferences.getString(KEY_SEARCH_QUERY, null)
            val type = object : TypeToken<List<String>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_SEARCH_QUERY, json).apply()
        }

    var topRestaurants: List<List<String>>
        get() {
            val json = sharedPreferences.getString(KEY_TOP_RESTAURANTS, null)
            val type = object : TypeToken<List<List<String>>>() {}.type
            return gson.fromJson(json, type) ?: emptyList()
        }
        set(value) {
            val updatedList = manageListSize(value)
            val json = gson.toJson(updatedList)
            sharedPreferences.edit().putString(KEY_TOP_RESTAURANTS, json).apply()
        }
    var latitude: String?
        get() = sharedPreferences.getString("latitude", null)
        set(value) = sharedPreferences.edit().putString("latitude", value).apply()

    var longitude: String?
        get() = sharedPreferences.getString("longitude", null)
        set(value) = sharedPreferences.edit().putString("longitude", value).apply()

    // Function to add a restaurant to the list of clicked restaurants
    fun addClickedRestaurant(restaurantName: String) {
        val currentList = clickedRestaurants.toMutableList()
        currentList.add(restaurantName)
        clickedRestaurants = currentList
    }

    private fun <T> manageListSize(list: List<T>): List<T> {
        return if (list.size > MAX_ITEMS) {
            list.takeLast(MAX_ITEMS)
        } else {
            list
        }
    }

    fun addSearchQuery(query: String) {
        val currentList = searchQuery.toMutableList()
        currentList.add(query)
        searchQuery = currentList
    }

    fun addTopRestaurants(restaurants: List<String>) {
        val currentList = topRestaurants.toMutableList()
        currentList.add(restaurants)
        topRestaurants = currentList
    }

    // Function to add a price entry
    fun addRestaurantPrice(priceEntry: String) {
        val currentList = restaurantPrices.toMutableList()
        currentList.add(priceEntry)
        restaurantPrices = currentList
    }

    // Function to add a style entry
    fun addRestaurantStyle(styleEntries: Any) {
        val currentList = restaurantStyles.toMutableList()
        when (styleEntries) {
            is String -> currentList.add(styleEntries) // Add a single string entry
            is List<*> -> currentList.addAll(styleEntries.filterIsInstance<String>()) // Add all valid strings from a list
            else -> Log.e("addRestaurantStyle", "Invalid input type: ${styleEntries::class.java}")
        }
        restaurantStyles = currentList
    }

    // Function to add a meals entry
    fun addRestaurantMeals(mealsEntries: Any) {
        val currentList = restaurantMeals.toMutableList()
        when (mealsEntries) {
            is String -> currentList.add(mealsEntries) // Add a single string entry
            is List<*> -> currentList.addAll(mealsEntries.filterIsInstance<String>()) // Add all valid strings from a list
            else -> Log.e("addRestaurantMeals", "Invalid input type: ${mealsEntries::class.java}")
        }
        restaurantMeals = currentList
    }

    // Function to add a features entry
    fun addRestaurantFeatures(featuresEntries: Any) {
        val currentList = restaurantFeatures.toMutableList()
        when (featuresEntries) {
            is String -> currentList.add(featuresEntries) // Add a single string entry
            is List<*> -> currentList.addAll(featuresEntries.filterIsInstance<String>()) // Add all valid strings from a list
            else -> Log.e("addRestaurantFeatures", "Invalid input type: ${featuresEntries::class.java}")
        }
        restaurantFeatures = currentList
    }

//    // Function to add a ratings entry
//    fun addRestaurantRatings(priceEntry: String) {
//        val currentList = restaurantRatings.toMutableList()
//        currentList.add(priceEntry)
//        userRatings = currentList
//    }

    // Function to add a price entry
    fun addUserPrice(priceEntry: String) {
        val currentList = userPrice.toMutableList()
        currentList.add(priceEntry)
        userPrice = currentList
    }

    // Function to add a style entry
    fun addUserStyle(styleEntry: String) {
        val currentList = userStyles.toMutableList()
        currentList.add(styleEntry)
        userStyles = currentList
    }

    // Function to add a meals entry
    fun addUserMeals(priceEntry: String) {
        val currentList = userMeals.toMutableList()
        currentList.add(priceEntry)
        userMeals = currentList
    }

    // Function to add a features entry
    fun addUserFeatures(priceEntry: String) {
        val currentList = userFeatures.toMutableList()
        currentList.add(priceEntry)
        userFeatures = currentList
    }

    fun getUserPreferenceList(key: String): List<String> {
        val json = sharedPreferences.getString(key, null) ?: return emptyList()

        return try {
            val type = object : TypeToken<List<String>>() {}.type
            gson.fromJson(json, type) ?: emptyList()
        } catch (e: Exception) {
            Log.e("UserPreferences", "Error parsing preferences for key: $key", e)
            emptyList()
        }
    }
//    // Function to add a ratings entry
//    fun addUserRatings(priceEntry: String) {
//        val currentList = userRatings.toMutableList()
//        currentList.add(priceEntry)
//        userRatings = currentList
//    }

}
