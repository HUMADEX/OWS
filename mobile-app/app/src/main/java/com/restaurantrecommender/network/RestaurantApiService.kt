package com.restaurantrecommender.network

import com.restaurantrecommender.RecommendationResponse
import okhttp3.RequestBody
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.Headers
import retrofit2.http.POST

interface RestaurantApiService {
    @Headers("Content-Type: application/json")
    @POST("recommend") // Adjust endpoint URL as per your server setup
    fun getRestaurants(
        @Body jsonPayload: RequestBody
    ): Call<RecommendationResponse>

    @POST("extract_entities") // Ensure this matches the API endpoint
    fun extractEntities(@Body jsonPayload: Map<String, String>): Call<Map<String, List<String>>>
}