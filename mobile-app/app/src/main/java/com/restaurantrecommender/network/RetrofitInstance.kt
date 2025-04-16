package com.restaurantrecommender.network

import okhttp3.ConnectionPool
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.Request
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit

object RetrofitInstance {

    private const val BASE_URL = "http://164.8.22.203:5000/"
    private const val API_KEY = "d89b5fbc-b80d-4c18-bd0e-b2444020db53"  // Add your API key here

    // Create an Interceptor to add the API key to the headers
    private val apiKeyInterceptor = Interceptor { chain ->
        val request: Request = chain.request().newBuilder()
            .addHeader("x-api-key", "$API_KEY")  // Add your API key in the Authorization header
            .build()
        chain.proceed(request)
    }

    // Create an OkHttpClient and attach the interceptor
    private val okHttpClient: OkHttpClient = OkHttpClient.Builder()
        .addInterceptor(apiKeyInterceptor)
        .connectionPool(ConnectionPool(0, 5, TimeUnit.MINUTES))
        .connectTimeout(30, TimeUnit.SECONDS)   // Increase connection timeout
        .readTimeout(30, TimeUnit.SECONDS)      // Increase read timeout
        .writeTimeout(30, TimeUnit.SECONDS)     // Increase write timeout
        .build()

    // Retrofit instance with the custom OkHttpClient
    val api: RestaurantApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(okHttpClient)  // Attach the OkHttpClient
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(RestaurantApiService::class.java)
    }
}
