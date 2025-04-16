package com.restaurantrecommender

import android.webkit.JavascriptInterface
import android.content.Context

class WebAppInterface(private val context: Context) {
    private val userPreferences = UserPreferences(context)

    @JavascriptInterface
    fun getLatitude(): String? {
        return userPreferences.latitude
    }

    @JavascriptInterface
    fun getLongitude(): String? {
        return userPreferences.longitude
    }
}