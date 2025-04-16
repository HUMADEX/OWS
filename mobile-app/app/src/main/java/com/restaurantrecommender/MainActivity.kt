package com.restaurantrecommender

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Resources
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.webkit.ConsoleMessage
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDropDown
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Phone
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationServices
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import com.restaurantrecommender.network.RetrofitInstance
import com.restaurantrecommender.ui.theme.RestaurantRecommenderTheme
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import okhttp3.MediaType
import okhttp3.RequestBody
import org.json.JSONObject
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader


data class Restaurant(
    val name: String,
    val url: String,
    val address: String,
    @SerializedName("phone number")
    val phone: String,
    val cuisines: List<String>? = null,
    val meals: List<String>? = null,
    @SerializedName("price range")
    val price: String,
    @SerializedName("price_category")
    val priceCat: String,
    @SerializedName("features_p")
    val features: List<String>? = null,
    @SerializedName("filtered_reviews")
    val reviews: List<String>? = null,
    @SerializedName("geo_lat")
    val latitude: Double? = null,
    @SerializedName("geo_long")
    val longitude: Double? = null,
    val similarity: Double
)

data class RecommendationResponse(
    val recommendations: List<Restaurant>
)

class MainActivity : ComponentActivity() {

    private lateinit var fusedLocationClient: FusedLocationProviderClient

    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)

        // Initialize the WebView
        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            webViewClient = WebViewClient()
        }
        setupWebView()

        enableEdgeToEdge()
        setContent {
            RestaurantRecommenderTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    val resources = resources
                    ContentWithTitle(
                        modifier = Modifier.padding(innerPadding),
                        resources = resources,
                        webView = webView
                    )
                }
            }
        }
        requestLocationPermission()
    }

    private fun requestLocationPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED -> {
                // Permission is already granted, fetch location
                getCurrentLocation()
            }
            shouldShowRequestPermissionRationale(Manifest.permission.ACCESS_FINE_LOCATION) -> {
                // Show an explanation to the user as to why the permission is needed
                Log.d("Location", "Show rationale for location permission")
            }
            else -> {
                // Directly request for permission
                requestPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
            }
        }
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            // Permission granted, fetch the location
            getCurrentLocation()
        } else {
            Log.d("Location", "Permission denied")
        }
    }

    private fun getCurrentLocation() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            fusedLocationClient.lastLocation
                .addOnSuccessListener { location ->
                    if (location != null) {
                        val latitude = location.latitude
                        val longitude = location.longitude
                        Log.d("Location", "Current location: Latitude: $latitude, Longitude: $longitude")

                        // Save location to user preferences
                        val userPreferences = UserPreferences(this)
                        userPreferences.latitude = latitude.toString()
                        userPreferences.longitude = longitude.toString()
                    } else {
                        Log.d("Location", "Location is null")
                    }
                }
                .addOnFailureListener { exception ->
                    Log.e("Location", "Failed to get location: ${exception.message}")
                }
        } else {
            Log.d("Location", "Location permission not granted")
        }
    }
    private fun setupWebView() {
        val webAppInterface = WebAppInterface(this)
        webView.addJavascriptInterface(webAppInterface, "Android")

        webView.settings.javaScriptEnabled = true
        webView.settings.domStorageEnabled = true

        // Capture start time right before loadUrl
        val mapLoadStart = System.currentTimeMillis()

        // Enable fullscreen functionality
        webView.webChromeClient = object : WebChromeClient() {
            private var customView: View? = null
            private var fullscreenCallback: CustomViewCallback? = null
            override fun onShowCustomView(view: View?, callback: CustomViewCallback?) {
                customView = view
                fullscreenCallback = callback

                // Add the view to the activity
                (window.decorView as FrameLayout).addView(view)
                window.decorView.systemUiVisibility = (
                        View.SYSTEM_UI_FLAG_FULLSCREEN
                                or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                                or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                        )
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }

            override fun onHideCustomView() {
                // Remove the custom view
                customView?.let {
                    (window.decorView as FrameLayout).removeView(it)
                    customView = null
                    fullscreenCallback = null
                }
                // Revert UI flags
                window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_VISIBLE
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }

            override fun onConsoleMessage(message: ConsoleMessage?): Boolean {
                Log.d("WebView", "Console Message: ${message?.message()}")
                return super.onConsoleMessage(message)
            }
        }

        webView.webViewClient = object : WebViewClient() {
            @Deprecated("Deprecated in Java")
            override fun shouldOverrideUrlLoading(view: WebView?, url: String?): Boolean {
                return false // Ensure navigation stays within the WebView
            }
            override fun onPageFinished(view: WebView?, url: String?) {
                super.onPageFinished(view, url)
                val mapLoadEnd = System.currentTimeMillis()
                val duration = mapLoadEnd - mapLoadStart
                Log.d("Performance", "Map load time: $duration ms")
            }
        }

        webView.loadUrl("file:///android_asset/map.html")
    }

}

@Composable
fun ContentWithTitle(modifier: Modifier = Modifier, resources: Resources,  webView: WebView) {
    val context = LocalContext.current
    val userPreferences = remember { UserPreferences(context) }

    val userId by remember { mutableStateOf(userPreferences.userId ?: "") }
    // State for search query
    var searchQuery by remember { mutableStateOf("") }

    // State for search results
    var searchResults by remember { mutableStateOf<List<Restaurant>>(emptyList()) }

    // State for loading status
    var isLoading by remember { mutableStateOf(false) }

    val keyboardController = LocalSoftwareKeyboardController.current
    val coroutineScope = rememberCoroutineScope()
    // State for search submission
    var hasSearched by remember { mutableStateOf(false) }
    // State to control WebView visibility
    var webViewVisible by remember { mutableStateOf(false) }
    // ✅ New States for City Dropdown & Radius Slider
    val cityOptions = loadCitiesFromCSV(context)
    var selectedCity by remember { mutableStateOf<String?>(null) }
    var radiusKm by remember { mutableStateOf(1f) } // Default radius 1t km

    // Function to make API call
    fun searchRestaurants(query: String, city: String?, radius: Float, userPreferences: UserPreferences, context: Context) {
        // Start measuring search time
        val overallStart = System.currentTimeMillis()

        isLoading = true
        hasSearched = true
        // ✅ Get user coordinates
        val userCoordinates = listOf(userPreferences.latitude, userPreferences.longitude)
        // Show the map WebView after search
        webViewVisible = true
        // Save search query
        userPreferences.addSearchQuery(query)

        fun callNERApi(query: String, callback: (Map<String, List<String>>) -> Unit) {
            val jsonPayload = mapOf("input" to query)
            Log.d("NER_API", "Calling API with: $jsonPayload")

            val call = RetrofitInstance.api.extractEntities(jsonPayload)
            call.enqueue(object : Callback<Map<String, List<String>>> {
                override fun onResponse(
                    call: Call<Map<String, List<String>>>,
                    response: Response<Map<String, List<String>>>
                ) {
                    if (response.isSuccessful) {
                        val nerResult = response.body() ?: emptyMap()

                        // Log the full structured response
                        Log.d("NER_API", "NER Response: $nerResult")
                        callback(nerResult)  // Send results back
                    } else {
                        Log.e("NER_API", "Failed to extract entities: ${response.code()}")
                        callback(emptyMap())  // Handle failure gracefully
                    }
                }

                override fun onFailure(call: Call<Map<String, List<String>>>, t: Throwable) {
                    val request = call.request()
                    val requestBody = request.body()?.toString() ?: "No Body"
                    val requestUrl = request.url().toString()
                    val requestHeaders = request.headers().toString()

                    Log.e("NER_API", "API Call failed!")
                    Log.e("NER_API", "Request URL: $requestUrl")
                    Log.e("NER_API", "Request Headers: $requestHeaders")
                    Log.e("NER_API", "Request Body: $requestBody")
                    Log.e("NER_API", "Error Message: ${t.localizedMessage}", t)  // Print full error stack trace

                    callback(emptyMap())  // Handle error gracefully
                }

            })
        }

        fun callRecommendationApi(query: String, extractedEntities: Map<String, List<String>>, userPreferences: UserPreferences, onComplete: () -> Unit)
        {
            // Prepare the JSON payload
            val jsonPayload = JSONObject().apply {
                put("input", query) // String input
                put("location", userCoordinates)
                put("city", city)
                put("radius", radius)
                put("entities", JSONObject(extractedEntities)) // Convert extractedEntities to JSON
            }.toString()

            // ✅ Convert JSON String to RequestBody
            val requestBody: RequestBody = RequestBody.create(MediaType.parse("application/json"), jsonPayload)
            Log.d("Search", "Making API call with query: $query")
            Log.d("Search", "Making API call jsonpayload: $jsonPayload")
            Log.d("Search", "Making API call requestBody: $requestBody")
            val call =  RetrofitInstance.api.getRestaurants(requestBody)

            call.enqueue(object : Callback<RecommendationResponse> {
                override fun onResponse(call: Call<RecommendationResponse>, response: Response<RecommendationResponse>) {
                    isLoading = false
                    if (response.isSuccessful) {
                        searchResults = response.body()?.recommendations ?: emptyList()
                        // ✅ Add this line to load markers on the map
                        loadMapWithMarkers(webView, searchResults)
                    } else {
                        Log.d("Search", "New API call failed: ${response.code()}")
                    }
                    // IMPORTANT: signal completion
                    onComplete()
                }
                override fun onFailure(call: Call<RecommendationResponse>, t: Throwable) {
                    isLoading = false
                    Log.e("Search", "New API call error: ${t.message}")
                    onComplete()
                }
            })
        }

        Log.d("Search", "Calling NER API...")
        // Example: measure how long until NER completes
        val startTimeNer = System.currentTimeMillis()
        callNERApi(query) { extractedEntities ->
            Log.d("Search", "Extracted Entities: $extractedEntities")
            val endTimeNer = System.currentTimeMillis()
            Log.d("Performance", "NER API call took: ${endTimeNer - startTimeNer} ms")
            val missingCategories = mutableListOf<String>()
            // ✅ Convert extractedEntities to a MutableMap so we can modify it
            val modifiedEntities: MutableMap<String, List<String>> = extractedEntities.mapValues { it.value as List<String> }.toMutableMap()

            var updatedQuery = query

            // Iterate over the extracted entities and update user preferences
            extractedEntities.forEach { (category, entityList) ->
                if (entityList.isNotEmpty()) {
                    // Call the corresponding method based on the category
                    when (category) {
                        "cuisine" -> entityList.forEach { userPreferences.addUserStyle(it) }
                        "meals" -> entityList.forEach { userPreferences.addUserMeals(it) }
                        "pricing category" -> entityList.forEach { userPreferences.addUserPrice(it) }
                        "features" -> entityList.forEach { userPreferences.addUserFeatures(it) }
                    }
                } else {
                    Log.d("Search", "category is empty: $category")
                    // If the list is empty, add category to missing list
                    missingCategories.add(category)
                }
            }

            // Log missing categories
            if (missingCategories.isNotEmpty()) {
                Log.d("Search", "Missing Categories: $missingCategories")

                // Try to complement missing categories using user context
                missingCategories.forEach { category ->
//                    if (category == "location") {
//                        // Handle missing location separately
//                        val userLatitude = userPreferences.latitude
//                        val userLongitude = userPreferences.longitude
//
//                        if (userLatitude != null && userLongitude != null) {
//                            Log.d("Search", "Location is missing in NER response. Using user location: $userLatitude, $userLongitude")
//                            modifiedEntities["location"] = listOf("$userLatitude", "$userLongitude")
//                        } else {
//                            Log.w("Search", "Location is missing in NER response, and no user location is available.")
//                        }
//                    } else {
                        val dominantPreference = analyzeUserContext(userPreferences, category)
                        if (dominantPreference != null) {
                            modifiedEntities[category] = listOf(dominantPreference)
                            Log.d("Search", "Complemented $category with user preference: $dominantPreference")
                            updatedQuery += " $dominantPreference"
                            Log.d("Search", "New query: $updatedQuery")
                        } else {
                            Log.d("Search", "No user preference found for $category")
                        }
//                    }
                }
            }
            Log.d("Search", "Extracted Entities: $extractedEntities")
            Log.d("Search", "modifiedEntities: $modifiedEntities")
            Log.d("Search", "NER done. Calling Recommendation API now with query:" +
                    " $updatedQuery")
            // Then you call your recommendation API
            val startTimeReco = System.currentTimeMillis()
            callRecommendationApi(updatedQuery, modifiedEntities,userPreferences){
                // Recommendation API complete
                val endTimeReco = System.currentTimeMillis()
                val recoDuration = endTimeReco - startTimeReco
                Log.d("Performance", "Recommendation API call took: $recoDuration ms")

                // Finally, measure total time from start of search to final results
                val overallEnd = System.currentTimeMillis()
                val totalDuration = overallEnd - overallStart
                Log.d("Performance", "Total time from search to displayed results: $totalDuration ms")
            }
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = Color.Black//
    ) {
        Column(modifier = modifier.padding(24.dp)) {
            Text(
                text = "Restaurant Recommender",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(30.dp)) // Add some space between title and greeting

            // Map Section
            if (webViewVisible) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .fillMaxHeight(0.35f)
                ) {
                    LocalWebView(webView = webView)
                }
            }

            Spacer(modifier = Modifier.height(8.dp))
            // Map Section ended

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp) // Add spacing between components
            ) {
                // ✅ City Selector Dropdown (With Placeholder)
                var expanded by remember { mutableStateOf(false) }

                Box(
                    modifier = Modifier
                        .weight(0.3f) // Controls the dropdown width
                        .border(1.dp, Color.Gray, RoundedCornerShape(8.dp)) // Add border
                        .clickable { expanded = true }
                        .background(
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(8.dp)
                        )
                        .padding(5.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Row (
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ){
                        Text(
                            text = selectedCity ?: "Select City", // Placeholder logic
                            color = Color.White,//MaterialTheme.colorScheme.primary,
                            modifier = Modifier.weight(1f)
                        )
                        Icon(imageVector = Icons.Default.ArrowDropDown, contentDescription = "Dropdown Arrow")
                    }

                    DropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false },
                        modifier = Modifier
                            .background(MaterialTheme.colorScheme.surface)
                        ) {
                        LazyColumn(
                            modifier = Modifier
                                .width(150.dp)
                                .height(250.dp)
                                .fillMaxWidth()
                        ) {
                            items(cityOptions) { city ->
                                DropdownMenuItem(
                                    text = { Text(city ?: "No City Selected") },
                                    onClick = {
                                        selectedCity = city
                                        expanded = false
                                    }
                                )
                            }
                        }
                    }
                }

                // ✅ Radius Slider
                Column(
                    modifier = Modifier.weight(0.7f), // Controls slider width
                ) {
                    Text(
                        text = "Or search for restaurants in ${radiusKm.toInt()} km radius",
                        color = Color.White,
                        fontSize = 12.sp
                    )// Show radius value above slider
                    Slider(
                        value = radiusKm,
                        onValueChange = { radiusKm = it },
                        valueRange = 1f..50f, // Range: 1 to 50 km
                        steps = 49, // 1 step per km
                        modifier = Modifier.fillMaxWidth(1f)
                    )
                }
            }

            Spacer(modifier = Modifier.height(8.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ){
                TextField(
                    value = searchQuery,
                    onValueChange = { newQuery ->
                        searchQuery = newQuery
                        hasSearched=false},
                    placeholder = { Text("Search for a restaurant") },
                    modifier = Modifier
                        .weight(1f) // Ensures the TextField takes available space
                        .padding(end = 8.dp), // Adds spacing between TextField and Button,
                    keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                    keyboardActions = KeyboardActions(
                        onDone = {
                            coroutineScope.launch {
                                searchRestaurants(
                                    searchQuery,
                                    selectedCity ?: "",
                                    radiusKm,
                                    userPreferences,
                                    context
                                )
                                delay(500) // Delay for 0.5 seconds (500 milliseconds)
                                keyboardController?.hide()
                            }
                        }
                    )
                )
                IconButton(
                    onClick = {
                        userPreferences.userId = userId
                        coroutineScope.launch {
                            searchRestaurants(searchQuery,selectedCity ?: "", radiusKm, userPreferences, context)
                            delay(500) // Delay for 0.5 seconds (500 milliseconds)
                            keyboardController?.hide()
                        }
                    },
                    modifier = Modifier.align(Alignment.CenterVertically)
                ) {
                    Icon(
                        imageVector = Icons.Default.Search, // Default search icon from Material Design
                        contentDescription = "Search",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            }
            Spacer(modifier = Modifier.height(16.dp)) // Add space after button
            // Show filtered restaurants only if searchQuery is not empty
            if (isLoading) {
                Text(text = "Loading...")
            } else if (searchResults.isEmpty() && hasSearched) {
                if (searchQuery.isNotEmpty()) {
                    Text(text = "No restaurants found.", color = MaterialTheme.colorScheme.error)
                }
            } else {
                LazyColumn {
                    items(searchResults) { restaurant ->
                        RestaurantCard(restaurant = restaurant, webView = webView)
                        Spacer(modifier = Modifier.height(3.dp))
                    }
                }
            }
        }
    }
}
@Composable
fun RestaurantCard(restaurant: Restaurant, webView: WebView) {
    val uriHandler = LocalUriHandler.current
    val context = LocalContext.current
    val userPreferences = UserPreferences(context)

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .clickable {
                userPreferences.addClickedRestaurant(restaurant.name)
                restaurant.cuisines?.let { userPreferences.addRestaurantStyle(it) }
                restaurant.meals?.let { userPreferences.addRestaurantMeals(it) }
                restaurant.features?.let { userPreferences.addRestaurantFeatures(it) }
                userPreferences.addRestaurantPrice(restaurant.priceCat)
                webView.evaluateJavascript(
                    """
                    (function() {
                        if (typeof markersLayerGroup === 'undefined') {
                            console.log('markersLayerGroup is undefined.');
                            return null;
                        }
                        var markers = markersLayerGroup.getLayers();
                        for (var i = 0; i < markers.length; i++) {
                            if (markers[i].getPopup().getContent().includes("${restaurant.name}")) {
                                map.flyTo(markers[i].getLatLng(), 18, { animate: true, duration: 1.4 });
                                setTimeout(() => { markers[i].openPopup(); }, 1600);
                                return 'success';
                            }
                        }
                        return null;
                    })();
                    """.trimIndent()
                ) { result -> Log.d("WebView", "Marker click triggered: $result") }
            },
        shape = MaterialTheme.shapes.medium,
        elevation = CardDefaults.cardElevation(10.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(modifier = Modifier.padding(10.dp)) {
            Text(
                text = restaurant.name,
                style = MaterialTheme.typography.titleMedium.copy(fontWeight = FontWeight.Bold)
            )
            Spacer(modifier = Modifier.height(2.dp))
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.clickable { uriHandler.openUri(restaurant.url) }
            ) {
                Icon(
                    imageVector = Icons.Default.Public, // 🌍 World Icon
                    contentDescription = "Website",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(12.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Visit Restaurant Page",
                    style = TextStyle(
                        fontSize = 12.sp,
                        color = Color.Blue,
                        textDecoration = TextDecoration.Underline
                    )
                )
            }
            Spacer(modifier = Modifier.height(2.dp))
            // Address & Phone
            FlowRow {
                InfoRow(Icons.Default.LocationOn, "Address", restaurant.address)
                InfoRow(Icons.Default.Phone, "Phone", restaurant.phone)
            }

            // Cuisines & Meals inline
            FlowRow {
                restaurant.cuisines?.let { InlineInfoList("Cuisines", it) }
                restaurant.meals?.let { InlineInfoList("Meals", it) }

                Text(
                    text = buildAnnotatedString {
                        withStyle(style = SpanStyle(fontWeight = FontWeight.Bold)) {
                            append("Price: ")
                        }
                        append(restaurant.price)
                    },
                    style = MaterialTheme.typography.bodySmall
                )
                restaurant.features?.let { InlineInfoList("Features", it) }
            }
            // Reviews (only show a few)
//            FlowRow {
//                restaurant.reviews?.take(2)?.let { InlineInfoList("Reviews", it, maxChars = 100) }
//            }
        }
    }
}



@OptIn(ExperimentalLayoutApi::class)
@Composable
fun FlowRow(
    modifier: Modifier = Modifier,
    padding: Dp = 0.dp, // Default padding
    horizontalSpacing: Dp = 2.dp, // Default horizontal spacing
    verticalSpacing: Dp = 1.dp, // Default vertical spacing
    content: @Composable () -> Unit
) {
    androidx.compose.foundation.layout.FlowRow(
        modifier = modifier
            .fillMaxWidth()
            .padding(padding),
        horizontalArrangement = Arrangement.spacedBy(horizontalSpacing),
        verticalArrangement = Arrangement.spacedBy(verticalSpacing)
    ) {
        content()
    }
}

@Composable
fun InlineInfoList(label: String, items: List<String>, maxChars: Int = 50) {
    Text(
        text = "$label: ",
        style = MaterialTheme.typography.bodySmall.copy(fontWeight = FontWeight.Bold)
    )

    items.forEachIndexed { index, item ->
        val truncatedText = if (item.length > maxChars) item.take(maxChars) + "..." else item
        Text(
            text = if (index < items.size - 1) "$truncatedText; " else truncatedText,
            style = MaterialTheme.typography.bodySmall
        )
    }
}


// Helper function for info rows with icons
@Composable
fun InfoRow(icon: ImageVector, label: String, value: String) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 2.dp)
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = MaterialTheme.colorScheme.primary,
            modifier = Modifier.size(12.dp)
        )
        Spacer(modifier = Modifier.width(2.dp))
        Text(
            text = value,//"$label: $value",
            style = MaterialTheme.typography.bodySmall
        )
    }
}


fun loadMapWithMarkers(webView: WebView, restaurants: List<Restaurant>) {
    val startTime = System.currentTimeMillis()
    Log.d("Search","loading Map With Markers.")
    val simplifiedRestaurants = restaurants.map {
        mapOf(
            "name" to it.name,
            "address" to it.address,
            "latitude" to (it.latitude ?: 0.0),
            "longitude" to (it.longitude ?: 0.0)
        )
    }
    // Convert lists to JSON
    val restaurantsJson = Gson().toJson(simplifiedRestaurants)

    // Inject JavaScript functions based on API type
    if (simplifiedRestaurants.isNotEmpty()) {
        webView.evaluateJavascript("""
            addMarkersWithCoordinates($restaurantsJson);
            fitBoundsToMarkers();
        """.trimIndent(), null)
    }
    // End of function timing
    val endTime = System.currentTimeMillis()
    Log.d("Performance","loadMapWithMarkers() took: ${endTime - startTime} ms")
    Log.d("Search","loading Map With Markers ended.")
}

@Composable
fun LocalWebView(webView: WebView) {
    AndroidView(factory = { context ->
        webView.apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            // Enable JavaScript if needed
            settings.javaScriptEnabled = true

            // Set a WebViewClient to handle loading
            webViewClient = WebViewClient()
        }
    })
}

fun analyzeUserContext(userPreferences: UserPreferences, category: String): String? {
    // Define mappings similar to the Python function
    val categoryKeys = mapOf(
        "cuisine" to listOf("user_styles", "clicked_restaurant_styles"),
        "pricing category" to listOf("user_prices", "clicked_restaurant_prices"),
        "meals" to listOf("user_meals", "clicked_restaurant_meals"),
        "features" to listOf("user_features", "clicked_restaurant_features"),
//        "ratings" to listOf("user_ratings", "restaurant_ratings")
    )

    // Get relevant keys for the missing category
    val keys = categoryKeys[category]
    if (keys == null) {
        Log.e("analyzeUserContext", "Category $category is not mapped!")
        return null
    }

    Log.d("analyzeUserContext", "Analyzing user context for category: $category")
    Log.d("analyzeUserContext", "Keys to check: $keys")

    val combinedList = mutableListOf<String>()

    // Retrieve past preferences from shared preferences
    keys.forEach { key ->
        val values = userPreferences.getUserPreferenceList(key)// This should return a List<String>
        Log.d("analyzeUserContext", "Retrieved values for key [$key]: $values")
        combinedList.addAll(values)
    }

    if (combinedList.isEmpty()) {
        Log.w("analyzeUserContext", "No preferences found for category: $category")
        return null
    }

    // Find the most common preference
    val mostCommonPreference = combinedList.groupingBy { it }
        .eachCount()
        .maxByOrNull { it.value }
        ?.key

    Log.d("analyzeUserContext", "Most common preference for $category: $mostCommonPreference")

    return mostCommonPreference
}


fun loadCitiesFromCSV(context: Context): List<String?> {
    val cities : MutableList<String?> = mutableListOf(null)
//    cities.add(null)
    try {
        val inputStream = context.resources.openRawResource(R.raw.cities)
        val reader = BufferedReader(InputStreamReader(inputStream))
        reader.forEachLine { line ->
            cities.add(line.trim()) // Trim spaces and add each city
        }
        reader.close()
    } catch (e: Exception) {
        e.printStackTrace()
    }
    return cities
}




