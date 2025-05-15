import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import requests
from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop_duplicates()
    df['time_proxy'] = (df['azimuth'] / 15).round()  
    df['is_daylight'] = (df['zenith'] < 85).astype(int)
    df['temp_humidity_interaction'] = df['temperature_2_m_above_gnd'] * df['relative_humidity_2_m_above_gnd']
    df['radiation_cloud_interaction'] = df['shortwave_radiation_backwards_sfc'] * (100 - df['total_cloud_cover_sfc'])
    
    def get_season(temp):
        if temp < 5: return 'winter'
        elif 5 <= temp < 15: return 'spring/fall'
        else: return 'summer'
    df['season'] = df['temperature_2_m_above_gnd'].apply(get_season)
    return df

def get_current_weather(api_key, city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Get timezone for accurate solar position calculations
        tf = TimezoneFinder()
        timezone = tf.timezone_at(lat=data['coord']['lat'], lng=data['coord']['lon'])
        tz = pytz.timezone(timezone)
        now = datetime.now(tz)
        hour = now.hour + now.minute/60
        azimuth = (hour - 12) * 15
        zenith = max(0, 90 - (90/12) * abs(12 - hour))
        
        # Extract relevant features
        weather_data = {
            'temperature_2_m_above_gnd': data['main']['temp'],
            'relative_humidity_2_m_above_gnd': data['main']['humidity'],
            'total_cloud_cover_sfc': data['clouds']['all'],
            'wind_speed_10_m_above_gnd': data['wind']['speed'],
            'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
            'city': city,
            'weather_description': data['weather'][0]['description'],
            'latitude': data['coord']['lat'],
            'longitude': data['coord']['lon'],
            'azimuth': azimuth,
            'zenith': zenith,
            'angle_of_incidence': 30,  # Assuming fixed panel angle
            'is_day': 1 if (zenith < 85) else 0,
            'time_proxy': round(azimuth / 15)
        }
        
        # Estimate solar radiation based on time and cloud cover
        max_radiation = 1000  # W/m² at solar noon on clear day
        radiation = max_radiation * (1 - 0.7 * (weather_data['total_cloud_cover_sfc']/100)) * np.cos(np.radians(zenith))
        weather_data['shortwave_radiation_backwards_sfc'] = max(0, radiation)
        
        return weather_data
        
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def predict_power(model, weather_data):
    """Predict solar power generation based on current weather"""
    # Create the same features as in training
    data = weather_data.copy()
    data['is_daylight'] = data['is_day']
    data['temp_humidity_interaction'] = data['temperature_2_m_above_gnd'] * data['relative_humidity_2_m_above_gnd']
    data['radiation_cloud_interaction'] = data['shortwave_radiation_backwards_sfc'] * (100 - data['total_cloud_cover_sfc'])
    
    # Features must match training exactly
    features = ['temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd',
               'total_cloud_cover_sfc', 'shortwave_radiation_backwards_sfc',
               'wind_speed_10_m_above_gnd', 'angle_of_incidence', 'zenith',
               'azimuth', 'is_daylight', 'temp_humidity_interaction',
               'radiation_cloud_interaction']
    
    # Create input DataFrame
    input_data = pd.DataFrame({f: [data.get(f, 0)] for f in features})
    
    # Predict and ensure non-negative
    prediction = max(0, model.predict(input_data)[0])
    
    return prediction

def generate_dynamic_tips(weather_data, predicted_power):
    tips = []
    current_hour = datetime.strptime(weather_data['timestamp'], '%Y-%m-%d %H:%M:%S').hour
    
    # Location header
    tips.append(f"📍 Location: {weather_data['city']} ({weather_data['latitude']:.2f}°N, {weather_data['longitude']:.2f}°E)")
    
    # Current conditions
    tips.append(f"⏰ Current Time: {weather_data['timestamp']}")
    tips.append(f"🌤️ Weather: {weather_data['weather_description'].title()}")
    tips.append(f"🌡️ Temperature: {weather_data['temperature_2_m_above_gnd']:.1f}°C")
    tips.append(f"☁️ Cloud Cover: {weather_data['total_cloud_cover_sfc']}%")
    tips.append(f"💨 Wind Speed: {weather_data['wind_speed_10_m_above_gnd']} m/s")
    
    # Prediction
    tips.append(f"⚡ Predicted Power Generation: {predicted_power:.1f} kW")
    
    # Time-based tips
    if 10 <= current_hour <= 14:
        tips.append("⏳ You're in peak solar generation hours!")
    else:
        next_peak = max(10, min(14, current_hour + 1))
        tips.append(f"⌛ Next peak hours: {next_peak}:00-14:00")
    
    # Cloud impact
    if weather_data['total_cloud_cover_sfc'] > 70:
        tips.append("🌧️ Heavy clouds significantly reducing output (consider battery storage)")
    elif weather_data['total_cloud_cover_sfc'] > 30:
        tips.append("⛅ Partial clouds moderately reducing output")
    else:
        tips.append("☀️ Clear skies - optimal generation conditions")
    
    # Temperature impact
    temp = weather_data['temperature_2_m_above_gnd']
    if temp < 5:
        tips.append("❄️ Cold temperatures reducing panel efficiency (up to 20% loss)")
    elif temp > 25:
        tips.append("🔥 Hot temperatures reducing efficiency (consider ventilation)")
    else:
        tips.append("🌡️ Temperature in optimal range for solar production")
    
    # Maintenance tips based on conditions
    if weather_data['total_cloud_cover_sfc'] < 30:
        tips.append("🧹 Good time for panel cleaning (clear skies forecast)")
    if weather_data['wind_speed_10_m_above_gnd'] > 5:
        tips.append("💨 Windy conditions - check for debris accumulation")
    
    return tips

# Load the pre-trained model globally - no training required
model = joblib.load('solar_power_predictor.pkl')
API_KEY = "your_api_key"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    city = data.get('city')
    
    if not city:
        return jsonify({"error": "City name is required"}), 400
    
    # Get weather data
    weather_data = get_current_weather(API_KEY, city)
    
    if not weather_data:
        return jsonify({"error": "Could not fetch weather data for the specified city"}), 400
    
    # Make prediction
    prediction = predict_power(model, weather_data)
    
    # Generate tips
    tips = generate_dynamic_tips(weather_data, prediction)
    
    # Prepare response
    response = {
        "prediction": round(prediction, 2),
        "weather": weather_data,
        "tips": tips
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

