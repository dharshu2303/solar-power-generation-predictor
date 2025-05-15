## solar-power-generation-predictor
A web application that predicts solar power generation based on real-time weather data using machine learning.

## Features 
- **Real-time Weather Integration**: Fetches current weather conditions from OpenWeatherMap API
- **Machine Learning Model**: Uses a pre-trained Random Forest Regressor to predict solar power output
- **Dynamic Tips**: Generates personalized optimization recommendations based on weather conditions
- **Interactive Visualization**: Displays prediction results with charts and detailed metrics
- **Responsive Design**: Works on both desktop and mobile devices

## Technologies Used 
Frontend:
  HTML
  CSS
  JavaScript
  Bootstrap 5 (UI framework)
  Chart.js (Data visualization)
  
Backend:
  python
  flask 

Dataset:
  solar_power_generation.csv(Kaggle datasets)
  
 ## Machine Learning
  - Random Forest Regressor
  - Feature engineering:
  - Time proxies from solar position
  - Weather interactions (temperature × humidity, radiation × cloud cover)
  - Seasonal indicators

### Installation & Setup 

## Prerequisites
- Python 3.8+
- pip package manager

## Steps
1. Clone the repository
2. Install dependencies
3. Set up your OpenWeatherMap API key from OpenWeatherMap
4. Replace the API key in app.py
5. Run the application

  



