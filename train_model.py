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


def load_and_preprocess_data(filepath):

    df = pd.read_csv("C:/xampp/htdocs/spgp/solarpowergeneration.csv")

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


def perform_eda(df):

    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    plt.figure(figsize=(10, 6))
    sns.histplot(df['generated_power_kw'], kde=True)
    plt.title('Distribution of Generated Power (kW)')
    plt.xlabel('Power Generated (kW)')
    plt.ylabel('Frequency')
    plt.show()
    
    # Power generation by time
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='time_proxy', y='generated_power_kw', data=df)
    plt.title('Solar Power Generation by Hour of Day')
    plt.xlabel('Hour of Day (Approximate)')
    plt.ylabel('Power Generated (kW)')
    plt.xticks(range(0, 24))
    plt.grid()
    plt.show()
    
    # Power vs radiation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='shortwave_radiation_backwards_sfc', y='generated_power_kw', data=df)
    plt.title('Power Generation vs. Solar Radiation')
    plt.xlabel('Solar Radiation (W/m²)')
    plt.ylabel('Power Generated (kW)')
    plt.show()
    
    # Seasonal analysis (using categorical column)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='season', y='generated_power_kw', data=df)
    plt.title('Power Generation by Season')
    plt.xlabel('Season')
    plt.ylabel('Power Generated (kW)')
    plt.show()



def train_model(df):
    """Train and evaluate the prediction model"""
    # Select only numerical features (excluding 'season')
    features = ['temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd',
               'total_cloud_cover_sfc', 'shortwave_radiation_backwards_sfc',
               'wind_speed_10_m_above_gnd', 'angle_of_incidence', 'zenith',
               'azimuth', 'is_daylight', 'temp_humidity_interaction',
               'radiation_cloud_interaction']
    
    X = df[features]
    y = df['generated_power_kw']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Feature importance
    importances = pipeline.named_steps['model'].feature_importances_
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()
    
    return pipeline


def analyze_peak_hours(df):
    """Identify and visualize peak generation hours"""
    hourly_generation = df.groupby('time_proxy')['generated_power_kw'].mean().reset_index()
    
    peak_hours = hourly_generation[hourly_generation['generated_power_kw'] > 
                 hourly_generation['generated_power_kw'].quantile(0.75)]
    
    print("\nPeak Generation Hours:")
    print(peak_hours.sort_values('generated_power_kw', ascending=False))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='time_proxy', y='generated_power_kw', data=hourly_generation)
    plt.title('Average Power Generation by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Power Generated (kW)')
    plt.axhline(y=hourly_generation['generated_power_kw'].quantile(0.75), 
                color='r', linestyle='--', label='Peak Threshold')
    plt.legend()
    plt.show()
    
    return peak_hours


def generate_optimization_tips(df):
    """Generate solar power optimization tips based on data analysis"""
    tips = []
    
    # Peak hours
    peak_hours = df.groupby('time_proxy')['generated_power_kw'].mean().nlargest(3).index.tolist()
    tips.append(f"Peak generation hours: {', '.join(map(str, sorted(peak_hours)))}")
    
    # Cloud impact
    cloud_impact = df.groupby(pd.cut(df['total_cloud_cover_sfc'], bins=[0, 30, 70, 100]))['generated_power_kw'].mean()
    tips.append(f"Average generation by cloud cover:\n"
               f"  - Clear (0-30%): {cloud_impact.iloc[0]:.1f} kW\n"
               f"  - Partly cloudy (30-70%): {cloud_impact.iloc[1]:.1f} kW\n"
               f"  - Overcast (70-100%): {cloud_impact.iloc[2]:.1f} kW")
    
    # Temperature impact
    temp_impact = df.groupby(pd.cut(df['temperature_2_m_above_gnd'], bins=[-10, 0, 10, 20, 30]))['generated_power_kw'].mean()
    tips.append("Temperature impact:\n"
               "  - Colder temperatures (<0°C) reduce panel efficiency\n"
               "  - Optimal performance typically between 10-20°C")
    
    # Seasonal patterns
    seasonal_impact = df.groupby('season')['generated_power_kw'].mean()
    tips.append(f"Seasonal averages:\n"
               f"  - Summer: {seasonal_impact.get('summer', 0):.1f} kW\n"
               f"  - Spring/Fall: {seasonal_impact.get('spring/fall', 0):.1f} kW\n"
               f"  - Winter: {seasonal_impact.get('winter', 0):.1f} kW")
    
    # Maintenance suggestion
    tips.append("Maintenance tips:\n"
               "  - Clean panels regularly (dirt can reduce efficiency by 5-25%)\n"
               "  - Ensure panels are properly angled for your latitude\n"
               "  - Trim nearby vegetation that may cast shadows")
    
    print("\n=== SOLAR POWER OPTIMIZATION TIPS ===")
    for tip in tips:
        print(f"\n{tip}")
        print("-" * 50)



def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('solarpowergeneration.csv')
    
    # Perform EDA
    print("\nPerforming exploratory data analysis...")
    perform_eda(df)
    
    # Train model
    print("\nTraining prediction model...")
    model = train_model(df)
    
    # Analyze peak hours
    print("\nAnalyzing peak generation hours...")
    peak_hours = analyze_peak_hours(df)
    
    # Generate optimization tips
    generate_optimization_tips(df)
    
    # Save model
    joblib.dump(model, 'solar_power_predictor.pkl')
    print("\nModel saved as 'solar_power_predictor.pkl'")

if __name__ == "__main__":
    main()
