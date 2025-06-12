import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datetime import timedelta

st.set_page_config(
    page_title="Meter Tampering Forecast Dashboard",
    page_icon="🔮",
    layout="wide",  # This makes it use full width
    initial_sidebar_state="expanded"
)

# 1. Load and preprocess data
@st.cache_data
def load_data():
    return pd.read_excel("D:/xamp_proj/htdocs/meter_proj/datasheet.xlsx")


df = load_data()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['year'] = df['timestamp'].dt.year

# 2. Encode categorical features
label_cols = ['peak_usage_shift', 'meter_location_type', 'consumer_type',
              'bill_payment_history', 'day_of_week', 'season']
label_encoders = {col: LabelEncoder().fit(df[col]) for col in label_cols}
for col in label_cols:
    df[col] = label_encoders[col].transform(df[col])

# 3. Train the tampering detection model on 2023 data
train_df = df[df['year'] == 2023]
X = train_df.drop(columns=['meter_id', 'timestamp', 'tampering_label', 'year'])
y = train_df['tampering_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Get the feature names used for training
required_features = list(X_train.columns)

# 4. Classification metrics on 2023 test data
y_pred = rf_model.predict(X_test)
metrics = classification_report(y_test, y_pred, output_dict=True)

# 5. Top predictive features
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# 6. Forecast 2024 consumption for each meter using ARIMA
forecast_horizon = 366  # days in 2024 (leap year)
forecast_list = []

for meter_id in train_df['meter_id'].unique():
    meter_data = train_df[train_df['meter_id'] == meter_id].sort_values('timestamp')
    # Set daily frequency and fill missing days with interpolation
    series = meter_data.set_index('timestamp')['consumption_kWh'].asfreq('D')
    series = series.interpolate(method='linear')
    try:
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)
        forecast_dates = pd.date_range(start=series.index.max() + timedelta(days=1), periods=forecast_horizon, freq='D')
        forecast_df = pd.DataFrame({
            'meter_id': meter_id,
            'timestamp': forecast_dates,
            'consumption_kWh': forecast.values
        })
        # Copy static/categorical and other features from last known value or fill with 0
        for col in required_features:
            if col in meter_data.columns:
                forecast_df[col] = meter_data.iloc[-1][col]
            else:
                forecast_df[col] = 0
        forecast_list.append(forecast_df)
    except Exception as e:
        st.warning(f"ARIMA failed for meter {meter_id}: {e}")

if len(forecast_list) == 0:
    st.error("No forecasts could be generated. Please check your data.")
    st.stop()

forecast_2024 = pd.concat(forecast_list, ignore_index=True)
forecast_2024['year'] = 2024

# --- Fix the year in the timestamp column to 2024 ---
forecast_2024['timestamp'] = forecast_2024['timestamp'].apply(lambda x: x.replace(year=2024))

# 7. Ensure all required features are present and in correct order
for col in required_features:
    if col not in forecast_2024.columns:
        forecast_2024[col] = 0  # or another default

X_2024 = forecast_2024[required_features]

# 8. Predict tampering on 2024 forecasted data
forecast_2024['tamper_prediction'] = rf_model.predict(X_2024)
alerts_2024 = forecast_2024[forecast_2024['tamper_prediction'] == 1]

# 9. Streamlit Dashboard
st.title("🔮 Meter Tampering Forecast Dashboard (2024)")
st.markdown("This dashboard forecasts 2024 meter consumption using ARIMA and predicts tampering alerts using a trained classifier.")

st.header("Top Predictive Features")
fig_feat = px.bar(
    feature_importance.head(10),
    x='importance',
    y='feature',
    orientation='h',
    title="Top 10 Predictive Features (Random Forest)",
    labels={'importance': 'Feature Importance', 'feature': 'Feature'}
)
st.plotly_chart(fig_feat, use_container_width=True)

st.header("2024 Forecasted Consumption Example")
st.dataframe(forecast_2024.head(20))

# --- Consumption Patterns Visualization ---
st.header("📈 Consumption Patterns vs Tampering (2024)")
fig = px.scatter(
    forecast_2024, 
    x='consumption_kWh', 
    y='load_variance', 
    color='tamper_prediction',
    title="Tampering Pattern by Load Variance (2024)",
    labels={'tamper_prediction': 'Tampering Predicted'}
)
st.plotly_chart(fig)

st.header("🚨 Predicted Tampering Alerts for 2024")
st.dataframe(alerts_2024[['meter_id', 'timestamp', 'consumption_kWh', 'tamper_prediction']])

st.download_button(
    "Download 2024 Tampering Alerts", 
    alerts_2024.to_csv(index=False), 
    file_name="tamper_alerts_2024.csv"
)
