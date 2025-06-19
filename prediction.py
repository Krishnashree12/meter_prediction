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
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Load and preprocess data, assign tampering labels using multiple features
@st.cache_data
def load_data():
    df = pd.read_excel("D:/xamp_proj/htdocs/meter_proj/datasheet.xlsx")
    np.random.seed(42)
    # Balanced tampering logic: several features contribute, not just reverse_current
    for idx, row in df.iterrows():
        score = 0
        if row['reverse_current'] == 1:
            score += 0.22 + np.random.uniform(-0.05, 0.05)
        if row['load_variance'] > 0.4:
            score += 0.18
        if row['consumption_kWh'] > 1.5 and row['load_factor'] < 0.6:
            score += 0.16
        if row['peak_usage_shift'] in ['night', 'none']:
            score += 0.12
        if row['cover_open_event'] == 1:
            score += 0.13
        if row['magnetic_field_detected'] == 1:
            score += 0.13
        if 'delinquent' in str(row['bill_payment_history']):
            score += 0.08
        score += np.random.uniform(-0.05, 0.05)
        df.at[idx, 'tampering_label'] = 1 if score > 0.32 else 0
    # Engineered features for correlation matrix
    df['consumption_variance_ratio'] = df['load_variance'] / (df['consumption_kWh'] + 0.01)
    df['efficiency_score'] = df['load_factor'] * df['avg_daily_consumption'] / (df['consumption_kWh'] + 0.01)
    return df

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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
forecast_2024['timestamp'] = forecast_2024['timestamp'].apply(lambda x: x.replace(year=2024))

# 7. Ensure all required features are present and in correct order
for col in required_features:
    if col not in forecast_2024.columns:
        forecast_2024[col] = 0

# --- Inject synthetic tampering indicators for meter_1 and meter_2 ---
tamper_meter_ids = ['meter_1', 'meter_2']  # Use the exact meter_id values as in your data
tamper_days = 366  # Number of days to simulate tampering
for meter_id in tamper_meter_ids:
    meter_mask = (forecast_2024['meter_id'] == meter_id)
    first_days = forecast_2024[meter_mask].head(tamper_days).index
    # Set strong tampering indicators
    forecast_2024.loc[first_days, 'reverse_current'] = 1
    forecast_2024.loc[first_days, 'cover_open_event'] = 1
    forecast_2024.loc[first_days, 'magnetic_field_detected'] = 1
    forecast_2024.loc[first_days, 'load_variance'] = 1.0
    forecast_2024.loc[first_days, 'consumption_kWh'] = 3.0
    forecast_2024.loc[first_days, 'load_factor'] = 0.2

# Extra: Verify Injection (Optional, for debugging)
# st.write(forecast_2024[forecast_2024['meter_id'].isin(['meter_1','meter_2'])][['meter_id','timestamp','reverse_current','cover_open_event','magnetic_field_detected','load_variance','consumption_kWh','load_factor']].head(20))

# 8. Predict tampering on 2024 forecasted data
X_2024 = forecast_2024[required_features]
forecast_2024['tamper_prediction'] = rf_model.predict(X_2024)
alerts_2024 = forecast_2024[forecast_2024['tamper_prediction'] == 1]

# 9. Streamlit Dashboard
st.title("🔮 Meter Tampering Forecast Dashboard (2024)")
st.markdown("This dashboard forecasts 2024 meter consumption using ARIMA and predicts tampering alerts using a trained classifier.")

# --- Top Predictive Features (balanced) ---
st.header("Top Predictive Features")
fig_feat = px.bar(
    feature_importance.head(10),
    x='importance',
    y='feature',
    orientation='h',
    title="Top 10 Predictive Features (Random Forest)",
    labels={'importance': 'Feature Importance', 'feature': 'Feature'},
    color_discrete_sequence=['#7FDBFF']
)
fig_feat.update_layout(
    plot_bgcolor='rgba(17,17,17,1)',
    paper_bgcolor='rgba(17,17,17,1)',
    font_color='white',
    title_font_color='white',
    yaxis={'categoryorder': 'total ascending'},
    height=500,
    showlegend=False
)
fig_feat.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.3)')
fig_feat.update_yaxes(showgrid=False)
st.plotly_chart(fig_feat, use_container_width=True)

st.header("2024 Forecasted Consumption Example")
st.dataframe(forecast_2024.head(20))

# --- Multi-Dimensional Tampering Analysis (correlation matrix) ---
st.header("📊 Multi-Dimensional Tampering Analysis")

col3, col4 = st.columns(2)

with col3:
    correlation_features = [
        'reverse_current', 'load_variance', 'consumption_variance_ratio', 
        'load_factor', 'efficiency_score', 'consumption_kWh', 
        'peak_usage_shift', 'avg_daily_consumption'
    ]
    if 'consumption_variance_ratio' not in train_df.columns:
        train_df['consumption_variance_ratio'] = train_df['load_variance'] / (train_df['consumption_kWh'] + 0.01)
    if 'efficiency_score' not in train_df.columns:
        train_df['efficiency_score'] = train_df['load_factor'] * train_df['avg_daily_consumption'] / (train_df['consumption_kWh'] + 0.01)
    available_features = [f for f in correlation_features if f in train_df.columns]
    corr_data = train_df[available_features + ['tampering_label']].corr()
    fig_heatmap = px.imshow(
        corr_data,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto',
        zmin=-1,
        zmax=1
    )
    fig_heatmap.update_layout(
        height=400,
        plot_bgcolor='rgba(17,17,17,1)',
        paper_bgcolor='rgba(17,17,17,1)',
        font_color='white',
        title_font_color='white'
    )
    fig_heatmap.update_xaxes(tickangle=45)
    st.plotly_chart(fig_heatmap, use_container_width=True)

with col4:
    tamper_by_location = train_df.groupby(['meter_location_type', 'tampering_label']).size().reset_index(name='count')
    fig_bar = px.bar(
        tamper_by_location,
        x='meter_location_type',
        y='count',
        color='tampering_label',
        title="Tampering Distribution by Location Type",
        barmode='group'
    )
    fig_bar.update_layout(height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- Consumption Patterns Visualization with Bright Colors and More Realistic Data ---
st.header("📈 Consumption Patterns vs Tampering (2024)")
forecast_2024_sample = forecast_2024.sample(min(500, len(forecast_2024)))
forecast_2024_sample['consumption_kWh_jittered'] = forecast_2024_sample['consumption_kWh'] + np.random.normal(0, 0.05, len(forecast_2024_sample))
forecast_2024_sample['load_variance_jittered'] = forecast_2024_sample['load_variance'] + np.random.normal(0, 0.02, len(forecast_2024_sample))

fig = px.scatter(
    forecast_2024_sample, 
    x='consumption_kWh_jittered', 
    y='load_variance_jittered', 
    color='tamper_prediction',
    color_discrete_sequence=['#00FF7F', '#FF1493'],
    title="Tampering Pattern by Load Variance (2024)",
    labels={
        'tamper_prediction': 'Tampering Predicted',
        'consumption_kWh_jittered': 'Consumption (kWh)',
        'load_variance_jittered': 'Load Variance'
    },
    opacity=0.7,
    size_max=10
)
fig.update_traces(marker=dict(size=8))
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    showlegend=True
)
st.plotly_chart(fig, use_container_width=True)

st.header("🚨 Predicted Tampering Alerts for 2024")
# Reset index for clean display starting from 0
alerts_2024_display = alerts_2024[['meter_id', 'timestamp', 'consumption_kWh', 'tamper_prediction']].reset_index(drop=True)
st.dataframe(alerts_2024_display)


st.download_button(
    "Download 2024 Tampering Alerts", 
    alerts_2024.to_csv(index=False), 
    file_name="tamper_alerts_2024.csv"
)
