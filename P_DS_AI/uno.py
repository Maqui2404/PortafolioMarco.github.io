import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

@st.cache_data
def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df['MedHouseVal'] = df['MedHouseVal'] * 100000
    return df

@st.cache_resource
def train_model(data):
    features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    target = 'MedHouseVal'

    X = data[features]
    y = data[target]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
    }
    return model, metrics

data = load_data()
st.title("Predicción de Precios de Viviendas en California")
st.write("Vista Previa de los Datos")
st.write(data.head())

model, metrics = train_model(data)

st.write("Métricas del Modelo")
st.write(f"MAE: {metrics['MAE']:.2f}")
st.write(f"RMSE: {metrics['RMSE']:.2f}")
st.write(f"R²: {metrics['R²']:.2f}")


st.sidebar.header("Características de la Vivienda")
med_inc = st.sidebar.slider("Ingreso Mediano (en decenas de miles)", float(data['MedInc'].min()), float(data['MedInc'].max()), 3.0)
house_age = st.sidebar.slider("Edad de la Casa", int(data['HouseAge'].min()), int(data['HouseAge'].max()), 20)
ave_rooms = st.sidebar.slider("Promedio de Habitaciones", float(data['AveRooms'].min()), float(data['AveRooms'].max()), 5.0)
ave_bedrms = st.sidebar.slider("Promedio de Dormitorios", float(data['AveBedrms'].min()), float(data['AveBedrms'].max()), 1.0)
population = st.sidebar.slider("Población", int(data['Population'].min()), int(data['Population'].max()), 500)
ave_occup = st.sidebar.slider("Promedio de Ocupantes", float(data['AveOccup'].min()), float(data['AveOccup'].max()), 3.0)
latitude = st.sidebar.slider("Latitud", float(data['Latitude'].min()), float(data['Latitude'].max()), 34.0)
longitude = st.sidebar.slider("Longitud", float(data['Longitude'].min()), float(data['Longitude'].max()), -118.0)

input_features = pd.DataFrame({
    'MedInc': [med_inc],
    'HouseAge': [house_age],
    'AveRooms': [ave_rooms],
    'AveBedrms': [ave_bedrms],
    'Population': [population],
    'AveOccup': [ave_occup],
    'Latitude': [latitude],
    'Longitude': [longitude],
})

predicted_price = model.predict(input_features)[0]
st.subheader("Precio Predicho de la Vivienda")
st.write(f"**${predicted_price:,.2f}**")
