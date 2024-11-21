import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Función para cargar y preparar los datos
@st.cache
def load_data():
    trips = pd.read_csv('BD_MachineLearning.csv')
    time_serie = trips[['date', 'total_viajes']]
    time_serie['date'] = pd.to_datetime(time_serie['date'])
    time_serie = time_serie.sort_values(by='date').reset_index(drop=True)
    mean_value = time_serie['total_viajes'].mean()
    time_serie['total_viajes'] = time_serie['total_viajes'].apply(
        lambda x: mean_value if x < 20000 else x
    )
    time_serie.set_index('date', inplace=True)
    return time_serie

# Función para ajustar el modelo ARIMA y hacer pronósticos
def arima_forecast(time_serie, periods):
    best_p, best_d, best_q = 5, 1, 5
    arima_model = ARIMA(time_serie['total_viajes'], order=(best_p, best_d, best_q))
    arima_result = arima_model.fit()
    forecast = arima_result.get_forecast(steps=periods)
    forecast_index = pd.date_range(start=time_serie.index[-1] + pd.Timedelta(days=1), periods=periods, freq='D')
    forecast_values = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    forecast_df = pd.DataFrame({
        'Fecha': forecast_index,
        'Pronóstico': forecast_values,
        'Confianza Inferior': forecast_ci.iloc[:, 0],
        'Confianza Superior': forecast_ci.iloc[:, 1]
    })
    return forecast_df

# Cargar datos
time_serie = load_data()

# Título de la aplicación
st.title("Modelo ARIMA para Pronóstico de Viajes")

# Selección del número de periodos a pronosticar
periods = st.slider('Selecciona el número de periodos a pronosticar', min_value=1, max_value=10, value=7)

# Realizar el pronóstico
forecast_df = arima_forecast(time_serie, periods)

# Mostrar los valores del pronóstico
st.subheader(f"Valores del pronóstico para los próximos {periods} días")
st.write(forecast_df)

# Crear el gráfico de pronóstico
st.subheader("Gráfico de Pronóstico")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time_serie.index, time_serie['total_viajes'], label='Observado')
ax.plot(forecast_df['Fecha'], forecast_df['Pronóstico'], color='red', label='Pronóstico')
ax.fill_between(forecast_df['Fecha'], forecast_df['Confianza Inferior'], forecast_df['Confianza Superior'], color='pink', alpha=0.3)
ax.set_title('Pronóstico del Modelo ARIMA')
ax.set_xlabel('Fecha')
ax.set_ylabel('Total de Viajes')
ax.legend()
st.pyplot(fig)
