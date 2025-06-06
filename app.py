import streamlit as st
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load('modelo_credito.pkl')

st.title("Predictor de Riesgo Crediticio")

# Inputs básicos (ajustá según tus features)
duration = st.number_input("Duración del crédito (meses)", min_value=1)
amount = st.number_input("Monto del crédito", min_value=1)
installment_rate = st.selectbox("Tasa de cuota", [1,2,3,4])
age = st.number_input("Edad", min_value=18)

if st.button("Predecir"):
    # Crear DataFrame con los inputs
    data = pd.DataFrame({
        'duration': [duration],
        'amount': [amount], 
        'installment_rate': [installment_rate],
        'age': [age]
        # Agregá todas tus features aquí
    })
    
    prediccion = modelo.predict_proba(data)[0][1]
    st.write(f"Probabilidad de mal crédito: {prediccion:.2%}")