import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Configuración de la página
st.set_page_config(
    page_title="Evaluación de Riesgo Crediticio",
    page_icon="💳",
    layout="wide"
)

# Cargar el modelo
@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_credito.pkl')
        return model
    except FileNotFoundError:
        st.error("Modelo no encontrado. Asegúrate de que 'modelo_credito.pkl' esté en el directorio.")
        return None

# Función para crear las features dummy
def create_dummy_features(input_data):
    # Todas las posibles categorías para cada variable categórica
    account_status_options = ['balance_0_to_200', 'balance_over_200', 'negative_balance', 'no_account']
    credit_history_options = ['all_paid_at_bank', 'critical_account', 'existing_paid_duly', 'no_credits_or_all_paid', 'past_payment_delays']
    purpose_options = ['business', 'domestic_appliances', 'education', 'furniture_equipment', 'new_car', 'other_purpose', 'radio_tv', 'repairs', 'retraining', 'used_car', 'vacation']
    savings_account_options = ['no_savings', 'savings_100_to_500', 'savings_500_to_1000', 'savings_below_100', 'savings_over_1000']
    employment_duration_options = ['employed_1_to_4yrs', 'employed_4_to_7yrs', 'employed_less_1yr', 'employed_over_7yrs', 'unemployed']
    personal_status_sex_options = ['female_not_single', 'female_single', 'male_divorced', 'male_married_widowed', 'male_single']
    property_type_options = ['car_other', 'no_property', 'real_estate', 'savings_insurance']
    other_installment_plans_options = ['bank', 'none', 'stores']
    housing_type_options = ['free', 'own', 'rent']
    job_type_options = ['management_selfemployed', 'skilled_employee', 'unemployed_unskilled_nonresident', 'unskilled_resident']
    telephone_options = ['none', 'yes']
    
    # Crear DataFrame con las columnas dummy
    dummy_data = {}
    
    # Variables numéricas
    dummy_data['duration_in_months'] = input_data['duration_in_months']
    dummy_data['credit_amount'] = input_data['credit_amount']
    dummy_data['installment_rate_pct'] = input_data['installment_rate_pct']
    dummy_data['residence_since'] = input_data['residence_since']
    dummy_data['age_years'] = input_data['age_years']
    dummy_data['existing_credits_count'] = input_data['existing_credits_count']
    dummy_data['dependents_count'] = input_data['dependents_count']
    
    # Variables categóricas - crear dummies
    for option in account_status_options:
        dummy_data[f'account_status_{option}'] = 1 if input_data['account_status'] == option else 0
    
    for option in credit_history_options:
        dummy_data[f'credit_history_{option}'] = 1 if input_data['credit_history'] == option else 0
    
    for option in purpose_options:
        dummy_data[f'purpose_{option}'] = 1 if input_data['purpose'] == option else 0
    
    for option in savings_account_options:
        dummy_data[f'savings_account_{option}'] = 1 if input_data['savings_account'] == option else 0
    
    for option in employment_duration_options:
        dummy_data[f'employment_duration_{option}'] = 1 if input_data['employment_duration'] == option else 0
    
    for option in personal_status_sex_options:
        dummy_data[f'personal_status_sex_{option}'] = 1 if input_data['personal_status_sex'] == option else 0
    
    for option in property_type_options:
        dummy_data[f'property_type_{option}'] = 1 if input_data['property_type'] == option else 0
    
    for option in other_installment_plans_options:
        dummy_data[f'other_installment_plans_{option}'] = 1 if input_data['other_installment_plans'] == option else 0
    
    for option in housing_type_options:
        dummy_data[f'housing_type_{option}'] = 1 if input_data['housing_type'] == option else 0
    
    for option in job_type_options:
        dummy_data[f'job_type_{option}'] = 1 if input_data['job_type'] == option else 0
    
    for option in telephone_options:
        dummy_data[f'telephone_{option}'] = 1 if input_data['telephone'] == option else 0
    
    return pd.DataFrame([dummy_data])

# Título de la aplicación
st.title("💳 Evaluación de Riesgo Crediticio")
st.markdown("---")

# Cargar modelo
model = load_model()

if model is not None:
    st.markdown("### 📊 Ingrese los datos del cliente:")
    
    # Crear columnas para organizar los inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Información Básica")
        age_years = st.number_input("Edad (años)", min_value=18, max_value=80, value=35)
        credit_amount = st.number_input("Monto del crédito", min_value=250, max_value=20000, value=3000)
        duration_in_months = st.number_input("Duración en meses", min_value=4, max_value=72, value=20)
        
        personal_status_sex = st.selectbox("Estado civil y sexo", [
            'male_single', 'female_single', 'male_married_widowed', 
            'female_not_single', 'male_divorced'
        ])
        
        dependents_count = st.number_input("Número de dependientes", min_value=1, max_value=2, value=1)
    
    with col2:
        st.subheader("Información Financiera")
        account_status = st.selectbox("Estado de cuenta", [
            'balance_0_to_200', 'balance_over_200', 'negative_balance', 'no_account'
        ])
        
        savings_account = st.selectbox("Cuenta de ahorros", [
            'savings_below_100', 'savings_100_to_500', 'savings_500_to_1000',
            'savings_over_1000', 'no_savings'
        ])
        
        installment_rate_pct = st.number_input("Porcentaje de cuota (%)", min_value=1, max_value=4, value=2)
        existing_credits_count = st.number_input("Número de créditos existentes", min_value=1, max_value=4, value=1)
        
        credit_history = st.selectbox("Historial crediticio", [
            'no_credits_or_all_paid', 'all_paid_at_bank', 'existing_paid_duly',
            'past_payment_delays', 'critical_account'
        ])
    
    with col3:
        st.subheader("Información Personal")
        employment_duration = st.selectbox("Duración del empleo", [
            'unemployed', 'employed_less_1yr', 'employed_1_to_4yrs',
            'employed_4_to_7yrs', 'employed_over_7yrs'
        ])
        
        job_type = st.selectbox("Tipo de trabajo", [
            'unemployed_unskilled_nonresident', 'unskilled_resident',
            'skilled_employee', 'management_selfemployed'
        ])
        
        purpose = st.selectbox("Propósito del crédito", [
            'new_car', 'used_car', 'furniture_equipment', 'radio_tv',
            'domestic_appliances', 'repairs', 'education', 'vacation',
            'retraining', 'business', 'other_purpose'
        ])
        
        property_type = st.selectbox("Tipo de propiedad", [
            'real_estate', 'savings_insurance', 'car_other', 'no_property'
        ])
        
        housing_type = st.selectbox("Tipo de vivienda", [
            'rent', 'own', 'free'
        ])
        
        residence_since = st.number_input("Años en residencia actual", min_value=1, max_value=4, value=2)
        
        other_installment_plans = st.selectbox("Otros planes de pago", [
            'bank', 'stores', 'none'
        ])
        
        telephone = st.selectbox("Teléfono", ['none', 'yes'])
    
    st.markdown("---")
    
    # Botón para hacer la predicción
    if st.button("🔍 Evaluar Riesgo Crediticio", type="primary"):
        # Crear diccionario con los datos de entrada
        input_data = {
            'duration_in_months': duration_in_months,
            'credit_amount': credit_amount,
            'installment_rate_pct': installment_rate_pct,
            'residence_since': residence_since,
            'age_years': age_years,
            'existing_credits_count': existing_credits_count,
            'dependents_count': dependents_count,
            'account_status': account_status,
            'credit_history': credit_history,
            'purpose': purpose,
            'savings_account': savings_account,
            'employment_duration': employment_duration,
            'personal_status_sex': personal_status_sex,
            'property_type': property_type,
            'other_installment_plans': other_installment_plans,
            'housing_type': housing_type,
            'job_type': job_type,
            'telephone': telephone
        }
        
        # Crear features dummy
        features_df = create_dummy_features(input_data)
        
        # Hacer la predicción
        try:
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]
            
            # Mostrar resultados
            st.markdown("### 📈 Resultados de la Evaluación")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success("✅ **BUEN PAGADOR**")
                    st.write("El cliente tiene alta probabilidad de cumplir con sus pagos.")
                else:
                    st.error("❌ **MAL PAGADOR**")
                    st.write("El cliente tiene alta probabilidad de entrar en default.")
            
            with col2:
                st.metric("Probabilidad de Buen Pagador", f"{prediction_proba[0]:.2%}")
                st.metric("Probabilidad de Mal Pagador", f"{prediction_proba[1]:.2%}")
            
            # Gráfico de probabilidades
            st.markdown("### 📊 Distribución de Probabilidades")
            prob_df = pd.DataFrame({
                'Clasificación': ['Buen Pagador', 'Mal Pagador'],
                'Probabilidad': [prediction_proba[0], prediction_proba[1]]
            })
            st.bar_chart(prob_df.set_index('Clasificación'))
            
        except Exception as e:
            st.error(f"Error al hacer la predicción: {str(e)}")
    
    # Información adicional
    st.markdown("---")
    st.markdown("### ℹ️ Información del Modelo")
    st.write("""
    Este modelo de riesgo crediticio utiliza un **Random Forest optimizado** entrenado en el dataset German Credit Data.
    
    **Características del modelo:**
    - Optimizado para minimizar falsos positivos (malos pagadores clasificados como buenos)
    - Utiliza class_weight para balancear las clases
    - ROC-AUC Score optimizado mediante GridSearchCV
    
    **Interpretación:**
    - **Buen Pagador (Clase 1)**: Cliente con baja probabilidad de default
    - **Mal Pagador (Clase 2)**: Cliente con alta probabilidad de default
    """)

else:
    st.error("No se pudo cargar el modelo. Verifica que el archivo 'modelo_credito.pkl' esté disponible.")