# Projeto 3 - Implementação de Feature Store em Pipeline de Deploy de Machine Learning
# Módulo de Consumo da API

# Imports
import requests
import streamlit as st
from datetime import datetime



# URL da API
url = 'http://127.0.0.1:5000/predict'


st.title("🚀 Previsão com Feature Store")

# Inputs numéricos
entity_id = st.number_input("Entity ID", min_value=1, step=1, value=1000)
feature1 = st.number_input("Feature 1", format="%.2f")
feature2 = st.number_input("Feature 2", format="%.2f")

# Inputs de tempo
start_time = st.time_input("Start Time", value=datetime.now().time())
end_time = st.time_input("End Time", value=datetime.now().time())

# Botão para enviar
if st.button("Enviar para API"):
    # Converter start_time e end_time em segundos (desde meia-noite)
    start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
    time_diff = end_seconds - start_seconds

    # Montar payload no formato esperado
    dsa_novos_dados = [
        {"entity_id": entity_id, "feature1": feature1, "feature2": feature2, "time_diff": time_diff}
    ]

    # Enviar requisição
    response = requests.post(url, json=dsa_novos_dados)

    st.subheader("📡 Resposta da API")
    if response.status_code == 200:
        result = response.json()
        
        
        
        # Mostrar previsão de forma amigável
        if "predictions" in result:
            prediction = result["predictions"][0]
            st.markdown(
                f"""
                <div style='padding:10px; border-radius:10px; 
                            background-color:#e6f0ff; 
                            color:#004080; 
                            font-size:20px; 
                            text-align:center;'>
                    🔵 Previsão: <b>{prediction}</b>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.error(f"❌ Erro {response.status_code} na requisição")