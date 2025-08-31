# Implementação de Feature Store em Pipeline de Deploy de Machine Learning


# Estrutura do Projeto:

Cap10/
├── dsa_dados/
│   ├── dados_brutos.csv
│   ├── feature_store.csv
│   ├── teste_features.csv
├── dsa_feature_store/
│   ├── __init__.py
│   ├── feature_engineering.py
│   ├── feature_store.py
├── dsa_ml_pipeline/
│   ├── __init__.py
│   ├── model_training.py
│   ├── model_inference.py
│   ├── modelo_dsa.pkl
├── dsa_testes/
│   ├── test_feature_store.py
├── projeto3-main.py
├── projeto3-app.py
├── projeto3-cliente-streamlit.py
├── requirements.txt


# Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute:

pip install -r requirements.txt

python -m unittest discover -s dsa_testes

python projeto3-main.py

python projeto3-app.py


# Abra outro terminal ou prompt de comando, navegue até a pasta com os arquivos e execute:

streamline run projeto3-cliente-streamlit.py

