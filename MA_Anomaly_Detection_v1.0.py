import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import streamlit as st

# Impostare il seme per la riproducibilità
np.random.seed(42)

# Interfaccia Streamlit
st.set_page_config(page_title="Anomaly Detection", layout="wide")

# Crea una riga per il logo e il titolo
col1, col2 = st.columns([1, 4])

# Inserimento dellogo nella prima colonna
with col1:
   logo_url = "https://commons.wikimedia.org/wiki/File:Marsh_McLennan_logo.png"  # Sostituisci con l'URL reale
   st.image(logo_url, width=80)


# Inserimento del titolo nella seconda colonna
with col2:
    st.title("Marsh Advisory - Anomaly Detection")
    st.markdown("<h5 style='margin-top: -10px;'>Analisi delle anomalie nei dati</h5>", unsafe_allow_html=True)

# Caricare il file CSV
uploaded_file = st.file_uploader("Carica un file CSV", type=["csv"])

if uploaded_file is not None:
    # Leggere il file CSV
    data = pd.read_csv(uploaded_file, delimiter=';')

    st.write("Dati caricati con successo!")
    st.dataframe(data.head())

# Rimuovere righe con ' Risparmio Stimato (TEP) ' = null, vuoto o "-"
    data = data[~data[' Risparmio Stimato (TEP) '].isnull() & 
                (data[' Risparmio Stimato (TEP) '] != '') & 
                (data[' Risparmio Stimato (TEP) '] != '-')]


    # Assicurarsi che le colonne necessarie siano presenti
    required_columns = [
        'Nome Richiesta', 
        'Allegato A Totale',  
        'Allegato B Totale',
        'Partita Iva ',
        'Data Iscrizione Registro Impresa',
        'Data Costituzione',
        'Data prevista avvio realizzazione',
        'Data prevista fine realizzazione',
        'Spese Autoconsumo ',
        'Spese Formazione ', 
        'Autoconsumo Totale',
        'Certificazione Risparmio Energetico Totale',
        'Totale',
        'Risparmio Energetico ', 
        'Credito Imposta Totale ', 
        'Risparmio Stimato Percentuale', 
        ' Risparmio Stimato (TEP) '
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Il file CSV manca delle seguenti colonne: {missing_columns}")
    else:
        # Sostituire le virgole con i punti e convertire in float
        for col in required_columns:
            if col in data.columns:
                if col not in ['Partita Iva ', 'Nome Richiesta']:
                    data[col] = data[col].astype(str).str.replace(',', '.')
                    data[col] = pd.to_numeric(data[col], errors='coerce')

        # Controlla se ci sono valori NaN
        nan_counts = data.isnull().sum()
        st.write("Conteggio valori NaN per colonna:", nan_counts)

        # Sostituisci i valori NaN con un valore predefinito
        default_value = 0
        data.fillna(default_value, inplace=True)

    
        # Definizione delle caratteristiche numeriche
        numeric_features = [
            'Credito Imposta Totale ', 
            'Risparmio Stimato Percentuale', 
            ' Risparmio Stimato (TEP) ',
            'Allegato A Totale',  
            'Allegato B Totale',
            'Autoconsumo Totale'
        ]

        # Parametri per Isolation Forest
        contamination = st.slider("Contaminazione", min_value=0.0, max_value=0.5, value=0.01, step=0.01)
        n_estimators = st.number_input("Numero di Estimatori", min_value=10, max_value=20000, value=100, step=10)

        # Pulsante per addestrare il modello
        if st.button("Addestra il modello"):
            # Creare il preprocessore e il modello
            numeric_transformer = StandardScaler()
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ]
            )

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('isolation_forest', IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42))
            ])

            # Fit del modello
            model.fit(data[numeric_features])

            # Calcolare i punteggi di anomalia
            anomaly_scores = model.named_steps['isolation_forest'].decision_function(model.named_steps['preprocessor'].transform(data[numeric_features]))

            # Normalizzazione dei punteggi di anomalia
            min_score = anomaly_scores.min()
            max_score = anomaly_scores.max()
            data['anomaly_score'] = 1 - ((anomaly_scores - min_score) / (max_score - min_score))

            # Calcolare la soglia per le anomalie
            threshold = np.percentile(data['anomaly_score'], 100 * (1 - contamination))  # Modificato per utilizzare il valore di contaminazione

            # Creare una colonna per indicare se è un'anomalia
            data['anomaly'] = np.where(data['anomaly_score'] > threshold, 'SI', 'NO')

            # Mappare i colori per le anomalie
            color_map = {'NO': 'blue', 'SI': 'red'}
            data['color'] = data['anomaly'].map(color_map)

            # Visualizzare con Seaborn
            plt.figure(figsize=(10, 6))
            pairplot = sns.pairplot(data, hue='anomaly', vars=numeric_features, palette=color_map)
            plt.suptitle('Pairplot delle Anomalie', y=1.02)
            st.pyplot(plt)

            # Visualizzare la tabella finale con i punteggi di anomalia e la colonna "anomalie"
            output_columns = ['Nome Richiesta', 'anomaly_score', 'anomaly']
            output_table = data[output_columns]

            # Mostrare la tabella finale
            st.write("Tabella dei risultati:", output_table)

            # Percorso e nome del file CSV
            output_file_path = r'C:\Users\u1208854\OneDrive - MMC\General\ML\GSE\DEMO SM\OUTPUT\Risultati_IF_T5.0_AL__v5.csv'

            # Salvare la tabella finale in un file CSV
            output_table.to_csv(output_file_path, index=False, encoding='utf-8')
            st.success(f"File salvato con successo in: {output_file_path}")
