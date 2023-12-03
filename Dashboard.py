import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import requests
import joblib
import io
from pycaret.classification import predict_model

@st.cache_resource
def load_data():
    df = pd.read_csv('https://github.com/Caiodrp/Classificar-Atividade-Humana/raw/main/df_train.csv')
    return df

@st.cache_resource
def carregar_modelo(url):
    response = requests.get(url)
    modelo = joblib.load(io.BytesIO(response.content))
    return modelo

def pca_analysis(df):
    # Removendo a variável de resposta e a identificação do voluntário
    df_features = df.drop(['cod_label', 'subject_id', 'Unnamed: 0'], axis=1)

    # Criando o padronizador
    scaler = StandardScaler()

    # Ajustando e transformando os dados
    df_pad = scaler.fit_transform(df_features)

    # Convertendo de volta para um DataFrame
    df_pad = pd.DataFrame(df_pad, columns=df_features.columns)

    # Aplicando PCA aos dados padronizados e originais
    pca_pad = PCA().fit(df_pad)
    pca = PCA().fit(df_features)

    # Calculando a variância acumulada
    var_cumperc_pad = pca_pad.explained_variance_ratio_.cumsum()
    var_cumperc = pca.explained_variance_ratio_.cumsum()

    # Número de componentes principais
    num_componentes_cumperc = len(var_cumperc_pad)

    # Números de componentes para o eixo x
    componentes_cumperc = range(1, num_componentes_cumperc + 1)

    # Criando um DataFrame para o gráfico
    df_graph = pd.DataFrame({
        'Componentes Principais': componentes_cumperc,
        'PCA Padronizado': var_cumperc_pad,
        'PCA Não Padronizado': var_cumperc
    })

    # Plotando o gráfico com Plotly Express
    fig = px.line(df_graph, x='Componentes Principais', y=df_graph.columns,
                  title='Comparação do PCA Padronizado e Não Padronizado')
    st.plotly_chart(fig)

    # Retornando o objeto PCA ajustado e o padronizador
    return pca_pad, scaler

def main():
    st.set_page_config(page_title='Classificação Atividade Humana', page_icon='🏃', layout='wide')

    st.markdown("<h1 style='text-align: center;'>Classificação Atividade Humana</h1>", unsafe_allow_html=True)

    df_treino = load_data()
    url_modelo = 'https://github.com/Caiodrp/Classificar-Atividade-Humana/raw/main/lr.pkl'
    model = carregar_modelo(url_modelo)

    # Dividindo o layout em colunas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Análise PCA")
        # Ajustando o PCA, padronização e gráfico 
        pca_pad, scaler = pca_analysis(df_treino)

    with col2:
        st.subheader("Classificação Novos Dados")
        uploaded_file = st.file_uploader("Escolha um arquivo .csv", type="csv")
        if uploaded_file is not None:
            # Carregando os dados de teste
            df_test = pd.read_csv(uploaded_file)

            # Removendo a identificação do voluntário
            df_test_features = df_test.drop(['subject_id','Unnamed: 0'], axis=1)

            # Usando o mesmo padronizador para transformar os dados de teste
            df_test_pad = scaler.transform(df_test_features)

            # Convertendo de volta para um DataFrame
            df_test_pad = pd.DataFrame(df_test_pad, columns=df_test_features.columns)

            # Aplicando PCA aos dados padronizados de teste
            # Usando o mesmo objeto PCA que foi ajustado nos dados de treino
            componentes_test_pad = pca_pad.transform(df_test_pad)

            # Renomeando as componentes
            nomes_pca_pad = ['CP'+str(x+1) for x in list(range(df_test_pad.shape[1]))]

            # DF das CPs com padronização para os dados de teste
            CP_test_pad = pd.DataFrame(data = componentes_test_pad, columns = nomes_pca_pad)

            # Fazendo previsões no conjunto de dados de teste
            predictions = predict_model(model, data=CP_test_pad)

            # As previsões estão na coluna 'Label' do dataframe de saída
            predicted_labels = predictions['prediction_label']

            # Mapeando os rótulos previstos para os nomes das classes
            dict_classes = {1: 'Caminhando', 2: 'Subindo Escadas', 3: 'Descendo Escadas', 4: 'Sentado', 5: 'Em Pé', 6: 'Deitado'}
            predicted_labels = predicted_labels.map(dict_classes)

            # Exibindo as previsões
            st.write(predicted_labels)

if __name__ == "__main__":
    main()
