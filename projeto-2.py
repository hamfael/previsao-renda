import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


sns.set(context='talk', style='ticks')

st.set_page_config(
     page_title="Previsão de Renda",
     page_icon=":?:",
     layout="wide",
)

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')

#plots
fig, ax = plt.subplots(8,1,figsize=(10,100))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,100))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
ax[3].tick_params(rotation=20)
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
ax[4].tick_params(rotation=20)
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
ax[6].tick_params(rotation=20)
sns.despine()
st.pyplot(plt)

st.title("Previsão da Renda")

# Carregar o modelo
model = joblib.load('./output/best_model.pkl')
columns = joblib.load('./output/columns.pkl')

def fazer_previsao(entrada):
    df = pd.DataFrame(tuple(entrada))
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    previsao = model.predict(df)
    return previsao[0]

# Entrada do usuário para as variáveis
qtd_filhos = st.number_input('Quantidade de Filhos', min_value=0, max_value=14, value=0)
idade = st.number_input('Idade', min_value=18, max_value=70, value=30)
tempo_emprego = st.number_input('Tempo de Emprego (anos)', min_value=0, max_value=50, value=5)
qt_pessoas_residencia = st.number_input('Quantidade de Pessoas na Residência', min_value=1, max_value=15, value=1)
sexo = st.selectbox('Sexo', ['M', 'F'])
posse_de_veiculo = st.selectbox('Posse de Veículo', ['Sim', 'Não'])
posse_de_imovel = st.selectbox('Posse de Imóvel', ['Sim', 'Não'])
tipo_renda = st.selectbox('Tipo de Renda', ['Assalariado', 'Empresário', 'Pensionista', 'Servidor público', 'Bolsista'])
educacao = st.selectbox('Nível de Escolaridade', ['Primário', 'Pós graduação', 'Superior completo', 'Superior incompleto', 'Secundário'])
estado_civil = st.selectbox('Estado Civil', ['Solteiro', 'Casado', 'Separado', 'Viúvo', 'União'])
tipo_residencia = st.selectbox('Tipo de Residência', ['Casa', 'Com os pais', 'Aluguel', 'Estúdio', 'Comunitário', 'Governamental'])

# Botao para fazer previsao
if st.button('Fazer previsão'):
    entrada = {
        'sexo': [sexo],
        'posse_de_veiculo': [posse_de_veiculo],
        'posse_de_imovel': [posse_de_imovel],
        'qtd_filhos': [qtd_filhos],
        'tipo_renda': [tipo_renda],
        'educacao': [educacao],
        'estado_civil': [estado_civil],
        'tipo_residencia': [tipo_residencia],
        'idade': [idade],
        'tempo_emprego': [tempo_emprego],
        'qt_pessoas_residencia': [qt_pessoas_residencia],   
    }
    resultado = fazer_previsao(entrada)
    st.subheader(f"A previsão da renda é: R$ {resultado:.2f}")










