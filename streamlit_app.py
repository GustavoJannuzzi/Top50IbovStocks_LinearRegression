# Aplicão de regressão linear das ações que compoem o índice bovespa

# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import streamlit as st
import cufflinks as cf
import datetime


# Web scraping ações do B3 do IBOV
# https://colab.research.google.com/drive/11uYskZLbH2Y8MuSYHCPJPakY31tXMiwB?usp=sharing#scrollTo=I7cBkX2f29tf
def busca_carteira_teorica(indice):
  url = 'http://bvmf.bmfbovespa.com.br/indices/ResumoCarteiraTeorica.aspx?Indice={}&idioma=pt-br'.format(indice.upper())
  return pd.read_html(url, decimal=',', thousands='.')[0][:-1]

ibov = busca_carteira_teorica('ibov')
ibov_ordenado = ibov.sort_values('Part. (%)', ascending=False)
top50_ibov = ibov_ordenado.head(50)
top50_ibov = top50_ibov['Código'] + '.SA'
top50_ibov = top50_ibov.tolist()

# STREAMLIT
st.title("Predição das TOP 50 ações do IBOV")

ticker = st.selectbox('Escolha um ativo', top50_ibov)


######################
### LINEAR GRESSION 01


# Stock data 
df = yf.download(ticker, period = '5y', interval = '1d')
df['day_prev_close'] = df['Close'].shift(-1)

#limpa as colunas que eu não preciso
df = df.drop(columns =['High', 'Low','Adj Close', 'Volume', 'Open'])

# função para veririficar se o ultimo dia subiu ou caiu 
def f (row):
    if row ['day_prev_close'] > row['Close']: 
        val = 1
    else:
        val = -1
    return val

# Criando coluna de trend days
df['trend_3_day'] = df.apply(f, axis=1)
df = df.reset_index()
df = df.dropna()


#Separando as features do modelo
features = ['day_prev_close', 'trend_3_day']
target = 'Close'

X_train, X_test = df.loc[:600, features], df.loc[600:, features]
y_train, y_test = df.loc[:600, target], df.loc[600:, target]

# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

# Train the model using the training set
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)



# The mean squared error
st.write('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_pred))))

'Explained variance score: 1 is perfect prediction'
st.write('Variance Score: {0:.2f}'.format(r2_score(y_test, y_pred)))

print('Root Mean Squared Error: {0:.2f}'.format(np.sqrt(mean_squared_error(y_test, X_test.day_prev_close))))


### TODAYS PREDICTION ###

df_today = df.tail(1)

#previsão de fechamento da ação pra hoje
features = ['day_prev_close', 'trend_3_day']
X_test = df_today[features]
today_pred = regr.predict(X_test)

# Valor de previsão de fechamento HOJE - tratando dado
today_pred = ' '.join(str(e) for e in today_pred)
today_pred = float(today_pred)


#print valor de fechamento ontem 

# Valor de fhcamento ontem - tratando dado 
df_today = df_today['Close'].values.astype(str)
df_today = ' '.join(str(e) for e in df_today)
yestday_close = float(df_today)

percent_change_today = ((today_pred - yestday_close)/yestday_close) *100

percent_change_today = str(percent_change_today)+'%'

if today_pred < yestday_close:
    st.error('Looks like today is not a good day for')
else:
    st.success('Today tends to be a good Day for')

st.metric(label="Predição do valor de fechamento HOJE", value=today_pred, delta= percent_change_today )


st.write('Valor de fechamento de ontem', yestday_close)






###########################
### STOCK INFORMATIONN ####

st.write('---')
col1,col2 = st.columns(2)
with col1:
    # Sidebar
    st.subheader('Query parameters')
    start_date = st.date_input("Start date", datetime.date(2022, 1, 1))
    end_date = st.date_input("End date", datetime.date(2022, 1, 31))

with col2: 
    # Retrieving tickers data
   #tickerSymbol = st.selectbox('Stock ticker', top50_ibov) # Select ticker symbol
    tickerData = yf.Ticker(ticker) # Get ticker data
    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date) #get the historical prices for this ticker
    # Ticker data
    st.header('**Ticker data**')
    st.write(tickerDf)

# Ticker information
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = ticker.info['longName']
st.header('**%s**' % string_name)

string_summary = ticker.info['longBusinessSummary']
st.info(string_summary)

# Bollinger bands
st.header('**Bollinger Bands**')
qf=cf.QuantFig(tickerDf,title='First Quant Figure',legend='top',name='GS')
qf.add_bollinger_bands()
fig = qf.iplot(asFigure=True)
st.plotly_chart(fig)

