#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Code for hiding warnings
import warnings
warnings.filterwarnings("ignore")

# In[2]:

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# In[3]:

import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st

# In[4]:

st.sidebar.image('st_logo.png')
st.sidebar.write("👋 Hi, I’m Shishir! I've always loved tinkering with things.")
st.sidebar.write("🌱 I’m a Finance Manager at Amazon and a former Data Scientist at Intel. I am also a Masters in Data Science Student at Northwestern.")
st.sidebar.write("📫 Reach me @ shishir.rd@gmail.com")
st.sidebar.write("My Github is https://github.com/shishirrd")

#Data Source
import yfinance as yf

#Data viz
import plotly.graph_objs as go

# In[5]:

from datetime import date

# In[ ]:

from dateutil.relativedelta import relativedelta
from datetime import date

# In[6]:

START = '2017-04-01'

# In[7]:

TODAY = date.today().strftime('%Y-%m-%d')

# In[ ]:

st.title("Tech stock prediction app")

# In[ ]:

stocks = ('HCLTECH.NS', 'KPITTECH.NS', 'AAPL', 'GOOG', 'AMD', 'NVDA', 'INTC',  
         'FB', 'MSFT', 'INFY.NS')

# In[ ]:

selected_stocks = st.selectbox("Select which stock you want to predict", stocks)

# In[ ]:

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# In[ ]:

#data_load_state = st.text("Loading stock data...")
data = load_data(selected_stocks)
#data_load_state.text("Loading stock data...Done!")

# In[ ]:

st.write("Last traded price is: ") 
st.write(round(data['Close'].iloc[-1],2))

# In[ ]:

n_years = st.slider("Select no. of years to predict", 1, 4)
season = st.slider('Select days to trend', 15, 180)

# In[ ]:

period = n_years * 365

# In[ ]:

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text = 'Current charts', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# In[ ]:


#ETS Forecasting
df_train = data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis='columns')
df_train = df_train.rename(columns={'Date':'DS', 'Close':'y'})

# In[ ]:

#Set index frequency as start of the day
df_train['DS'].freq = 'DS'

# In[ ]:

#Build a multiplicative ETS model
HWmodel2 = ExponentialSmoothing(df_train['y'], trend = 'mul', seasonal = 'mul', seasonal_periods=season).fit()

# In[ ]:

#Build regression model based on multiplicative ETS model
HWpred2 = HWmodel2.forecast(period)

# In[ ]:

#Wrap predictions into a pandas dataframe
prediction_data2 = pd.DataFrame(HWpred2, columns = ['Close'])

# In[ ]:

#Reset index
prediction_data2.reset_index(inplace=True)

# In[ ]:

#Add date column and wrap into a pandas dataframe
prediction_data2['Date'] = pd.DataFrame(pd.date_range(start=pd.to_datetime(date.today().strftime('%Y-%m-%d')), 
                                                     end=pd.to_datetime(date.today().strftime('%Y-%m-%d')) + 
                                                     np.timedelta64(period,'D'), freq='D'), columns=['Date'])

# In[ ]:

#Brief display of our forecast 
st.write("The final predicted price is...")
st.write(round(prediction_data2['Close'].iloc[-1],2))

# In[ ]:

plot_raw_data()

# In[ ]:

def plot_prediction2_data():
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=prediction_data2['Date'], y=prediction_data2['Close'], name='stock_forecast'))
    fig3.layout.update(title_text = 'ETS forecasted stock price', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig3)

# In[ ]:

plot_prediction2_data()

