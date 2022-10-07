#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import datetime


# In[4]:


import pandas as pd
import numpy as np


# In[5]:


from matplotlib import pyplot as plt


# In[6]:


from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from statsmodels.tsa.arima_model import ARIMA


# In[7]:


cd C:\Users\hp\Download


# In[8]:


2006-1996


# In[9]:


dx = pd.DataFrame()
for i in range(11):
    df = pd.read_csv('ogora' + str(1996+i) + '_' + str(20220901) + '.txt', sep=",", header=None)
    df = df[df.columns[:8]]
    df.columns = ['LeaseNo', 'CompletionNo', 'Month', 'DaysOnProd', 'ProductCode', 'MonthlyOil', 'MonthlyGas', 'MonthlyWater']
    dx = pd.concat([dx, df], axis = 0)


# In[10]:


2021-2006


# In[11]:


dy = pd.DataFrame()
for i in range(15):
    df = pd.read_csv('ogora' + str(2007+i) + '_' + str(20220915) + '.txt', sep=",", header=None)
    df = df[df.columns[:8]]
    df.columns = ['LeaseNo', 'CompletionNo', 'Month', 'DaysOnProd', 'ProductCode', 'MonthlyOil', 'MonthlyGas', 'MonthlyWater']
    dy = pd.concat([dy, df], axis = 0)


# In[12]:


dz = pd.concat([dx, dy], axis = 0)


# In[13]:


dx['Month'].unique()


# In[14]:


dy['Month'].unique()


# In[15]:


dz['total'] = dz['MonthlyOil'] + dz['MonthlyGas'] + dz['MonthlyWater']


# In[16]:


dz


# In[17]:


d1 = pd.DataFrame()
d1['time'] = dz['Month']
d1['oil'] = dz['MonthlyOil']
d1 = d1[d1['oil']>0]

d11 = d1.groupby(['time'])['oil'].sum()
d11 = pd.DataFrame(d11)
d11.reset_index(inplace = True)


# In[18]:


for i in d11['time'].index:
#     print(i)
#     print(str(d11.loc[i, 'time'])[-2:])
    b = str(d11.loc[i, 'time'])[-2:]
    a = str(d11.loc[i, 'time'])[:4]
    d11.loc[i, 'year'] = a
    d11.loc[i, 'month'] = b
    d11.loc[i, 'date'] = a + '-'+ b


# In[19]:


d11['date'] = pd.to_datetime(d11['date'])
d11.set_index('date', inplace = True)


# In[20]:


d2 = pd.DataFrame()
d2['time'] = dz['Month']
d2['gas'] = dz['MonthlyGas']
d2 = d2[d2['gas']>0]

d12 = d2.groupby(['time'])['gas'].sum()
d12 = pd.DataFrame(d12)
d12.reset_index(inplace = True)


# In[21]:


for i in d12['time'].index:
#     print(i)
#     print(str(d11.loc[i, 'time'])[-2:])
    b = str(d12.loc[i, 'time'])[-2:]
    a = str(d12.loc[i, 'time'])[:4]
    d12.loc[i, 'year'] = a
    d12.loc[i, 'month'] = b
    d12.loc[i, 'date'] = a + '-'+ b
    
d12['date'] = pd.to_datetime(d12['date'])
d12.set_index('date', inplace = True)


# In[22]:


d3 = pd.DataFrame()
d3['time'] = dz['Month']
d3['water'] = dz['MonthlyWater']
d3 = d3[d3['water']>0]

d13 = d3.groupby(['time'])['water'].sum()
d13 = pd.DataFrame(d13)
d13.reset_index(inplace = True)


# In[23]:


for i in d13['time'].index:
#     print(i)
#     print(str(d11.loc[i, 'time'])[-2:])
    b = str(d13.loc[i, 'time'])[-2:]
    a = str(d13.loc[i, 'time'])[:4]
    d13.loc[i, 'year'] = a
    d13.loc[i, 'month'] = b
    d13.loc[i, 'date'] = a + '-'+ b
    
d13['date'] = pd.to_datetime(d13['date'])
d13.set_index('date', inplace = True)


# In[24]:


d11


# In[25]:


d12


# In[26]:


d13


# In[ ]:





# In[27]:


#decompose
ts_oil = d11['oil']
from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(ts_oil)
decompose.plot();


# In[46]:


#rolling stats
#simple moving average
plt.plot(ts_oil)
plt.plot(ts_oil.rolling(window=4).mean())


# In[29]:


#augmented dicky fuller

adfuller(ts_oil)


# In[30]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[31]:


ts_oil


# In[32]:


np.log(ts_oil)


# In[33]:


log_diff_oil=np.log(ts_oil).diff(periods=1).dropna()
log_diff_oil2=np.log(ts_oil).diff(periods=2).dropna()


# In[34]:


#augmented dicky fuller

adfuller(log_diff_oil)


# In[35]:


#rolling stats
#simple moving average
plt.plot(log_diff_oil)
plt.plot(log_diff_oil.rolling(window=8).mean())


# In[37]:


smt.graphics.plot_acf(log_diff_oil);


# In[38]:


smt.graphics.plot_pacf(log_diff_oil);


# In[39]:


smt.graphics.plot_acf(log_diff_oil2);


# In[40]:


smt.graphics.plot_pacf(log_diff_oil2);


# In[41]:


model = ARIMA(ts_oil, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[42]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[43]:


# Actual vs Fitted
plt.figure(figsize = (16, 4))
model_fit.plot_predict(dynamic=False)
plt.show()


# In[44]:


ts_oil


# In[47]:


#log_oil = np.log(ts_oil)


# In[ ]:





# In[ ]:





# In[72]:


#decompose
ts_gas = d12['gas']
decompose=seasonal_decompose(ts_gas)
decompose.plot();


# In[73]:


#rolling stats
#simple moving average
plt.plot(ts_gas)
plt.plot(ts_gas.rolling(window=16).mean())


# In[74]:


#augmented dicky fuller

adfuller(ts_gas)


# In[75]:


log_diff_gas=np.log(ts_gas).diff(periods=1).dropna()
log_diff_gas2=np.log(ts_gas).diff(periods=2).dropna()


# In[76]:


#augmented dicky fuller

adfuller(log_diff_gas)


# In[77]:


#rolling stats
#simple moving average

plt.plot(log_diff_gas)
plt.plot(log_diff_gas.rolling(window=8).mean())


# In[78]:


smt.graphics.plot_acf(log_diff_gas);


# In[79]:


smt.graphics.plot_pacf(log_diff_gas);


# In[80]:


model = ARIMA(ts_gas, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[81]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[82]:


# Actual vs Fitted
plt.figure(figsize = (16, 4))
model_fit.plot_predict(dynamic=False)
plt.show()


# In[83]:


ts_gas


# In[94]:


#decompose
ts_water= d13['water']
decompose=seasonal_decompose(ts_water)
decompose.plot();


# In[95]:


#rolling stats
#simple moving average
plt.plot(ts_water)
plt.plot(ts_water.rolling(window=8).mean())


# In[96]:


#augmented dicky fuller

adfuller(ts_water)


# In[97]:


log_diff_water=np.log(ts_water).diff(periods=1).dropna()
log_diff_water2=np.log(ts_water).diff(periods=2).dropna()


# In[98]:


#augmented dicky fuller

adfuller(log_diff_water)


# In[99]:


#rolling stats
#simple moving average
plt.plot(log_diff_water)
plt.plot(log_diff_water.rolling(window=8).mean())


# In[100]:


smt.graphics.plot_acf(log_diff_water);


# In[101]:


smt.graphics.plot_pacf(log_diff_water);


# In[102]:


model = ARIMA(ts_water, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[103]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[104]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False);


# In[ ]:





# In[ ]:





# # Predicting on test dataset

# In[69]:


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    return({'mape':mape})


# In[68]:


# Create Training and Test
train = ts_oil[:260]
test = ts_oil[260:]

# Build Model
# model = ARIMA(train, order=(3,2,1))  
model = ARIMA(train, order=(5, 1, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())
# Forecast
fc, se, conf = fitted.forecast(52, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.plot(ts_oil.rolling(window=12).mean())
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.ylim(0, 2)
plt.show()


# In[112]:


forecast_accuracy(fc, test.values)


# In[92]:


# Create Training and Test
train = ts_gas[:260]
test = ts_gas[260:]

# Build Model  
model = ARIMA(train, order=(4, 1, 4))  
fitted = model.fit(disp=-1)  
print(fitted.summary())
# Forecast
fc, se, conf = fitted.forecast(52, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.plot(ts_gas.rolling(window=12).mean())
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.ylim(10, 20)
plt.show()


# In[70]:


forecast_accuracy(fc, test.values)


# In[111]:


# Create Training and Test
train = ts_water[:260]
test = ts_water[260:]

# Build Model
# model = ARIMA(train, order=(3,2,1))  
model = ARIMA(train, order=(4, 2, 2))  
fitted = model.fit(disp=-1)  
print(fitted.summary())
# Forecast
fc, se, conf = fitted.forecast(52, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.plot(ts_water.rolling(window=12).mean())
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.ylim(10, 20)
plt.show()


# In[93]:


forecast_accuracy(fc, test.values)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




