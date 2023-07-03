import pandas as pd
import plotly.express as px
from tkinter import Tk
from tkinter.filedialog import askopenfilename
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
Tk().withdraw()
filename = askopenfilename()
separador=','
indice='time'
valor='close'

df = pd.read_csv(filename, sep=separador)
df = df.astype({valor: float})
df = df.astype({indice: 'datetime64[s]'})
df[indice]= pd.to_datetime(getattr(df,indice), unit='s')
df.index = df[indice]
df = df.sort_index(ascending=True,axis=0)
data = pd.DataFrame(index=range(0,len(df)),columns=[indice,valor])
for i in range(0,len(data)):
    data[indice][i]=df[indice][i]
    data[valor][i]=df[valor][i]
data.index=getattr(df,indice)
data.drop(indice,axis=1,inplace=True)


final_data = data.values
print(final_data)
#final_data = final_data[len(final_data):,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_data)

# generate the input and output sequences
n_lookback = 360 # length of input sequences (lookback period)
n_forecast = 20  # length of output sequences (forecast period)

X = []
Y = []

for i in range(n_lookback, len(scaled_data) - n_forecast + 1):
    X.append(scaled_data[i - n_lookback: i])
    Y.append(scaled_data[i: i + n_forecast])


X = np.array(X)
Y = np.array(Y)
# fit the model
model = Sequential()
model.add(GRU(units=256, return_sequences=True, input_shape=(np.shape(X)[1],1)))
model.add(GRU(units=256))
model.add(Dense(n_forecast))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=20, batch_size=1024, verbose=2)

# generate the forecasts
X_ = scaled_data[- n_lookback:]  # last available input sequence
X_ = X_.reshape(1, n_lookback, 1)

Y_ = model.predict(X_).reshape(-1, 1)
Y_ = scaler.inverse_transform(Y_)

# organize the results in a data frame
df_past = df[[valor]].reset_index()
df_past.rename(columns={'index': indice, valor: 'Actual'}, inplace=True)
df_past[indice] = pd.to_datetime(df_past[indice])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=[indice, 'Actual', 'Forecast'])
df_future[indice] = pd.date_range(start=df_past[indice].iloc[-1] + pd.Timedelta(minutes=1), periods=n_forecast, freq='min')
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = df_past.append(df_future).set_index(indice)

fig = px.line(results)
fig.show()