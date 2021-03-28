try:
	import pandas as pd
except:
	import os
	os.system("pip install pandas")
	import pandas as pd

try:
	import numpy as np
except:
	import os
	os.system("pip install numpy")
	import numpy as np

try:
	import matplotlib.pyplot as plt
except:
	import os
	os.system("pip install matplotlib")
	import matplotlib.pyplot as plt

try:
	import statsmodels.tsa.seasonal as stss
except:
	import os
	os.system("pip install statsmodels")
	import statsmodels.tsa.seasonal as stss

try:
	import easygui as eg
except:
	import os
	os.system("pip install easygui")
	import easygui as eg

import statsmodels.tsa.stattools as stst
import statsmodels.api as sm
import itertools
from datetime import datetime, timedelta

plt.rcParams["axes.unicode_minus"] = False

file_directory = eg.fileopenbox("open file", "open file")
df = pd.read_csv(file_directory)
eg.textbox("Original data checking (top 5 records)", "Original data", str(df.head(5)))
df.info()
print("\n")
print("\n")

df = pd.read_csv(file_directory, index_col=['DELIVERY_DATE'], parse_dates=['DELIVERY_DATE'])
df.sort_index(inplace=True)
df = df.fillna(0)

plt.plot(df.index, df["DELIVERED_VOLUME"])
plt.xlabel("year-month-day")
plt.ylabel("Delivery Volume")
plt.show()

ts = df[pd.Series(pd.to_datetime(df.index, errors = "coerce")).notnull().values]
ts["DELIVERED_VOLUME"] = pd.to_numeric(ts["DELIVERED_VOLUME"], errors = "coerce")
ts.fillna(0)
ts = ts["DELIVERED_VOLUME"]

decomposition = stss.seasonal_decompose(ts, period = 7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(4, 1, 1)
plt.plot(ts)
plt.title("Original data")
plt.subplot(4, 1, 2)
plt.plot(trend)
plt.title("Trend")
plt.subplot(4, 1, 3)
plt.plot(seasonal)
plt.title("Seasonality")
plt.subplot(4, 1, 4)
plt.plot(residual)
plt.title("Residual")
plt.tight_layout()
plt.show()

def TestStationaryPlot(ts): 

	rol_mean = ts.rolling(window = 7, center = False).mean() 
	rol_std = ts.rolling(window = 7, center = False).std()          

	plt.plot(ts, color = "blue", label = "Original Data")                           
	plt.plot(rol_mean, color = "red", linestyle = "-.", label = "Moving Average")  
	plt.plot(rol_std, color = "green", linestyle = "--", label = "Standard Deviation")    

	plt.xlabel("year-month-day")  
	plt.ylabel("Delivery Volume")    
	plt.legend(loc = "best")   
	plt.title("Moving Average & Standard Deviation")
	plt.show(block = True)

def TestStationaryAdfuller(ts, cutoff = 0.05):                                                                                   
	ts_test = stst.adfuller(ts, autolag = "AIC")                                                                                 
	ts_test_output = pd.Series(ts_test[0:4], index = ["Test Statistic", "p-value", "#Lags Used", "Numbers of Observation Used"]) 
	global non_stationary                                                                                                        
	non_stationary = 0                                                                                                        

	for key, value in ts_test[4].items():
		ts_test_output["Critical Value (%s)" % key] = value
	print(ts_test_output)                                    
	ts_test_output = str(ts_test_output)
	if ts_test[1] <= cutoff:
		ts_test_output = ts_test_output + "\n\n We reject the null hypothesis, that is, the data is stationary."
		eg.textbox("Analysis of Stationariness", "Result", ts_test_output)
		print("\n")
		print("\n")
	else:
		ts_test_output = ts_test_output + "We fail to reject the null hypothesis, that is, the data is not stationary."
		eg.textbox("Analysis of Stationariness", "Result", ts_test_output)
		non_stationary = 1
		print("\n")
		print("\n")

TestStationaryPlot(ts)                                                                                                                                                              
cutoff_setting = float(eg.enterbox(msg = "Setting alpha： \nif n < 100 set alpha = 0.001\nif 100 < n < 400 set alpha = 0.01\nif n > 400 set alpha = 0.05\n\nthe current n is：" + str(ts.count()), 
                                  title = "", default = "", strip = True))                                                                                                           
TestStationaryAdfuller(ts, cutoff = cutoff_setting)  

ts.dropna(inplace = True)                                        
r, q, p = sm.tsa.acf(ts.values.squeeze(), qstat = True)
data = np.c_[range(1, 41), r[1:], q, p]                             
table = pd.DataFrame(data, columns = ["lag", "AC", "Q", "Prob(>Q)"])
white_noise_result = str(table.set_index("lag"))
eg.textbox("Test of White Noise", "Result", white_noise_result)  

fig = plt.figure(figsize = (12, 8))                      
ax1 = fig.add_subplot(2, 1, 1)                           
fig = sm.graphics.tsa.plot_acf(ts, lags = 14, ax = ax1)  
ax2 = fig.add_subplot(2, 1, 2)                           
fig = sm.graphics.tsa.plot_pacf(ts, lags = 14, ax = ax2) 
plt.show()                                               

p = range(0, 3)
d = range(0, 2)
q = range(0, 2)                                               
pdq = list(itertools.product(p, d, q))                                        
pdq_x_PDQs = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

a = [] 
b = [] 
c = [] 

wf = pd.DataFrame()
for param in pdq:
	for seasonal_param in pdq_x_PDQs:
		try:
			mod = sm.tsa.statespace.SARIMAX(ts, order = param, seasonal_order = seasonal_param, enforce_stationarity = False, enforce_invertibility = False)
			results = mod.fit()
			print('ARIMA{}x{} - AIC:{}'.format(param, seasonal_param, results.aic))
			a.append(param)
			b.append(seasonal_param)
			c.append(results.aic)
		except:
			continue
wf['pdq'] = a
wf['pdq_x_PDQs'] = b
wf['aic'] = c
wf_best = wf[wf['aic']==wf['aic'].min()]
arima_par = str(wf_best["pdq"].values.squeeze())
sarima_par = str(wf_best["pdq_x_PDQs"].values.squeeze())
print("\n")
print("\n")
grid_search_result = "ARIMA parameter = " + str(arima_par) + "\n\n" + "SARIMA parameter = " + str(sarima_par)
eg.textbox("Result：Best combination", "Grid Search SARIMA parameter", grid_search_result)  

arima_input = eg.enterbox(msg = "Setting ARIMA parameter：Previous Best combination is：" + grid_search_result, 
                                  title = "", default = "", strip = True)
sarima_input = eg.enterbox(msg = "Setting SARIMA parameter：Previous Best combination is：" + grid_search_result, 
                                  title = "", default = "", strip = True)

arima_par = arima_input.split(",")
arima_par = [int(i) for i in arima_par]
sarima_par = sarima_input.split(",")
sarima_par = [int(i) for i in sarima_par]

mod = sm.tsa.statespace.SARIMAX(ts, order = arima_par, seasonal_order = sarima_par, enforce_stationarity = False, enforce_invertibility = False)
results = mod.fit()
print(results.summary())
print("\n")
print("\n")

pred = results.get_prediction(start = pd.to_datetime(ts.index[-10]), dynamic = False)
pred_ci = pred.conf_int()
ax = ts['2019-03-13':].plot(label = 'observed')
pred.predicted_mean.plot(ax = ax, label='One-step ahead Forecast', alpha = 0.7, figsize = (14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = 0.15)
ax.set_xlabel('Date')
ax.set_ylabel('Delivery Volume')
plt.legend()
plt.show()

ts_truth = ts[ts.index[-10]: ]
ts_forecasted = pred.predicted_mean
ts_diff = ts_truth - ts_forecasted
ts_diff_percentage = np.abs(ts_diff / ts_truth).replace(np.inf, np.nan).dropna().mean()

print(ts_diff_percentage)

# SARIMA(3, 1, 2, 7), MAPE = 0.09

# import warnings
# import itertools
# import numpy as np
# import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")
# plt.style.use('fivethirtyeight')
# import pandas as pd
# import matplotlib.pyplot as plt
# import datetime as dt 
# from matplotlib.pylab import rcParams
# import statsmodels.api as sm
# from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
# from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

# rcParams['axes.labelsize'] = 24
# rcParams['xtick.labelsize'] = 12
# rcParams['ytick.labelsize'] = 12
# rcParams['text.color'] = 'k'

# def mean_absolute_percentage_error(y_true, y_pred): 
#     print(np.mean(np.abs((y_true - y_pred) / y_true)))

# Delivery = pd.read_csv('F:/Baruch College/Data Challenge/Delivery_Volume.csv', index_col=['DELIVERY_DATE'], parse_dates=['DELIVERY_DATE'])
# Delivery.sort_index(inplace=True)
# Delivery=Delivery.fillna(0)

# print(Delivery)
# print(Delivery.index.min(), Delivery.index.max())

# plt.plot(Delivery.index, Delivery.DELIVERED_VOLUME)
# plt.show()

# y = Delivery["DELIVERED_VOLUME"].resample("1D").mean()
# y = y.fillna(0)
# print(y)

# z = Delivery["DELIVERED_VOLUME"].dt.dayofweek()

# print(z)

# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.show()

# # The plot above clearly shows that the delivered volume are unstable, along with its obvious seasonality.

# # p = d = q = range(0, 5)
# # pdq = list(itertools.product(p, d, q))
# # seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# # print('Examples of parameter combinations for Seasonal ARIMA...')
# # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# # for param in pdq:
# # 	for param_seasonal in seasonal_pdq:
# # 		mod = sm.tsa.statespace.SARIMAX(y, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)

# # 		results = mod.fit()

# # 		print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

# mod = sm.tsa.statespace.SARIMAX(y, order = (0, 0, 4), seasonal_order = (3, 1, 0, 12), enforce_stationarity = False, enforce_invertibility = False)

# results = mod.fit()

# print(results.summary().tables[1])

# results.plot_diagnostics(figsize = (16, 8))
# plt.show()

# pred = results.get_prediction(start = pd.to_datetime(y.index[-10]), dynamic = False)
# pred_ci = pred.conf_int()
# ax = y['2019':].plot(label = 'observed')
# pred.predicted_mean.plot(ax = ax, label='One-step ahead Forecast', alpha = 0.7, figsize = (14, 7))
# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color = 'k', alpha = 0.15)
# ax.set_xlabel('Date')
# ax.set_ylabel('Delivery Volume')
# plt.legend()
# plt.show()

# y_forecasted = pred.predicted_mean
# y_truth = y['2019-05-24': ]

# y_diff = y_truth - y_forecasted
# print(y_truth)
# print(y_forecasted)
# y_diff_values = []
# y_diff_values_squared = []
# for x in y_diff.index:
# 	y_diff_values.append(y_diff[x])

# for i in y_diff_values:
# 	y_diff_values_squared.append(i*i)

# mse = np.mean(y_diff_values_squared)
# print("The MSE of SARIMA Forecast is {}".format(round(mse, 2)))
# print('The RMSE of SARIMA Forecast is {}'.format(round(np.sqrt(mse), 2)))

# y_diff_percentage = np.abs(y_diff / y_truth)

# print(y_diff_percentage)
