---
layout: docs
docid: "non-daily_data"
title: "Non-Daily Data"
permalink: /docs/non-daily_data.html
---
## Sub-daily data

Prophet can make forecasts for time series with sub-daily observations by passing in a dataframe with timestamps in the `ds` column. When sub-daily data are used, daily seasonality will automatically be fit. Here we fit Prophet to data with 5-minute resolution (daily temperatures at Yosemite):

```R
# R
df <- read.csv('../examples/example_yosemite_temps.csv')
m <- prophet(df, changepoint.prior.scale=0.01)
future <- make_future_dataframe(m, periods = 300, freq = 60 * 60)
fcst <- predict(m, future)
plot(m, fcst);
```
```python
# Python
df = pd.read_csv('../examples/example_yosemite_temps.csv')
m = Prophet(changepoint_prior_scale=0.01).fit(df)
future = m.make_future_dataframe(periods=300, freq='H')
fcst = m.predict(future)
m.plot(fcst);
```
 
![png](/prophet/static/non-daily_data_files/non-daily_data_4_0.png) 


The daily seasonality will show up in the components plot:

```R
# R
prophet_plot_components(m, fcst)
```
```python
# Python
m.plot_components(fcst);
```
 
![png](/prophet/static/non-daily_data_files/non-daily_data_7_0.png) 


## Monthly data

You can use Prophet to fit monthly data. However, the underlying model is continuous-time, which means that you can get strange results if you fit the model to monthly data and then ask for daily forecasts. Here we forecast US retail sales volume for the next 10 years:

```R
# R
df <- read.csv('../examples/example_retail_sales.csv')
m <- prophet(df)
future <- make_future_dataframe(m, periods = 3652)
fcst <- predict(m, future)
plot(m, fcst);
```
```python
# Python
df = pd.read_csv('../examples/example_retail_sales.csv')
m = Prophet().fit(df)
future = m.make_future_dataframe(periods=3652)
fcst = m.predict(future)
m.plot(fcst);
```
 
![png](/prophet/static/non-daily_data_files/non-daily_data_10_0.png) 


The forecast here seems very noisy. What's happening is that this particular data set only provides monthly data. When we fit the yearly seasonality, it only has data for the first of each month and the seasonality components for the remaining days are unidentifiable and overfit. When you are fitting Prophet to monthly data, only make monthly forecasts, which can be done by passing the frequency into make_future_dataframe:

```R
# R
future <- make_future_dataframe(m, periods = 120, freq = 'month')
fcst <- predict(m, future)
plot(m, fcst)
```
```python
# Python
future = m.make_future_dataframe(periods=120, freq='M')
fcst = m.predict(future)
m.plot(fcst);
```
 
![png](/prophet/static/non-daily_data_files/non-daily_data_13_0.png) 

