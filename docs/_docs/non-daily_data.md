---
layout: docs
docid: "non-daily_data"
title: "Non-Daily Data"
permalink: /docs/non-daily_data.html
---
Prophet doesn't strictly require daily data, but you can get strange results if you ask for daily forecasts from non-daily data and fit seasonalities. Here we forecast US retail sales volume for the next 10 years:

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
 
![png](/prophet/static/non-daily_data_files/non-daily_data_4_0.png) 


The forecast here seems very noisy. What's happening is that this particular data set only provides monthly data. When we fit the yearly seasonality, it only has data for the first of each month and the seasonality components for the remaining days are unidentifiable and overfit. When you are fitting Prophet to monthly data, only make monthly forecasts, which can be done by passing the frequency into make_future_dataframe:

```R
# R
future <- make_future_dataframe(m, periods = 120, freq = 'm')
fcst <- predict(m, future)
plot(m, fcst)
```
```python
# Python
future = m.make_future_dataframe(periods=120, freq='M')
fcst = m.predict(future)
m.plot(fcst);
```
 
![png](/prophet/static/non-daily_data_files/non-daily_data_7_0.png) 

