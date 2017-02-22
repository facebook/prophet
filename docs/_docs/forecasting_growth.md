---
layout: docs
docid: "forecasting_growth"
title: "Forecasting Growth"
permalink: /docs/forecasting_growth.html
---
By default, Prophet uses a linear model for its forecast. When forecasting growth, there is usually some maximum achievable point: total market size, total population size, etc. This is called the carrying capacity, and the forecast should saturate at this point.

Prophet allows you to make forecasts using a [logistic growth](https://en.wikipedia.org/wiki/Logistic_function) trend model, with a specified carrying capacity. We illustrate this with the log number of page visits to the [R (programming language)](https://en.wikipedia.org/wiki/R_%28programming_language%29) page on Wikipedia:

```python
# Python
df = pd.read_csv('../examples/example_wp_R.csv')
import numpy as np
df['y'] = np.log(df['y'])
```
```R
# R
df <- read.csv('../examples/example_wp_R.csv')
df$y <- log(df$y)
```
We must specify the carrying capacity in a column `cap`. Here we will assume a particular value, but this would usually be set using data or expertise about the market size.

```python
# Python
df['cap'] = 8.5
```
```R
# R
df$cap <- 8.5
```
The important things to note are that `cap` must be specified for every row in the dataframe, and that it does not have to be constant. If the market size is growing, then `cap` can be an increasing sequence.

We then fit the model as before, except pass in an additional argument to specify logistic growth:

```python
# Python
m = Prophet(growth='logistic')
m.fit(df)
```
```R
# R
m <- prophet(df, growth = 'logistic')
```
We make a dataframe for future predictions as before, except we must also specify the capacity in the future. Here we keep capacity constant at the same value as in the history, and forecast 3 years into the future:

```python
# Python
future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
m.plot(fcst);
```
```R
# R
future <- make_future_dataframe(m, periods = 1826)
future$cap <- 8.5
fcst <- predict(m, future)
plot(m, fcst);
```
 
![png](/prophet/static/forecasting_growth_files/forecasting_growth_13_0.png) 

