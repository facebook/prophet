---
layout: docs
docid: "quick_start"
title: "Quick Start"
permalink: /docs/quick_start.html
subsections:
  - title: Python API
    id: python-api
  - title: R API
    id: r-api
---
<a id="python-api"> </a>

## Python API



Prophet follows the `sklearn` model API.  We create an instance of the `Prophet` class and then call its `fit` and `predict` methods.  


The input to Prophet is always a dataframe with two columns: `ds` and `y`.  The `ds` (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The `y` column must be numeric, and represents the measurement we wish to forecast.



As an example, let's look at a time series of the log daily page views for the Wikipedia page for [Peyton Manning](https://en.wikipedia.org/wiki/Peyton_Manning).  We scraped this data using the [Wikipediatrend](https://cran.r-project.org/package=wikipediatrend) package in R.  Peyton Manning provides a nice example because it illustrates some of Prophet's features, like multiple seasonality, changing growth rates, and the ability to model special days (such as Manning's playoff and superbowl appearances). The CSV is available [here](https://github.com/facebook/prophet/blob/master/examples/example_wp_log_peyton_manning.csv).



First we'll import the data:


```python
# Python
import pandas as pd
from prophet import Prophet
```
```python
# Python
df = pd.read_csv('../examples/example_wp_log_peyton_manning.csv')
df.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2007-12-10</td>
      <td>9.590761</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2007-12-11</td>
      <td>8.519590</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2007-12-12</td>
      <td>8.183677</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2007-12-13</td>
      <td>8.072467</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2007-12-14</td>
      <td>7.893572</td>
    </tr>
  </tbody>
</table>
</div>



We fit the model by instantiating a new `Prophet` object.  Any settings to the forecasting procedure are passed into the constructor.  Then you call its `fit` method and pass in the historical dataframe. Fitting should take 1-5 seconds.


```python
# Python
m = Prophet()
m.fit(df)
```
Predictions are then made on a dataframe with a column `ds` containing the dates for which a prediction is to be made. You can get a suitable dataframe that extends into the future a specified number of days using the helper method `Prophet.make_future_dataframe`. By default it will also include the dates from the history, so we will see the model fit as well. 


```python
# Python
future = m.make_future_dataframe(periods=365)
future.tail()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3265</th>
      <td>2017-01-15</td>
    </tr>
    <tr>
      <th>3266</th>
      <td>2017-01-16</td>
    </tr>
    <tr>
      <th>3267</th>
      <td>2017-01-17</td>
    </tr>
    <tr>
      <th>3268</th>
      <td>2017-01-18</td>
    </tr>
    <tr>
      <th>3269</th>
      <td>2017-01-19</td>
    </tr>
  </tbody>
</table>
</div>



The `predict` method will assign each row in `future` a predicted value which it names `yhat`.  If you pass in historical dates, it will provide an in-sample fit. The `forecast` object here is a new dataframe that includes a column `yhat` with the forecast, as well as columns for components and uncertainty intervals.


```python
# Python
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>yhat</th>
      <th>yhat_lower</th>
      <th>yhat_upper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3265</th>
      <td>2017-01-15</td>
      <td>8.211542</td>
      <td>7.444742</td>
      <td>8.903545</td>
    </tr>
    <tr>
      <th>3266</th>
      <td>2017-01-16</td>
      <td>8.536553</td>
      <td>7.847804</td>
      <td>9.211145</td>
    </tr>
    <tr>
      <th>3267</th>
      <td>2017-01-17</td>
      <td>8.323968</td>
      <td>7.541829</td>
      <td>9.035461</td>
    </tr>
    <tr>
      <th>3268</th>
      <td>2017-01-18</td>
      <td>8.156621</td>
      <td>7.404457</td>
      <td>8.830642</td>
    </tr>
    <tr>
      <th>3269</th>
      <td>2017-01-19</td>
      <td>8.168561</td>
      <td>7.438865</td>
      <td>8.908668</td>
    </tr>
  </tbody>
</table>
</div>



You can plot the forecast by calling the `Prophet.plot` method and passing in your forecast dataframe.


```python
# Python
fig1 = m.plot(forecast)
```
 
![png](/prophet/static/quick_start_files/quick_start_12_0.png) 


If you want to see the forecast components, you can use the `Prophet.plot_components` method.  By default you'll see the trend, yearly seasonality, and weekly seasonality of the time series.  If you include holidays, you'll see those here, too.


```python
# Python
fig2 = m.plot_components(forecast)
```
 
![png](/prophet/static/quick_start_files/quick_start_14_0.png) 


An interactive figure of the forecast and components can be created with plotly. You will need to install plotly 4.0 or above separately, as it will not by default be installed with prophet. You will also need to install the `notebook` and `ipywidgets` packages.


```python
# Python
from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)
```
```python
# Python
plot_components_plotly(m, forecast)
```
More details about the options available for each method are available in the docstrings, for example, via `help(Prophet)` or `help(Prophet.fit)`. The [R reference manual](https://cran.r-project.org/web/packages/prophet/prophet.pdf) on CRAN provides a concise list of all of the available functions, each of which has a Python equivalent.


<a id="r-api"> </a>

## R API



In R, we use the normal model fitting API.  We provide a `prophet` function that performs fitting and returns a model object.  You can then call `predict` and `plot` on this model object.


```R
# R
library(prophet)
```
    R[write to console]: Loading required package: Rcpp
    
    R[write to console]: Loading required package: rlang
    


First we read in the data and create the outcome variable. As in the Python API, this is a dataframe with columns `ds` and `y`, containing the date and numeric value respectively. The ds column should be YYYY-MM-DD for a date, or YYYY-MM-DD HH:MM:SS for a timestamp. As above, we use here the log number of views to Peyton Manning's Wikipedia page, available [here](https://github.com/facebook/prophet/blob/master/examples/example_wp_log_peyton_manning.csv).


```R
# R
df <- read.csv('../examples/example_wp_log_peyton_manning.csv')
```
We call the `prophet` function to fit the model.  The first argument is the historical dataframe.  Additional arguments control how Prophet fits the data and are described in later pages of this documentation.


```R
# R
m <- prophet(df)
```
Predictions are made on a dataframe with a column `ds` containing the dates for which predictions are to be made. The `make_future_dataframe` function takes the model object and a number of periods to forecast and produces a suitable dataframe. By default it will also include the historical dates so we can evaluate in-sample fit.


```R
# R
future <- make_future_dataframe(m, periods = 365)
tail(future)
```
                 ds
    3265 2017-01-14
    3266 2017-01-15
    3267 2017-01-16
    3268 2017-01-17
    3269 2017-01-18
    3270 2017-01-19


As with most modeling procedures in R, we use the generic `predict` function to get our forecast. The `forecast` object is a dataframe with a column `yhat` containing the forecast. It has additional columns for uncertainty intervals and seasonal components.


```R
# R
forecast <- predict(m, future)
tail(forecast[c('ds', 'yhat', 'yhat_lower', 'yhat_upper')])
```
                 ds     yhat yhat_lower yhat_upper
    3265 2017-01-14 7.818359   7.071228   8.550957
    3266 2017-01-15 8.200125   7.475725   8.869495
    3267 2017-01-16 8.525104   7.747071   9.226915
    3268 2017-01-17 8.312482   7.551904   9.046774
    3269 2017-01-18 8.145098   7.390770   8.863692
    3270 2017-01-19 8.156964   7.381716   8.866507


You can use the generic `plot` function to plot the forecast, by passing in the model and the forecast dataframe.


```R
# R
plot(m, forecast)
```
 
![png](/prophet/static/quick_start_files/quick_start_30_0.png) 


You can use the `prophet_plot_components` function to see the forecast broken down into trend, weekly seasonality, and yearly seasonality.


```R
# R
prophet_plot_components(m, forecast)
```
 
![png](/prophet/static/quick_start_files/quick_start_32_0.png) 


An interactive plot of the forecast using Dygraphs can be made with the command `dyplot.prophet(m, forecast)`.



More details about the options available for each method are available in the docstrings, for example, via `?prophet` or `?fit.prophet`. This documentation is also available in the [reference manual](https://cran.r-project.org/web/packages/prophet/prophet.pdf) on CRAN.

