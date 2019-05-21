---
layout: docs
docid: "diagnostics"
title: "Diagnostics"
permalink: /docs/diagnostics.html
subsections:
---
Prophet includes functionality for time series cross validation to measure forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. We can then compare the forecasted values to the actual values. This figure illustrates a simulated historical forecast on the Peyton Manning dataset, where the model was fit to a initial history of 5 years, and a forecast was made on a one year horizon.


 
![png](/prophet/static/diagnostics_files/diagnostics_3_0.png) 


[The Prophet paper](https://peerj.com/preprints/3190.pdf) gives further description of simulated historical forecasts.



This cross validation procedure can be done automatically for a range of historical cutoffs using the `cross_validation` function. We specify the forecast horizon (`horizon`), and then optionally the size of the initial training period (`initial`) and the spacing between cutoff dates (`period`). By default, the initial training period is set to three times the horizon, and cutoffs are made every half a horizon.



The output of `cross_validation` is a dataframe with the true values `y` and the out-of-sample forecast values `yhat`, at each simulated forecast date and for each cutoff date. In particular, a forecast is made for every observed point between `cutoff` and `cutoff + horizon`. This dataframe can then be used to compute error measures of `yhat` vs. `y`.



Here we do cross-validation to assess prediction performance on a horizon of 365 days, starting with 730 days of training data in the first cutoff and then making predictions every 180 days. On this 8 year time series, this corresponds to 11 total forecasts.


```R
# R
df.cv <- cross_validation(m, initial = 730, period = 180, horizon = 365, units = 'days')
head(df.cv)
```
```python
# Python
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, initial='730 days', period='180 days', horizon = '365 days')
df_cv.head()
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
      <th>y</th>
      <th>cutoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-02-16</td>
      <td>8.956572</td>
      <td>8.460049</td>
      <td>9.460400</td>
      <td>8.242493</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-02-17</td>
      <td>8.723004</td>
      <td>8.200557</td>
      <td>9.236561</td>
      <td>8.008033</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-02-18</td>
      <td>8.606823</td>
      <td>8.070835</td>
      <td>9.123754</td>
      <td>8.045268</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-02-19</td>
      <td>8.528688</td>
      <td>8.034782</td>
      <td>9.042712</td>
      <td>7.928766</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-02-20</td>
      <td>8.270706</td>
      <td>7.754891</td>
      <td>8.739012</td>
      <td>7.745003</td>
      <td>2010-02-15</td>
    </tr>
  </tbody>
</table>
</div>



In R, the argument `units` must be a type accepted by `as.difftime`, which is weeks or shorter. In Python, the string for `initial`, `period`, and `horizon` should be in the format used by Pandas Timedelta, which accepts units of days or shorter.



The `performance_metrics` utility can be used to compute some useful statistics of the prediction performance (`yhat`, `yhat_lower`, and `yhat_upper` compared to `y`), as a function of the distance from the cutoff (how far into the future the prediction was). The statistics computed are mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), mean absolute percent error (MAPE), and coverage of the `yhat_lower` and `yhat_upper` estimates. These are computed on a rolling window of the predictions in `df_cv` after sorting by horizon (`ds` minus `cutoff`). By default 10% of the predictions will be included in each window, but this can be changed with the `rolling_window` argument.


```R
# R
df.p <- performance_metrics(df.cv)
head(df.p)
```
```python
# Python
from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()
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
      <th>horizon</th>
      <th>mse</th>
      <th>rmse</th>
      <th>mae</th>
      <th>mape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37 days</td>
      <td>0.495378</td>
      <td>0.703831</td>
      <td>0.505713</td>
      <td>0.058593</td>
      <td>0.680448</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38 days</td>
      <td>0.501134</td>
      <td>0.707908</td>
      <td>0.510680</td>
      <td>0.059169</td>
      <td>0.679077</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39 days</td>
      <td>0.523334</td>
      <td>0.723418</td>
      <td>0.516755</td>
      <td>0.059766</td>
      <td>0.677707</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40 days</td>
      <td>0.530625</td>
      <td>0.728440</td>
      <td>0.519645</td>
      <td>0.060075</td>
      <td>0.678849</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41 days</td>
      <td>0.538117</td>
      <td>0.733565</td>
      <td>0.520663</td>
      <td>0.060156</td>
      <td>0.686386</td>
    </tr>
  </tbody>
</table>
</div>



Cross validation performance metrics can be visualized with `plot_cross_validation_metric`, here shown for MAPE. Dots show the absolute percent error for each prediction in `df_cv`. The blue line shows the MAPE, where the mean is taken over a rolling window of the dots. We see for this forecast that errors around 5% are typical for predictions one month into the future, and that errors increase up to around 11% for predictions that are a year out.


```R
# R
plot_cross_validation_metric(df.cv, metric = 'mape')
```
```python
# Python
from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric='mape')
```
 
![png](/prophet/static/diagnostics_files/diagnostics_12_0.png) 


The size of the rolling window in the figure can be changed with the optional argument `rolling_window`, which specifies the proportion of forecasts to use in each rolling window. The default is 0.1, corresponding to 10% of rows from `df_cv` included in each window; increasing this will lead to a smoother average curve in the figure.



The `initial` period should be long enough to capture all of the components of the model, in particular seasonalities and extra regressors: at least a year for yearly seasonality, at least a week for weekly seasonality, etc.

