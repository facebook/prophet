---
layout: docs
docid: "diagnostics"
title: "Diagnostics"
permalink: /docs/diagnostics.html
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
      <td>8.957184</td>
      <td>8.438130</td>
      <td>9.431683</td>
      <td>8.242493</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2010-02-17</td>
      <td>8.723619</td>
      <td>8.228941</td>
      <td>9.225985</td>
      <td>8.008033</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2010-02-18</td>
      <td>8.607378</td>
      <td>8.086717</td>
      <td>9.125563</td>
      <td>8.045268</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010-02-19</td>
      <td>8.529250</td>
      <td>8.053584</td>
      <td>9.056437</td>
      <td>7.928766</td>
      <td>2010-02-15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2010-02-20</td>
      <td>8.271228</td>
      <td>7.748368</td>
      <td>8.756539</td>
      <td>7.745003</td>
      <td>2010-02-15</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>3297</th>
      <td>37 days</td>
      <td>0.481970</td>
      <td>0.694241</td>
      <td>0.502930</td>
      <td>0.058371</td>
      <td>0.673367</td>
    </tr>
    <tr>
      <th>35</th>
      <td>37 days</td>
      <td>0.480991</td>
      <td>0.693535</td>
      <td>0.502007</td>
      <td>0.058262</td>
      <td>0.675879</td>
    </tr>
    <tr>
      <th>2207</th>
      <td>37 days</td>
      <td>0.480936</td>
      <td>0.693496</td>
      <td>0.501928</td>
      <td>0.058257</td>
      <td>0.675879</td>
    </tr>
    <tr>
      <th>2934</th>
      <td>37 days</td>
      <td>0.481455</td>
      <td>0.693870</td>
      <td>0.502999</td>
      <td>0.058393</td>
      <td>0.675879</td>
    </tr>
    <tr>
      <th>393</th>
      <td>37 days</td>
      <td>0.483990</td>
      <td>0.695694</td>
      <td>0.503418</td>
      <td>0.058494</td>
      <td>0.675879</td>
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
