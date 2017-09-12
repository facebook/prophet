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

The output of `cross_validation` is a dataframe with the true values `y` and the out-of-sample forecast values `yhat`, at each simulated forecast date and for each cutoff date. This dataframe can then be used to compute error measures of `yhat` vs. `y`.

```R
# R
df.cv <- cross_validation(m, horizon = 730, units = 'days')
head(df.cv)
```
```python
# Python
from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(m, horizon = '730 days')
df_cv.head()
```



<div>
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
      <td>2014-01-21</td>
      <td>9.439510</td>
      <td>8.799215</td>
      <td>10.080240</td>
      <td>10.542574</td>
      <td>2014-01-20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-01-22</td>
      <td>9.267086</td>
      <td>8.645900</td>
      <td>9.882225</td>
      <td>10.004283</td>
      <td>2014-01-20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-01-23</td>
      <td>9.263447</td>
      <td>8.628803</td>
      <td>9.852847</td>
      <td>9.732818</td>
      <td>2014-01-20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-01-24</td>
      <td>9.277452</td>
      <td>8.693226</td>
      <td>9.897891</td>
      <td>9.866460</td>
      <td>2014-01-20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-01-25</td>
      <td>9.087565</td>
      <td>8.447306</td>
      <td>9.728898</td>
      <td>9.370927</td>
      <td>2014-01-20</td>
    </tr>
  </tbody>
</table>
</div>


