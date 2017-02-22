---
layout: docs
docid: "holiday_effects"
title: "Holiday Effects"
permalink: /docs/holiday_effects.html
---
### Modeling Holidays
If you have holidays that you'd like to model, you must create a dataframe for them. It has two columns (`holiday` and `ds`) and a row for each occurrence of the holiday. It must include all occurrences of the holiday, both in the past (back as far as the historical data go) and in the future (out as far as the forecast is being made). If they won't repeat in the future, Prophet will model them and then not include them in the forecast.

You can also include columns `lower_window` and `upper_window` which extend the holiday out to `[lower_window, upper_window]` days around the date. For instance, if you wanted to included Christmas Eve in addition to Christmas you'd include `lower_window=-1,upper_window=0`. If you wanted to use Black Friday in addition to Thanksgiving, you'd include `lower_window=0,upper_window=1`.

Here we create a dataframe that includes the dates of all of Peyton Manning's playoff appearances:

```python
# Python
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))
```
```R
# R
library(dplyr)
playoffs <- data_frame(
  holiday = 'playoff',
  ds = as.Date(c('2008-01-13', '2009-01-03', '2010-01-16',
                 '2010-01-24', '2010-02-07', '2011-01-08',
                 '2013-01-12', '2014-01-12', '2014-01-19',
                 '2014-02-02', '2015-01-11', '2016-01-17',
                 '2016-01-24', '2016-02-07')),
  lower_window = 0,
  upper_window = 1
)
superbowls <- data_frame(
  holiday = 'superbowl',
  ds = as.Date(c('2010-02-07', '2014-02-02', '2016-02-07')),
  lower_window = 0,
  upper_window = 1
)
holidays <- bind_rows(playoffs, superbowls)
```
Above we have include the superbowl days as both playoff games and superbowl games. This means that the superbowl effect will be an additional additive bonus on top of the playoff effect.

Once the table is created, holiday effects are included in the forecast by passing them in with the `holidays` argument. Here we do it with the Peyton Manning data from the Quickstart:

```python
# Python
m = Prophet(holidays=holidays)
forecast = m.fit(df).predict(future)
```
```R
# R
m <- prophet(df, holidays = holidays)
forecast <- predict(m, future)
```
The holiday effect can be seen in the `forecast` dataframe:

```R
# R
forecast %>% 
  select(ds, playoff, superbowl) %>% 
  filter(abs(playoff + superbowl) > 0) %>%
  tail(10)
```
```python
# Python
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
        ['ds', 'playoff', 'superbowl']][-10:]
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>playoff</th>
      <th>superbowl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2190</th>
      <td>2014-02-02</td>
      <td>1.220308</td>
      <td>1.204992</td>
    </tr>
    <tr>
      <th>2191</th>
      <td>2014-02-03</td>
      <td>1.900465</td>
      <td>1.444581</td>
    </tr>
    <tr>
      <th>2532</th>
      <td>2015-01-11</td>
      <td>1.220308</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2533</th>
      <td>2015-01-12</td>
      <td>1.900465</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>2016-01-17</td>
      <td>1.220308</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>2016-01-18</td>
      <td>1.900465</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>2016-01-24</td>
      <td>1.220308</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>2016-01-25</td>
      <td>1.900465</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>2016-02-07</td>
      <td>1.220308</td>
      <td>1.204992</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>2016-02-08</td>
      <td>1.900465</td>
      <td>1.444581</td>
    </tr>
  </tbody>
</table>
</div>



The holiday effects will also show up in the components plot, where we see that there is a spike on the days around playoff appearances, with an especially large spike for the superbowl:

```python
# Python
m.plot_components(forecast);
```
```R
# R
prophet_plot_components(m, forecast);
```
 
![png](/prophet/static/holiday_effects_files/holiday_effects_13_0.png) 


### Prior scale for holidays and seasonality
If you find that the holidays are overfitting, you can adjust their prior scale to smooth them using the parameter `holidays_prior_scale`, which by default is 10:

```R
# R
m <- prophet(df, holidays = holidays, holidays.prior.scale = 1)
forecast <- predict(m, future)
forecast %>% 
  select(ds, playoff, superbowl) %>% 
  filter(abs(playoff + superbowl) > 0) %>%
  tail(10)
```
```python
# Python
m = Prophet(holidays=holidays, holidays_prior_scale=1).fit(df)
forecast = m.predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
    ['ds', 'playoff', 'superbowl']][-10:]
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ds</th>
      <th>playoff</th>
      <th>superbowl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2190</th>
      <td>2014-02-02</td>
      <td>1.362312</td>
      <td>0.693425</td>
    </tr>
    <tr>
      <th>2191</th>
      <td>2014-02-03</td>
      <td>2.033471</td>
      <td>0.542254</td>
    </tr>
    <tr>
      <th>2532</th>
      <td>2015-01-11</td>
      <td>1.362312</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2533</th>
      <td>2015-01-12</td>
      <td>2.033471</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>2016-01-17</td>
      <td>1.362312</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>2016-01-18</td>
      <td>2.033471</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>2016-01-24</td>
      <td>1.362312</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>2016-01-25</td>
      <td>2.033471</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>2016-02-07</td>
      <td>1.362312</td>
      <td>0.693425</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>2016-02-08</td>
      <td>2.033471</td>
      <td>0.542254</td>
    </tr>
  </tbody>
</table>
</div>



The magnitude of the holiday effect has been reduced compared to before, especially for superbowls, which had the fewest observations. There is a parameter `seasonality_prior_scale` which similarly adjusts the extent to which the seasonality model will fit the data.
