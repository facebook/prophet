---
layout: docs
docid: "seasonality,_holiday_effects,_and_regressors"
title: "Seasonality, Holiday Effects, And Regressors"
permalink: /docs/seasonality,_holiday_effects,_and_regressors.html
subsections:
  - title: Modeling Holidays and Special Events
    id: modeling-holidays-and-special-events
  - title: Built-in Country Holidays
    id: built-in-country-holidays
  - title: Holidays for subdivisions (Python)
    id: holidays-for-subdivisions-(python)
  - title: Fourier Order for Seasonalities
    id: fourier-order-for-seasonalities
  - title: Specifying Custom Seasonalities
    id: specifying-custom-seasonalities
  - title: Seasonalities that depend on other factors
    id: seasonalities-that-depend-on-other-factors
  - title: Prior scale for holidays and seasonality
    id: prior-scale-for-holidays-and-seasonality
  - title: Additional regressors
    id: additional-regressors
  - title: Coefficients of additional regressors
    id: coefficients-of-additional-regressors
  - title: Uncertainty in future regressor values
    id: uncertainty-in-future-regressor-values
---
<a id="modeling-holidays-and-special-events"> </a>

### Modeling Holidays and Special Events

If you have holidays or other recurring events that you'd like to model, you must create a dataframe for them. It has two columns (`holiday` and `ds`) and a row for each occurrence of the holiday. It must include all occurrences of the holiday, both in the past (back as far as the historical data go) and in the future (out as far as the forecast is being made). If they won't repeat in the future, Prophet will model them and then not include them in the forecast.



You can also include columns `lower_window` and `upper_window` which extend the holiday out to `[lower_window, upper_window]` days around the date. For instance, if you wanted to include Christmas Eve in addition to Christmas you'd include `lower_window=-1,upper_window=0`. If you wanted to use Black Friday in addition to Thanksgiving, you'd include `lower_window=0,upper_window=1`. You can also include a column `prior_scale` to set the prior scale separately for each holiday, as described below.



Here we create a dataframe that includes the dates of all of Peyton Manning's playoff appearances:


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
Above we have included the superbowl days as both playoff games and superbowl games. This means that the superbowl effect will be an additional additive bonus on top of the playoff effect.



Once the table is created, holiday effects are included in the forecast by passing them in with the `holidays` argument. Here we do it with the Peyton Manning data from the [Quickstart](https://facebook.github.io/prophet/docs/quick_start.html):


```R
# R
m <- prophet(df, holidays = holidays)
forecast <- predict(m, future)
```
```python
# Python
m = Prophet(holidays=holidays)
forecast = m.fit(df).predict(future)
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
      <th>playoff</th>
      <th>superbowl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2190</th>
      <td>2014-02-02</td>
      <td>1.223965</td>
      <td>1.201517</td>
    </tr>
    <tr>
      <th>2191</th>
      <td>2014-02-03</td>
      <td>1.901742</td>
      <td>1.460471</td>
    </tr>
    <tr>
      <th>2532</th>
      <td>2015-01-11</td>
      <td>1.223965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2533</th>
      <td>2015-01-12</td>
      <td>1.901742</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>2016-01-17</td>
      <td>1.223965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>2016-01-18</td>
      <td>1.901742</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>2016-01-24</td>
      <td>1.223965</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>2016-01-25</td>
      <td>1.901742</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>2016-02-07</td>
      <td>1.223965</td>
      <td>1.201517</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>2016-02-08</td>
      <td>1.901742</td>
      <td>1.460471</td>
    </tr>
  </tbody>
</table>
</div>



The holiday effects will also show up in the components plot, where we see that there is a spike on the days around playoff appearances, with an especially large spike for the superbowl:


```R
# R
prophet_plot_components(m, forecast)
```
```python
# Python
fig = m.plot_components(forecast)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_14_0.png)


Individual holidays can be plotted using the `plot_forecast_component` function (imported from `prophet.plot` in Python) like `plot_forecast_component(m, forecast, 'superbowl')` to plot just the superbowl holiday component.


<a id="built-in-country-holidays"> </a>

### Built-in Country Holidays



You can use a built-in collection of country-specific holidays using the `add_country_holidays` method (Python) or function (R). The name of the country is specified, and then major holidays for that country will be included in addition to any holidays that are specified via the `holidays` argument described above:


```R
# R
m <- prophet(holidays = holidays)
m <- add_country_holidays(m, country_name = 'US')
m <- fit.prophet(m, df)
```
```python
# Python
m = Prophet(holidays=holidays)
m.add_country_holidays(country_name='US')
m.fit(df)
```
You can see which holidays were included by looking at the `train_holiday_names` (Python) or `train.holiday.names` (R) attribute of the model:


```R
# R
m$train.holiday.names
```
     [1] "playoff"                     "superbowl"                  
     [3] "New Year's Day"              "Martin Luther King Jr. Day" 
     [5] "Washington's Birthday"       "Memorial Day"               
     [7] "Independence Day"            "Labor Day"                  
     [9] "Columbus Day"                "Veterans Day"               
    [11] "Veterans Day (Observed)"     "Thanksgiving"               
    [13] "Christmas Day"               "Independence Day (Observed)"
    [15] "Christmas Day (Observed)"    "New Year's Day (Observed)"  


```python
# Python
m.train_holiday_names
```



    0                         playoff
    1                       superbowl
    2                  New Year's Day
    3      Martin Luther King Jr. Day
    4           Washington's Birthday
    5                    Memorial Day
    6                Independence Day
    7                       Labor Day
    8                    Columbus Day
    9                    Veterans Day
    10                   Thanksgiving
    11                  Christmas Day
    12       Christmas Day (Observed)
    13        Veterans Day (Observed)
    14    Independence Day (Observed)
    15      New Year's Day (Observed)
    dtype: object



The holidays for each country are provided by the `holidays` package in Python. A list of available countries, and the country name to use, is available on their page: https://github.com/vacanza/python-holidays/.



In Python, most holidays are computed deterministically and so are available for any date range; a warning will be raised if dates fall outside the range supported by that country. In R, holiday dates are computed for 1995 through 2044 and stored in the package as `data-raw/generated_holidays.csv`. If a wider date range is needed, this script can be used to replace that file with a different date range: https://github.com/facebook/prophet/blob/main/python/scripts/generate_holidays_file.py.



As above, the country-level holidays will then show up in the components plot:


```R
# R
forecast <- predict(m, future)
prophet_plot_components(m, forecast)
```
```python
# Python
forecast = m.predict(future)
fig = m.plot_components(forecast)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_24_0.png)


<a id="holidays-for-subdivisions-(python)"> </a>

#### Holidays for subdivisions (Python)



We can use the utility function `make_holidays_df` to easily create custom holidays DataFrames, e.g. for certain states, using data from the [holidays](https://pypi.org/project/holidays/) package. This can be passed directly to the `Prophet()` constructor.


```python
# Python
from prophet.make_holidays import make_holidays_df

nsw_holidays = make_holidays_df(
    year_list=[2019 + i for i in range(10)], country='AU', province='NSW'
)
nsw_holidays.head(n=10)
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
      <th>holiday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019-01-26</td>
      <td>Australia Day</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2019-01-28</td>
      <td>Australia Day (Observed)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019-04-25</td>
      <td>Anzac Day</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2019-12-25</td>
      <td>Christmas Day</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-26</td>
      <td>Boxing Day</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2019-04-20</td>
      <td>Easter Saturday</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2019-04-21</td>
      <td>Easter Sunday</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2019-10-07</td>
      <td>Labour Day</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2019-06-10</td>
      <td>Queen's Birthday</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2019-08-05</td>
      <td>Bank Holiday</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Python
from prophet import Prophet

m_nsw = Prophet(holidays=nsw_holidays)
```
<a id="fourier-order-for-seasonalities"> </a>

### Fourier Order for Seasonalities



Seasonalities are estimated using a partial Fourier sum. See [the paper](https://peerj.com/preprints/3190/) for complete details, and [this figure on Wikipedia](https://en.wikipedia.org/wiki/Fourier_series#/media/File:Fourier_Series.svg) for an illustration of how a partial Fourier sum can approximate an arbitrary periodic signal. The number of terms in the partial sum (the order) is a parameter that determines how quickly the seasonality can change. To illustrate this, consider the Peyton Manning data from the [Quickstart](https://facebook.github.io/prophet/docs/quick_start.html). The default Fourier order for yearly seasonality is 10, which produces this fit:


```R
# R
m <- prophet(df)
prophet:::plot_yearly(m)
```
```python
# Python
from prophet.plot import plot_yearly
m = Prophet().fit(df)
a = plot_yearly(m)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_30_0.png)


The default values are often appropriate, but they can be increased when the seasonality needs to fit higher-frequency changes, and generally be less smooth. The Fourier order can be specified for each built-in seasonality when instantiating the model, here it is increased to 20:


```R
# R
m <- prophet(df, yearly.seasonality = 20)
prophet:::plot_yearly(m)
```
```python
# Python
from prophet.plot import plot_yearly
m = Prophet(yearly_seasonality=20).fit(df)
a = plot_yearly(m)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_33_0.png)


Increasing the number of Fourier terms allows the seasonality to fit faster changing cycles, but can also lead to overfitting: N Fourier terms corresponds to 2N variables used for modeling the cycle



<a id="specifying-custom-seasonalities"> </a>

### Specifying Custom Seasonalities



Prophet will by default fit weekly and yearly seasonalities, if the time series is more than two cycles long. It will also fit daily seasonality for a sub-daily time series. You can add other seasonalities (monthly, quarterly, hourly) using the `add_seasonality` method (Python) or function (R).



The inputs to this function are a name, the period of the seasonality in days, and the Fourier order for the seasonality. For reference, by default Prophet uses a Fourier order of 3 for weekly seasonality and 10 for yearly seasonality. An optional input to `add_seasonality` is the prior scale for that seasonal component - this is discussed below.



As an example, here we fit the Peyton Manning data from the [Quickstart](https://facebook.github.io/prophet/docs/quick_start.html), but replace the weekly seasonality with monthly seasonality. The monthly seasonality then will appear in the components plot:


```R
# R
m <- prophet(weekly.seasonality=FALSE)
m <- add_seasonality(m, name='monthly', period=30.5, fourier.order=5)
m <- fit.prophet(m, df)
forecast <- predict(m, future)
prophet_plot_components(m, forecast)
```
```python
# Python
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_36_0.png)


<a id="seasonalities-that-depend-on-other-factors"> </a>

### Seasonalities that depend on other factors

In some instances the seasonality may depend on other factors, such as a weekly seasonal pattern that is different during the summer than it is during the rest of the year, or a daily seasonal pattern that is different on weekends vs. on weekdays. These types of seasonalities can be modeled using conditional seasonalities.



Consider the Peyton Manning example from the [Quickstart](https://facebook.github.io/prophet/docs/quick_start.html). The default weekly seasonality assumes that the pattern of weekly seasonality is the same throughout the year, but we'd expect the pattern of weekly seasonality to be different during the on-season (when there are games every Sunday) and the off-season. We can use conditional seasonalities to construct separate on-season and off-season weekly seasonalities.



First we add a boolean column to the dataframe that indicates whether each date is during the on-season or the off-season:


```R
# R
is_nfl_season <- function(ds) {
  dates <- as.Date(ds)
  month <- as.numeric(format(dates, '%m'))
  return(month > 8 | month < 2)
}
df$on_season <- is_nfl_season(df$ds)
df$off_season <- !is_nfl_season(df$ds)
```
```python
# Python
def is_nfl_season(ds):
    date = pd.to_datetime(ds)
    return (date.month > 8 or date.month < 2)

df['on_season'] = df['ds'].apply(is_nfl_season)
df['off_season'] = ~df['ds'].apply(is_nfl_season)
```
Then we disable the built-in weekly seasonality, and replace it with two weekly seasonalities that have these columns specified as a condition. This means that the seasonality will only be applied to dates where the `condition_name` column is `True`. We must also add the column to the `future` dataframe for which we are making predictions.


```R
# R
m <- prophet(weekly.seasonality=FALSE)
m <- add_seasonality(m, name='weekly_on_season', period=7, fourier.order=3, condition.name='on_season')
m <- add_seasonality(m, name='weekly_off_season', period=7, fourier.order=3, condition.name='off_season')
m <- fit.prophet(m, df)

future$on_season <- is_nfl_season(future$ds)
future$off_season <- !is_nfl_season(future$ds)
forecast <- predict(m, future)
prophet_plot_components(m, forecast)
```
```python
# Python
m = Prophet(weekly_seasonality=False)
m.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_season')
m.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_season')

future['on_season'] = future['ds'].apply(is_nfl_season)
future['off_season'] = ~future['ds'].apply(is_nfl_season)
forecast = m.fit(df).predict(future)
fig = m.plot_components(forecast)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_42_0.png)


Both of the seasonalities now show up in the components plots above. We can see that during the on-season when games are played every Sunday, there are large increases on Sunday and Monday that are completely absent during the off-season.


<a id="prior-scale-for-holidays-and-seasonality"> </a>

### Prior scale for holidays and seasonality

If you find that the holidays are overfitting, you can adjust their prior scale to smooth them using the parameter `holidays_prior_scale`. By default this parameter is 10, which provides very little regularization. Reducing this parameter dampens holiday effects:


```R
# R
m <- prophet(df, holidays = holidays, holidays.prior.scale = 0.05)
forecast <- predict(m, future)
forecast %>% 
  select(ds, playoff, superbowl) %>% 
  filter(abs(playoff + superbowl) > 0) %>%
  tail(10)
```
```python
# Python
m = Prophet(holidays=holidays, holidays_prior_scale=0.05).fit(df)
forecast = m.predict(future)
forecast[(forecast['playoff'] + forecast['superbowl']).abs() > 0][
    ['ds', 'playoff', 'superbowl']][-10:]
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
      <th>playoff</th>
      <th>superbowl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2190</th>
      <td>2014-02-02</td>
      <td>1.206086</td>
      <td>0.964914</td>
    </tr>
    <tr>
      <th>2191</th>
      <td>2014-02-03</td>
      <td>1.852077</td>
      <td>0.992634</td>
    </tr>
    <tr>
      <th>2532</th>
      <td>2015-01-11</td>
      <td>1.206086</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2533</th>
      <td>2015-01-12</td>
      <td>1.852077</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2901</th>
      <td>2016-01-17</td>
      <td>1.206086</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2902</th>
      <td>2016-01-18</td>
      <td>1.852077</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2908</th>
      <td>2016-01-24</td>
      <td>1.206086</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2909</th>
      <td>2016-01-25</td>
      <td>1.852077</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2922</th>
      <td>2016-02-07</td>
      <td>1.206086</td>
      <td>0.964914</td>
    </tr>
    <tr>
      <th>2923</th>
      <td>2016-02-08</td>
      <td>1.852077</td>
      <td>0.992634</td>
    </tr>
  </tbody>
</table>
</div>



The magnitude of the holiday effect has been reduced compared to before, especially for superbowls, which had the fewest observations. There is a parameter `seasonality_prior_scale` which similarly adjusts the extent to which the seasonality model will fit the data.



Prior scales can be set separately for individual holidays by including a column `prior_scale` in the holidays dataframe. Prior scales for individual seasonalities can be passed as an argument to `add_seasonality`. For instance, the prior scale for just weekly seasonality can be set using:


```R
# R
m <- prophet()
m <- add_seasonality(
  m, name='weekly', period=7, fourier.order=3, prior.scale=0.1)
```
```python
# Python
m = Prophet()
m.add_seasonality(
    name='weekly', period=7, fourier_order=3, prior_scale=0.1)
```


<a id="additional-regressors"> </a>

### Additional regressors

Additional regressors can be added to the linear part of the model using the `add_regressor` method or function. A column with the regressor value will need to be present in both the fitting and prediction dataframes. For example, we can add an additional effect on Sundays during the NFL season. On the components plot, this effect will show up in the 'extra_regressors' plot:


```R
# R
nfl_sunday <- function(ds) {
  dates <- as.Date(ds)
  month <- as.numeric(format(dates, '%m'))
  as.numeric((weekdays(dates) == "Sunday") & (month > 8 | month < 2))
}
df$nfl_sunday <- nfl_sunday(df$ds)

m <- prophet()
m <- add_regressor(m, 'nfl_sunday')
m <- fit.prophet(m, df)

future$nfl_sunday <- nfl_sunday(future$ds)

forecast <- predict(m, future)
prophet_plot_components(m, forecast)
```
```python
# Python
def nfl_sunday(ds):
    date = pd.to_datetime(ds)
    if date.weekday() == 6 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0
df['nfl_sunday'] = df['ds'].apply(nfl_sunday)

m = Prophet()
m.add_regressor('nfl_sunday')
m.fit(df)

future['nfl_sunday'] = future['ds'].apply(nfl_sunday)

forecast = m.predict(future)
fig = m.plot_components(forecast)
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_52_0.png)


NFL Sundays could also have been handled using the "holidays" interface described above, by creating a list of past and future NFL Sundays. The `add_regressor` function provides a more general interface for defining extra linear regressors, and in particular does not require that the regressor be a binary indicator. Another time series could be used as a regressor, although its future values would have to be known.



[This notebook](https://nbviewer.jupyter.org/github/nicolasfauchereau/Auckland_Cycling/blob/master/notebooks/Auckland_cycling_and_weather.ipynb) shows an example of using weather factors as extra regressors in a forecast of bicycle usage, and provides an excellent illustration of how other time series can be included as extra regressors.



The `add_regressor` function has optional arguments for specifying the prior scale (holiday prior scale is used by default) and whether or not the regressor is standardized - see the docstring with `help(Prophet.add_regressor)` in Python and `?add_regressor` in R. Note that regressors must be added prior to model fitting. Prophet will also raise an error if the regressor is constant throughout the history, since there is nothing to fit from it.



The extra regressor must be known for both the history and for future dates. It thus must either be something that has known future values (such as `nfl_sunday`), or something that has separately been forecasted elsewhere. The weather regressors used in the notebook linked above is a good example of an extra regressor that has forecasts that can be used for future values. 



One can also use as a regressor another time series that has been forecasted with a time series model, such as Prophet. For instance, if `r(t)` is included as a regressor for `y(t)`, Prophet can be used to forecast `r(t)` and then that forecast can be plugged in as the future values when forecasting `y(t)`. A note of caution around this approach: This will probably not be useful unless `r(t)` is somehow easier to forecast than `y(t)`. This is because error in the forecast of `r(t)` will produce error in the forecast of `y(t)`. One setting where this can be useful is in hierarchical time series, where there is top-level forecast that has higher signal-to-noise and is thus easier to forecast. Its forecast can be included in the forecast for each lower-level series.



Extra regressors are put in the linear component of the model, so the underlying model is that the time series depends on the extra regressor as either an additive or multiplicative factor (see the next section for multiplicativity).



<a id="coefficients-of-additional-regressors"> </a>

#### Coefficients of additional regressors



To extract the beta coefficients of the extra regressors, use the utility function `regressor_coefficients` (`from prophet.utilities import regressor_coefficients` in Python, `prophet::regressor_coefficients` in R) on the fitted model. The estimated beta coefficient for each regressor roughly represents the increase in prediction value for a unit increase in the regressor value (note that the coefficients returned are always on the scale of the original data). If `mcmc_samples` is specified, a credible interval for each coefficient is also returned, which can help identify whether the regressor is meaningful to the model (a credible interval that includes the 0 value suggests the regressor is not meaningful).


<a id="uncertainty-in-future-regressor-values"> </a>

## Uncertainty in future regressor values



In the examples above, regressor values are linked to holidays / calendar events, so their future values are known and can be set directly in the `future` DataFrame and passed to `.predict()`.



In multi-stage prediction pipelines, future regressor values are unknown and carry uncertainty. As of Prophet v1.4.0 there is support for nested Prophet models (one per extra_regressor) that integrate natively with `.predict()` and `cross_validate()`.



1. In `.add_regressor()`, set `predict_spec` to enable a nested Prophet model for that regressor. `predictor_spec=True` uses defaults; pass a dictionary like `predict_spec={'changepoint_range': 0.7}` for a custom config. Note that we only support **simple** nested Prophet models, so models that require custom seasonalities with `.add_seasonality()` or their own regressors via `.add_regressor()` are **not** supported.



2. When running `predict()` on the main model, regressor values are filled with the `yhat` values from their respective nested Prophet models.



3. When running `cross_validate()` on the main model, each run will first predict the regressor values for the lookahead range, and those values are used to predict `y` for the lookahead range. This prevents data leakage and overstated cross-validation results. 


```python
# Python
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import pandas as pd

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

pedestrians = (
    pd.read_csv(
        'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_pedestrians_multivariate.csv',
        parse_dates=["ds"],
    )
    .astype({"location_4": np.float64, "location_41": np.float64})
    .rename({"location_41": "y"}, axis=1)
)

target = "location_41"

test_mask = pedestrians["ds"] >= pd.to_datetime("2023-04-24 00:00:00")
pedestrians_train = pedestrians.loc[~test_mask]
pedestrians_test = pedestrians.loc[test_mask]
```
```python
# Python
m1 = Prophet(
    changepoint_range=0.9,
    seasonality_prior_scale=5.0,
)
m1.add_regressor("location_4", mode="additive", regressor_predictor={"seasonality_prior_scale": 10.0});
```
```python
# Python
m1.fit(pedestrians_train);
```
    INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:prophet:Fitting regressor model 'location_4' with 552 observations
    INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:cmdstanpy:Chain [1] start processing
    INFO:cmdstanpy:Chain [1] done processing
    INFO:cmdstanpy:Chain [1] start processing
    INFO:cmdstanpy:Chain [1] done processing


```python
# Python
preds = m1.predict(pedestrians_test)
```
    INFO:prophet:Running regressor model 'location_4' for 168 future dates
    INFO:prophet:Collected 1000 predictive samples for regressor 'location_4'


```python
# Python
m1.plot(preds);
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_59_0.png)


```python
# Python
m1.plot_components(preds);
```

![png](/prophet/static/seasonality,_holiday_effects,_and_regressors_files/seasonality,_holiday_effects,_and_regressors_60_0.png)


```python
# Python
m1_cv = cross_validation(
    m1,
    horizon=pd.Timedelta(3, unit="D"),
    period=pd.Timedelta(5, unit="D"),
    initial=pd.Timedelta(14, unit="D"),
)
```
    INFO:prophet:Making 2 forecasts with cutoffs between 2023-04-15 23:00:00 and 2023-04-20 23:00:00



      0%|          | 0/2 [00:00<?, ?it/s]


    INFO:prophet:Fitting regressor model 'location_4' with 360 observations
    INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:cmdstanpy:Chain [1] start processing
    INFO:cmdstanpy:Chain [1] done processing
    INFO:cmdstanpy:Chain [1] start processing
    INFO:cmdstanpy:Chain [1] done processing
    INFO:prophet:Running regressor model 'location_4' for 72 future dates
    INFO:prophet:Collected 1000 predictive samples for regressor 'location_4'
    INFO:prophet:Fitting regressor model 'location_4' with 480 observations
    INFO:prophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
    INFO:cmdstanpy:Chain [1] start processing
    INFO:cmdstanpy:Chain [1] done processing
    INFO:cmdstanpy:Chain [1] start processing
    INFO:cmdstanpy:Chain [1] done processing
    INFO:prophet:Running regressor model 'location_4' for 72 future dates
    INFO:prophet:Collected 1000 predictive samples for regressor 'location_4'


```python
# Python
performance_metrics(m1_cv).tail(n=5).round(4)
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
      <th>mdape</th>
      <th>smape</th>
      <th>coverage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>2 days 20:00:00</td>
      <td>121911.5202</td>
      <td>349.1583</td>
      <td>230.5407</td>
      <td>0.0627</td>
      <td>0.0506</td>
      <td>0.0654</td>
      <td>0.9286</td>
    </tr>
    <tr>
      <th>62</th>
      <td>2 days 21:00:00</td>
      <td>120364.2551</td>
      <td>346.9355</td>
      <td>233.8541</td>
      <td>0.0707</td>
      <td>0.0547</td>
      <td>0.0722</td>
      <td>0.9286</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2 days 22:00:00</td>
      <td>117161.5735</td>
      <td>342.2887</td>
      <td>217.8324</td>
      <td>0.0674</td>
      <td>0.0539</td>
      <td>0.0689</td>
      <td>0.9286</td>
    </tr>
    <tr>
      <th>64</th>
      <td>2 days 23:00:00</td>
      <td>127608.6985</td>
      <td>357.2236</td>
      <td>239.9352</td>
      <td>0.0878</td>
      <td>0.0601</td>
      <td>0.0866</td>
      <td>0.9286</td>
    </tr>
    <tr>
      <th>65</th>
      <td>3 days 00:00:00</td>
      <td>192080.9740</td>
      <td>438.2704</td>
      <td>308.6779</td>
      <td>0.2583</td>
      <td>0.0734</td>
      <td>0.1865</td>
      <td>0.8571</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Python

```