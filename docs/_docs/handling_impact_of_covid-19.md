---
layout: docs
docid: "handling_impact_of_covid-19"
title: "Handling Impact Of Covid-19"
permalink: /docs/handling_impact_of_covid-19.html
subsections:
  - title: Default model without any adjustments
    id: default-model-without-any-adjustments
  - title: Treating COVID-19 lockdowns as a one-off holidays
    id: treating-covid-19-lockdowns-as-a-one-off-holidays
  - title: Allowing Prophet to incorporate post-lockdown trends into the forecast
    id: allowing-prophet-to-incorporate-post-lockdown-trends-into-the-forecast
  - title: Changes in seasonality between pre- and post-COVID
    id: changes-in-seasonality-between-pre--and-post-covid
  - title: Further reading
    id: further-reading
---
```python
# Python
%matplotlib inline
from prophet import Prophet
import pandas as pd
from matplotlib import pyplot as plt
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['figure.figsize'] = 9, 6
```
As a result of the lockdowns caused by the COVID-19 pandemic, many time series experienced "shocks" during 2020 and into 2021. Some examples are:



* Online activity spiking - media (Netflix, YouTube, etc.) consumption, e-commerce (Amazon, eBay) purchases, social media, etc.

* Physical activity declining heavily - restaurant visits, cinema sales, etc.



Along with a heavy increase or decline around the time of the lockdown, most of these time series would also afterwards maintain a higher or lower average level compared to pre-COVID. These levels may fluctuate again as lockdowns are loosened and tightened in response to outbreaks, or with the rollout of vaccines.



Finally, seasonal patterns could have changed: for example, people may have consumed less media (in total hours) on weekdays compared to weekends before the COVID lockdowns, but during lockdowns weekday consumption could be much closer to weekend consumption.



All of the above presents some tough challenges for forecasting:



1. Models shouldn't assume the one-off spikes at the beginning of COVID will occur again in the near term.

2. Models should properly capture the changes in the underlying trend cause by medium/long-term changes people's activity as a result of lockdowns.

3. Models should properly capture any changes in seasonality caused by changes in people's routines (work from home, less going out, etc.).



It's unlikely that our models will be able to do all of the above. Furthermore, other influences like the rollout of vaccines, and repeated lockdowns caused by second/third waves, are impossible to predict.



In this page we suggest some strategies to help _mitigate_ (but not necessarily fully solve) the impact of COVID-19 lockdowns in time series projections.


The dataset we will use to demonstrate these strategies is [Pedestrian Sensor data from the City of Melbourne](https://data.melbourne.vic.gov.au/Transport/Pedestrian-Counting-System-Monthly-counts-per-hour/b2ak-trbp). This data measures foot traffic from sensors in various places in the Central Business District (CBD) in Melbourne, Australia. For this case study, we've taken data from the sensor with the most volume (`Sensor_ID = 4, Town Hall (West)`), and aggregated the data to the daily level (the original dataset provides hourly counts) from May 2009 to May 2021. The aggregated dataset can be found in the examples folder [here](https://github.com/facebook/prophet/tree/master/examples/example_pedestrians.csv). 


```python
# Python
df = pd.read_csv('../examples/example_pedestrians.csv')
```
The full history of data (May 2009 to May 2021):


```python
# Python
df.set_index('ds').plot()
```



    <AxesSubplot:xlabel='ds'>



 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_5_1.png) 


Zooming in closer to the COVID impacts:


```python
# Python
df.loc[df['ds'] >= '2019-09-01'].set_index('ds').plot()
```



    <AxesSubplot:xlabel='ds'>



 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_7_1.png) 


<a id="default-model-without-any-adjustments"> </a>

#### Default model without any adjustments


First we'll fit a model with the default Prophet settings:


```python
# Python
m = Prophet()
m.fit(df)
future = m.make_future_dataframe(periods=366)
```
```python
# Python
forecast = m.predict(future)
```
```python
# Python
m.plot(forecast)
plt.axhline(y=0, color='red')
plt.title('Default Prophet')
```



    Text(0.5, 1.0, 'Default Prophet')



 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_12_1.png) 


The forecast from the default Prophet model for this dataset shows the following issues:



* The model doesn't pick up the low foot traffic around mid-March 2020, then later from early July to mid-October 2020, as outliers, and instead incorporates them into the trend.

* The model does not pick up the increasing trend in early 2021, when lockdowns were loosened.

* The negative sloping trend causes predictions to go negative in late 2021 and 2022.


<a id="treating-covid-19-lockdowns-as-a-one-off-holidays"> </a>

### Treating COVID-19 lockdowns as a one-off holidays


This first strategy treats the days impacted by COVID-19 as a holiday that will not repeat again in the future. Adding custom holidays is explained in more detail [here](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#modeling-holidays-and-special-events).


```python
# Python
# Lockdown 1 = 2020-03-21 to 2020-06-06 inclusive
# Lockdown 2 (second wave) = 2020-07-09 to 2021-10-27 inclusive
lockdowns = pd.DataFrame({
    'holiday': ['lockdown1', 'lockdown2'],
    'ds': pd.to_datetime(["2020-03-21", '2020-07-09']),
    'lower_window': [0, 0],
    'upper_window': [78, 110],
})
```
```python
# Python
m2 = Prophet(holidays=lockdowns)
m2.fit(df)
future2 = m2.make_future_dataframe(periods=366)
```
```python
# Python
forecast2 = m2.predict(future2)
```
```python
# Python
m2.plot(forecast2)
plt.axhline(y=0, color='red')
plt.title('Lockdowns as one-off holidays')
```



    Text(0.5, 1.0, 'Lockdowns as one-off holidays')



 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_19_1.png) 


The forecasts already look more reasonable by specifying the two lockdown periods - they are no longer negative.



We can also plot the model components to check the magnitude of the effect that Prophet is "attributing" to the lockdowns:


```python
# Python
m2.plot_components(forecast2)
plt.show()
```
 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_21_0.png) 


* Prophet is sensibly assigning a large negative effect to the days within the lockdown periods.

* Day-of-year seasonality also looks reasonable (i.e. March and July seasonality don't seem to be impacted too heavily by the 2020 shocks).



However, the trend component is still has a negative slope from 2020 onwards, and doesn't incorporate the recovery from early 2021 onwards (when the second-wave lockdown was lifted). The main cause of this is that Prophet uses the first 80% of the data to fit trend changepoints by default. The trend changes in early 2021 fall outside this 80%, so we need to adjust this parameter.


<a id="allowing-prophet-to-incorporate-post-lockdown-trends-into-the-forecast"> </a>

### Allowing Prophet to incorporate post-lockdown trends into the forecast


How Prophet automatically finds changepoints for the trend component is explained in detail [here](https://facebook.github.io/prophet/docs/trend_changepoints.html). 



Since the default 80% changepoint range doesn't consider the post-lockdown recovery for the pedestrians dataset, we find a value such that Prophet will consider data up to April 2021 in its changepoint estimation:


```python
# Python
def percent_history_at(df: pd.DataFrame, ds: str) -> float:
    """
    For a given timestamp, find the x% of the historical data that is required to contain that datapoint.
    Used for customizing changepoint_range for Prophet.
    """
    nrows_before = df[pd.to_datetime(df['ds']) <= pd.to_datetime(ds)].shape[0]
    return nrows_before / df.shape[0]
```
```python
# Python
print(f"{percent_history_at(df, '2021-04-01'):.2%}")
```
    99.30%


Note that this value will vary per dataset - the reason it's such a high number for the pedestrians dataset is because we have data from 2009 to 2021. Datasets that span a shorter range of time wouldn't need such a high `changepoint_range`.


```python
# Python
m3 = Prophet(holidays=lockdowns, changepoint_range=0.993)
m3.fit(df)
```



    <prophet.forecaster.Prophet at 0x122e1a220>



```python
# Python
forecast3 = m3.predict(future2)
```
```python
# Python
m3.plot(forecast3)
plt.axhline(y=0, color='red')
plt.title('Lockdowns as one-off holidays + Incorporate recent data into trend')
```



    Text(0.5, 1.0, 'Lockdowns as one-off holidays + Incorporate recent data into trend')



 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_30_1.png) 


```python
# Python
m3.plot_components(forecast3)
plt.show()
```
 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_31_0.png) 


We can see the difference in the trend estimation between this model and the model fitted in the previous section.



* Instead of the trend continuing to slope down, there is a little elbow towards the end of 2020, when the second lockdown was lifted.

* The projected pedestrian values are now more stable around the level of the most recent data, instead trending downwards towards 0.


<a id="changes-in-seasonality-between-pre--and-post-covid"> </a>

### Changes in seasonality between pre- and post-COVID


As we mentioned in the introduction, COVID (both the virus and the lockdowns) could also affect people's routines, to the point where seasonality patterns pre-COVID no longer hold. 



For our case study of pedestrian activity, the day-of-week seasonality of our models in the previous sections shows that there was a lot more activity on Friday (and to an extent Thursday and Saturday) compared to other days of the week. If we're not sure whether this will still hold after lockdowns are lifted (especially given that vaccines weren't yet rolled out widely when the lockdown was lifted), we can add _conditional seasonalities_ to the model. Conditional seasonalities are explained in more detail [here](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#seasonalities-that-depend-on-other-factors).


```python
# Python
df2 = df.copy()
```
```python
# Python
df2['pre_covid'] = pd.to_datetime(df2['ds']) < pd.to_datetime('2020-03-21')
```
```python
# Python
df2['post_covid'] = ~df2['pre_covid']
```
```python
# Python
m4 = Prophet(holidays=lockdowns, changepoint_range=0.9930, weekly_seasonality=False)
m4.add_seasonality(
    name='weekly_pre_covid',
    period=7,
    fourier_order=3,
    condition_name='pre_covid',
)
m4.add_seasonality(
    name='weekly_post_covid',
    period=7,
    fourier_order=3,
    condition_name='post_covid',
)
```



    <prophet.forecaster.Prophet at 0x122f71b20>



```python
# Python
m4.fit(df2)
```



    <prophet.forecaster.Prophet at 0x122f71b20>



```python
# Python
future4 = m4.make_future_dataframe(periods=366)
```
```python
# Python
future4['pre_covid'] = pd.to_datetime(future4['ds']) < pd.to_datetime('2020-03-21')
future4['post_covid'] = ~future4['pre_covid']
```
```python
# Python
forecast4 = m4.predict(future4)
```
```python
# Python
m4.plot(forecast4)
plt.axhline(y=0, color='red')
plt.title('Lockdowns as one-off holidays + Incorporate recent data into trend + Conditional weekly seasonality')
```



    Text(0.5, 1.0, 'Lockdowns as one-off holidays + Incorporate recent data into trend + Conditional weekly seasonality')



 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_43_1.png) 


```python
# Python
m4.plot_components(forecast4)
plt.show()
```
 
![png](/prophet/static/handling_impact_of_covid-19_files/handling_impact_of_covid-19_44_0.png) 


* Interestingly, the model with conditional seasonalities suggests that, post-COVID, pedestrian activity in this area peaks on Saturdays, instead of Fridays. This could make sense if most people are still working from home and are hence less likely to go out on Friday nights.

* At a glance, the overall level of the predictions don't change too much, but the trend does flatten out more. The difference in the Friday and Saturday forecasts in particular could also affect per-day planning.

* The best way to validate whether this is a better model is to use [cross validation](https://facebook.github.io/prophet/docs/diagnostics.html).


<a id="further-reading"> </a>

### Further reading


The case study presented above may not be suitable to all contexts. See the following thread for a more extensive discussion of different strategies for handling the impact of COVID-19 in different contexts:



* https://github.com/facebook/prophet/issues/1416



See also these articles:



* [How to forecast demand despite COVID](https://medium.com/swlh/how-to-forecast-demand-despite-covid-82d22a0a6ff7)

