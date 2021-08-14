---
layout: docs
docid: "multivariate_top_level_regressor"
title: "Multivariate Top Level Regressor"
permalink: /docs/multivariate_top_level_regressor.html
subsections:
  - title: Dataset for case study - Pedestrian traffic by location
    id: dataset-for-case-study---pedestrian-traffic-by-location
  - title: Data prep - filling missing dates
    id: data-prep---filling-missing-dates
  - title: Evaluation criteria
    id: evaluation-criteria
  - title: Prophet configuration
    id: prophet-configuration
  - title: Strategy 1: Independent Models
    id: strategy-1:-independent-models
  - title: Strategy 2: Top-level regressor
    id: strategy-2:-top-level-regressor
  - title: Comparison
    id: comparison
  - title: Visual comparison of out-of-sample predictions
    id: visual-comparison-of-out-of-sample-predictions
  - title: `SMAPE` comparisons
    id: `smape`-comparisons
  - title: Uncertainty interval comparisons 
    id: uncertainty-interval-comparisons-
  - title: Further reading
    id: further-reading
---
```python
# Python
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter
from prophet import Prophet

plt.rcParams['figure.figsize'] = 9, 6
```
In this article we explore an approach to segment-level forecasts (a form of multivariate forecasting) using the top-level series as an extra regressor.


<a id="dataset-for-case-study---pedestrian-traffic-by-location"> </a>

### Dataset for case study - Pedestrian traffic by location



We use the pedestrians dataset from the City of Melbourne, which proxies foot traffic via sensors at various locations in the Melbourne CBD (central business district). We want to predict future foot traffic at key sensor locations; we can think of this as a multivariate forecasting problem, or producing "segment-level" forecasts.



The dataset has been cleaned and aggregated to contain daily counts by location in long form (i.e. one row per day + location), shown below. We have 10 key sensor locations, and the aim is to produce accurate out-of-sample forecasts for each.


```python
# Python
peds = pd.read_csv("../examples/example_pedestrians_multivariate.csv", parse_dates=['ds'])
```
```python
# Python
peds
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
      <th>sensor_id</th>
      <th>ds</th>
      <th>y</th>
      <th>sensor_description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2010-01-01</td>
      <td>14024</td>
      <td>Bourke Street Mall (North)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2010-01-02</td>
      <td>18895</td>
      <td>Bourke Street Mall (North)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2010-01-03</td>
      <td>14775</td>
      <td>Bourke Street Mall (North)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2010-01-04</td>
      <td>21857</td>
      <td>Bourke Street Mall (North)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2010-01-05</td>
      <td>20640</td>
      <td>Bourke Street Mall (North)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24626</th>
      <td>15</td>
      <td>2016-12-27</td>
      <td>16790</td>
      <td>State Library</td>
    </tr>
    <tr>
      <th>24627</th>
      <td>15</td>
      <td>2016-12-28</td>
      <td>17008</td>
      <td>State Library</td>
    </tr>
    <tr>
      <th>24628</th>
      <td>15</td>
      <td>2016-12-29</td>
      <td>14963</td>
      <td>State Library</td>
    </tr>
    <tr>
      <th>24629</th>
      <td>15</td>
      <td>2016-12-30</td>
      <td>18702</td>
      <td>State Library</td>
    </tr>
    <tr>
      <th>24630</th>
      <td>15</td>
      <td>2016-12-31</td>
      <td>23528</td>
      <td>State Library</td>
    </tr>
  </tbody>
</table>
<p>24631 rows × 4 columns</p>
</div>



```python
# Python
segments = peds['sensor_description'].unique()
segments
```



    array(['Bourke Street Mall (North)', 'Bourke Street Mall (South)',
           'Princes Bridge', 'Flinders Street Station Underpass',
           'Webb Bridge', 'Southern Cross Station', 'Waterfront City',
           'New Quay', 'Flagstaff Station', 'State Library'], dtype=object)



```python
# Python
peds['ds'].min(), peds['ds'].max()
```



    (Timestamp('2010-01-01 00:00:00'), Timestamp('2016-12-31 00:00:00'))



<a id="data-prep---filling-missing-dates"> </a>

#### Data prep - filling missing dates



Often, the way we collect data for segment-level forecasts leads to missing dates for certain segments. This can happen as a result of the segment having a zero value for that date, or as a result of a bug in data collection. It's important to distinguish between the two scenarios, otherwise the model training and model evaluation could ignore zero values and hence be biased.



In this dataset, all sensor locations should have non-zero daily counts, so any missing values are likely a result of a physical fault with the sensor or loss of data. This means we do want these values to be `NA` (so they can be excluded from training and validation), and it's good to be explicit about this.



Below we create rows for every day for each segment, and fill the missing values with `np.nan`. 



We'll also rename the column describing the sensor locations to `segment` and use this to label our forecasts.


```python
# Python
def clean_data(df: pd.DataFrame, segment_col, missing_fill):
    """
    Ensures each segment has a measurement for each date between the min and max time within the dataframe.

    Fills missing measurements with `fill_value`, which defaults to 0.
    """
    dates = pd.DataFrame({'ds': pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='D')})
    segs = pd.DataFrame({'segment': df[segment_col].unique()})
    base_table = pd.merge(dates, segs, how='cross')
    base_table = pd.merge(base_table, df[['ds', segment_col, 'y']].rename(columns={segment_col: 'segment'}), on=['ds', 'segment'], how='left')
    base_table['y'] = base_table['y'].fillna(missing_fill)
    return base_table.sort_values(['segment', 'ds'])
```
```python
# Python
df = clean_data(peds, 'sensor_description', missing_fill=np.nan)
```
```python
# Python
df
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
      <th>segment</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-01-01</td>
      <td>Bourke Street Mall (North)</td>
      <td>14024.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2010-01-02</td>
      <td>Bourke Street Mall (North)</td>
      <td>18895.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2010-01-03</td>
      <td>Bourke Street Mall (North)</td>
      <td>14775.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2010-01-04</td>
      <td>Bourke Street Mall (North)</td>
      <td>21857.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2010-01-05</td>
      <td>Bourke Street Mall (North)</td>
      <td>20640.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25524</th>
      <td>2016-12-27</td>
      <td>Webb Bridge</td>
      <td>2388.0</td>
    </tr>
    <tr>
      <th>25534</th>
      <td>2016-12-28</td>
      <td>Webb Bridge</td>
      <td>2841.0</td>
    </tr>
    <tr>
      <th>25544</th>
      <td>2016-12-29</td>
      <td>Webb Bridge</td>
      <td>2540.0</td>
    </tr>
    <tr>
      <th>25554</th>
      <td>2016-12-30</td>
      <td>Webb Bridge</td>
      <td>3124.0</td>
    </tr>
    <tr>
      <th>25564</th>
      <td>2016-12-31</td>
      <td>Webb Bridge</td>
      <td>4271.0</td>
    </tr>
  </tbody>
</table>
<p>25570 rows × 3 columns</p>
</div>



<a id="evaluation-criteria"> </a>

### Evaluation criteria


To compare the performance of our modelling strategies, we perform 5-fold cross-validation using the latest year in the data (2016) for validation, with validation horizons of 28 days. `SMAPE` will be the evaluation metric for the out-of-sample forecasts.


```python
# Python
train_start = datetime(2010, 1, 1)
val_horizon_days = 28
num_vals = 5
val_starts = sorted([datetime(2016, 12, 31) - timedelta(days=val_horizon_days) * i for i in range(1, num_vals + 1)])
```
```python
# Python
def calculate_smape(y: np.array, yhat: np.array) -> float:
    abs_error = np.abs(y - yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    smape = np.nanmean(abs_error / denom)
    return smape
```
<a id="prophet-configuration"> </a>

### Prophet configuration



Here we define our Prophet settings. Since the purpose of the exercise is to evaluate the top-level-regressor modelling strategy, we won't change the default values too much. There are a few things we know about the time series that we want to capture though:



* There is day-of-week and month-of-year seasonality in foot traffic (more people in work-related areas on weekdays vs. leisure-related areas on weekends, less people out in the colder months of the year).

* Seasonality is likely multiplicative rather than additive, given the population growth in Melbourne.

* We want to use more of the recent data for estimating the trend, given the population growth in Melbourne.



We'll also be looking at how the different modelling strategies affect the confidence of our predictions, so we modify the `interval_width` to capture a 90% credible interval, and increase the number of uncertainty samples to 5,000 to get more stable lower and upper bound estimates.



For the cross-validation procedure, we define a function that fits up to a certain date in the dataframe history, then predicts on the validation period dates.


```python
# Python
model = Prophet(
    changepoint_range=0.95,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    interval_width=0.90,
    uncertainty_samples=5000,
)
```
```python
# Python
def train_validate(df: pd.DataFrame, val_start: datetime, val_horizon_days: timedelta, model: Prophet) -> pd.DataFrame:
    val_end = val_start + timedelta(days=val_horizon_days)
    train = df.loc[df['ds'] < val_start]
    val = df.loc[(df['ds'] >= val_start) & (df['ds'] < val_end)]
    model = model.fit(train)
    oos_preds = model.predict(val)
    oos_preds[['yhat_lower', 'yhat', 'yhat_upper']] = oos_preds[['yhat_lower', 'yhat', 'yhat_upper']].clip(lower=0)
    oos_preds['y'] = val['y'].values
    return oos_preds
```
<a id="strategy-1:-independent-models"> </a>

## Strategy 1: Independent Models



This will be our simple baseline - modelling each segment using only the data in the segment. For each segment, we output the out-of-sample prediction dataframe for each cross-validation fold.


```python
# Python
oos_independent = {s: {} for s in segments}
for val_start in val_starts:
    for segment, df_segment in df.groupby('segment'):
        segment_model = deepcopy(model)
        oos_independent[segment][str(val_start)] = train_validate(df_segment, val_start, val_horizon_days, segment_model)
```
<a id="strategy-2:-top-level-regressor"> </a>

## Strategy 2: Top-level regressor



This is the strategy we're interested in exploring in this tutorial. The strategy is outlined [here](https://github.com/facebook/prophet/issues/49#issuecomment-430383681) by Ben Letham. It involves treating the groups of time series as a hierarchy, with a "top-level" time series that combines them.



The approach involves:



1. Forecasting the top-level series, including any future values.



Note that for cross-validation, we need to use the **forecasted** top-level series values, rather than the true values. This is to prevent information leakage, as the top-level series values would not be known when producing a real forecast.



For the top-level series model, we simply use the same Prophet configurations described in the beginning. This is reasonable, as we would expect the top-level time series to have the same characteristics its constituent segments.



2. Using the historical data from the top-level series, and the future prediction values, as an additional regressor for the time series of each segment.



The Prophet model becomes:



$y_t = (1 + \mathrm{seasonality}_t) * \mathrm{trend}_t + \beta * \mathrm{top level}_t$



Having the top-level series in the model effectively includes information about the growth and seasonality of other segments, which we hypothesize will help produce more accurate segment-level forecasts.



We configure the top-level regressor in the Prophet model as follows:



* The top-level regressor values are standardized.

* We assume the relationship is additive. This is simpler to interpret.

* We use a `prior_scale` of 1.0. This controls the level of regularization on $\beta$; in practice, it represents how much we weight the data and trends in the target segment vs. the data and trends in other segments. For example, if we were confident that a particular segment's trajectory is better determined by its own data (because its growth drivers are very different to that of other segments), we could set a lower prior scale to pull $\beta$ closer to 0.


```python
# Python
df_toplevel = df.groupby('ds', as_index=False).agg({'y': 'sum'})
```
```python
# Python
regressor_mode = 'additive'
prior_scale = 1.0
oos_toplevelregr = {s: {} for s in segments}
for val_start in val_starts:
    toplevel_model = deepcopy(model)
    toplevel_history = df_toplevel.loc[df_toplevel['ds'] < val_start]
    toplevel_oos = train_validate(df_toplevel, val_start, val_horizon_days, toplevel_model)
    toplevel = pd.concat([
        toplevel_history[['ds', 'y']].rename(columns={'y': 'top_level'}),
        toplevel_oos[['ds', 'yhat']].rename(columns={'yhat': 'top_level'})
    ], ignore_index=True)
    for segment, df_segment in df.groupby('segment'):
        segment_model = deepcopy(model)
        segment_model.add_regressor('top_level', standardize=True, mode=regressor_mode, prior_scale=prior_scale)
        model_df = pd.merge(df_segment, toplevel, on='ds', how='left')
        model_df['top_level'] = model_df['top_level'].fillna(method='ffill')
        oos_toplevelregr[segment][str(val_start)] = train_validate(model_df, val_start, val_horizon_days, segment_model)
```
<a id="comparison"> </a>

## Comparison


<a id="visual-comparison-of-out-of-sample-predictions"> </a>

### Visual comparison of out-of-sample predictions


First we'll do a visual comparison of the out-of-sample actuals vs. forecasts under each strategy.


```python
# Python
def extract_predictions(
    oos_independent: Dict[str, Dict[str, pd.DataFrame]], 
    oos_toplevelregr: Dict[str, Dict[str, pd.DataFrame]], 
    segment: str
):
    preds_independent = pd.concat(oos_independent[segment], axis=0, names=['val_start']).reset_index()
    preds_toplevelregr = pd.concat(oos_toplevelregr[segment], axis=0, names=['val_start']).reset_index()
    preds = pd.merge(
        preds_independent[['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']],
        preds_toplevelregr[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='inner',
        suffixes=('_independent', '_toplevelregr')
    )
    return preds
```
```python
# Python
def plot_segment(
    oos_independent: Dict[str, Dict[str, pd.DataFrame]],
    oos_toplevelregr: Dict[str, Dict[str, pd.DataFrame]],
    segment: str,
    ax=None
):
    plot_df = extract_predictions(oos_independent, oos_toplevelregr, segment)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(plot_df['ds'], plot_df['y'], marker=".", c="black", linestyle="")
    ax.plot(plot_df['ds'], plot_df['yhat_independent'], label='yhat_independent')
    ax.fill_between(x=plot_df['ds'], y1=plot_df['yhat_lower_independent'], y2=plot_df['yhat_upper_independent'], alpha=0.2)
    ax.plot(plot_df['ds'], plot_df['yhat_toplevelregr'], label='yhat_toplevelregr')
    ax.fill_between(x=plot_df['ds'], y1=plot_df['yhat_lower_toplevelregr'], y2=plot_df['yhat_upper_toplevelregr'], alpha=0.2)
    ax.set_title(segment)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.legend()
    
    return ax
```
```python
# Python
def plot_all():
    rows, cols = 5, 2
    fig, axs = plt.subplots(
        nrows=rows, 
        ncols=cols, 
        sharex=True, 
        figsize=np.array(plt.rcParams['figure.figsize']) * np.array([cols, rows])
    )
    for segment, ax in zip(segments, axs.flat):
        plot_segment(oos_independent, oos_toplevelregr, segment, ax)
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig
```
```python
# Python
plot_all();
```
 
![png](/prophet/static/multivariate_top_level_regressor_files/multivariate_top_level_regressor_29_0.png) 


In general, the top-level-regressor strategy forecasts (`yhat`) don't differ too much compared to the independent models strategy. It does, however, seem to provide _less extreme forecasts_, which can be either beneficial or harmful:



* In the `Bourke Street Mall (North)` segment, the independent model captures an increased growth starting in November, and slightly overforecasts the future. The model with the top-level-regressor does not overforecast, likely reflecting the lower "average" growth across all the segments.

* The `Southern Cross Station` and `Flagstaff Station` segments exhibit strong negative weekend effects, which are **not** picked up well by the top-level-regressor strategy. The top-level-regressor strategy overpredicts weekends, likely because the seasonality effect is not as strong in the other segments.

* The `Waterfront City` and `New Quay` segments are not forecasted by either strategy, due to their low volume and fairly irregular pattern. The top-level-regressor strategy produces forecasts with weaker seasonality effects, which seems to be beneficial for `New Quay` but not `Waterfront City`.



Given the mixed performance of the top-level-regressor strategy, we should keep in mind the following when using it for segment-level forecasts:



1. Performing cross-validation to compare the top-level-regressor strategy to the independent model strategy is important. The top-level-regressor strategy is most useful for pulling the forecast of the target segment back to an "average level" of growth, i.e. producing less extreme forecasts. We shouldn't expect it to automatically provide more accurate forecasts.

1. Choice of `prior_scale` is important and should be chosen explicitly for each segment being forecasted. This controls the weight the model places on the data and trends in the other segments. For example, a lower `prior_scale` places less weight on the other segments, which makes sense when the target segment has unique underlying growth drivers.

1. Adding the top-level series as a regressor can lead to less accurate seasonality fits. To prevent this, we could choose segments that exhibit similar seasonal patterns to include in the top-level series, and exclude others.


<a id="`smape`-comparisons"> </a>

### `SMAPE` comparisons



We've seen above that the top-level-regressor strategy gives mixed results depending on the segment being forecasted. We now summarise the `SMAPE` values for the out-of-sample forecasts for each segment. Note that a lower `SMAPE` implies a more accurate forecast.


```python
# Python
def summarise_smape(
    oos_independent: Dict[str, Dict[str, pd.DataFrame]], 
    oos_toplevelregr: Dict[str, Dict[str, pd.DataFrame]], 
):
    smape_summ = []
    for segment in segments:
        preds = extract_predictions(oos_independent, oos_toplevelregr, segment)
        smape_i = calculate_smape(preds['y'], preds['yhat_independent'])
        smape_tlr = calculate_smape(preds['y'], preds['yhat_toplevelregr'])
        smape_summ.append(pd.DataFrame([[segment, smape_i, smape_tlr]], columns=['segment', 'smape_independent', 'smape_toplevelregr'], index=[0]))
    smape_summ = pd.concat(smape_summ, ignore_index=True).set_index('segment')
    smape_summ['diff_toplevelregr'] = smape_summ['smape_toplevelregr'] - smape_summ['smape_independent']
    return smape_summ
```
```python
# Python
smape_summary = summarise_smape(oos_independent, oos_toplevelregr)
```
```python
# Python
smape_summary.style.format("{:.2%}")
```



<style  type="text/css" >
</style><table id="T_06e2d_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >smape_independent</th>        <th class="col_heading level0 col1" >smape_toplevelregr</th>        <th class="col_heading level0 col2" >diff_toplevelregr</th>    </tr>    <tr>        <th class="index_name level0" >segment</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_06e2d_level0_row0" class="row_heading level0 row0" >Bourke Street Mall (North)</th>
                        <td id="T_06e2d_row0_col0" class="data row0 col0" >11.05%</td>
                        <td id="T_06e2d_row0_col1" class="data row0 col1" >11.32%</td>
                        <td id="T_06e2d_row0_col2" class="data row0 col2" >0.27%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row1" class="row_heading level0 row1" >Bourke Street Mall (South)</th>
                        <td id="T_06e2d_row1_col0" class="data row1 col0" >8.91%</td>
                        <td id="T_06e2d_row1_col1" class="data row1 col1" >10.30%</td>
                        <td id="T_06e2d_row1_col2" class="data row1 col2" >1.39%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row2" class="row_heading level0 row2" >Princes Bridge</th>
                        <td id="T_06e2d_row2_col0" class="data row2 col0" >9.84%</td>
                        <td id="T_06e2d_row2_col1" class="data row2 col1" >9.86%</td>
                        <td id="T_06e2d_row2_col2" class="data row2 col2" >0.02%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row3" class="row_heading level0 row3" >Flinders Street Station Underpass</th>
                        <td id="T_06e2d_row3_col0" class="data row3 col0" >9.15%</td>
                        <td id="T_06e2d_row3_col1" class="data row3 col1" >8.54%</td>
                        <td id="T_06e2d_row3_col2" class="data row3 col2" >-0.62%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row4" class="row_heading level0 row4" >Webb Bridge</th>
                        <td id="T_06e2d_row4_col0" class="data row4 col0" >13.32%</td>
                        <td id="T_06e2d_row4_col1" class="data row4 col1" >13.02%</td>
                        <td id="T_06e2d_row4_col2" class="data row4 col2" >-0.30%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row5" class="row_heading level0 row5" >Southern Cross Station</th>
                        <td id="T_06e2d_row5_col0" class="data row5 col0" >26.34%</td>
                        <td id="T_06e2d_row5_col1" class="data row5 col1" >30.36%</td>
                        <td id="T_06e2d_row5_col2" class="data row5 col2" >4.01%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row6" class="row_heading level0 row6" >Waterfront City</th>
                        <td id="T_06e2d_row6_col0" class="data row6 col0" >33.79%</td>
                        <td id="T_06e2d_row6_col1" class="data row6 col1" >37.13%</td>
                        <td id="T_06e2d_row6_col2" class="data row6 col2" >3.34%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row7" class="row_heading level0 row7" >New Quay</th>
                        <td id="T_06e2d_row7_col0" class="data row7 col0" >24.67%</td>
                        <td id="T_06e2d_row7_col1" class="data row7 col1" >23.45%</td>
                        <td id="T_06e2d_row7_col2" class="data row7 col2" >-1.22%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row8" class="row_heading level0 row8" >Flagstaff Station</th>
                        <td id="T_06e2d_row8_col0" class="data row8 col0" >28.09%</td>
                        <td id="T_06e2d_row8_col1" class="data row8 col1" >20.35%</td>
                        <td id="T_06e2d_row8_col2" class="data row8 col2" >-7.73%</td>
            </tr>
            <tr>
                        <th id="T_06e2d_level0_row9" class="row_heading level0 row9" >State Library</th>
                        <td id="T_06e2d_row9_col0" class="data row9 col0" >8.04%</td>
                        <td id="T_06e2d_row9_col1" class="data row9 col1" >9.06%</td>
                        <td id="T_06e2d_row9_col2" class="data row9 col2" >1.02%</td>
            </tr>
    </tbody></table>



The top-level-regressor strategy produces:



* In **2 out of 10** segments, an SMAPE _improvement_ of >1% absolute.

* In **4 out of 10** segments forecasted, an SMAPE _deterioration_ of >1% absolute.

* In **4 out of 10** segments, no material change in SMAPE (between -1% and 1% absolute impact).


<a id="uncertainty-interval-comparisons-"> </a>

### Uncertainty interval comparisons 



One thing we haven't touched on is the uncertainty intervals produced with the top-level-regressor strategy. Notice that the intervals are generally tighter compared to that of the independent models, but this is **not** representative of the actual uncertainty because:



* We haven't performed MCMC sampling for the distribution of the $\beta$ coefficient. We assume it takes the mean the value for all future predictions, but the ideally we account for the range of possible values $\beta$ could take.

* We haven't factored in the uncertainty in the forecast of the top-level series. We use the `yhat` of the top-level series in our target segment prediction, which doesn't capture the potential variation in the top-level series forecast.



For this reason, the tighter intervals may not accurately capture the range of possible outcomes. We check this below by calculating the size of the 90% uncertainty interval, and the proportion of actual `y` values that lie within the interval, across the two strategies.


```python
# Python
def summarise_uncertainty(
    oos_independent: Dict[str, Dict[str, pd.DataFrame]], 
    oos_toplevelregr: Dict[str, Dict[str, pd.DataFrame]], 
):
    uncertainty = []
    for segment in segments:
        preds = extract_predictions(oos_independent, oos_toplevelregr, segment)
        interval_size_i = np.mean(preds['yhat_upper_independent'] - preds['yhat_lower_independent'])
        interval_size_tlr = np.mean(preds['yhat_upper_toplevelregr'] - preds['yhat_lower_toplevelregr'])
        perc_within_i = np.mean((preds['y'] > preds['yhat_lower_independent']) & (preds['y'] < preds['yhat_upper_independent']))
        perc_within_tlr = np.mean((preds['y'] > preds['yhat_lower_toplevelregr']) & (preds['y'] < preds['yhat_upper_toplevelregr']))
        uncertainty.append(
            pd.DataFrame(
                [[segment, interval_size_i, interval_size_tlr, perc_within_i, perc_within_tlr]], 
                columns=['segment', 'interval_size_independent', 'interval_size_toplevelregr', 'perc_within_independent', 'perc_within_toplevelregr'], 
                index=[0]
            )
        )
    uncertainty = pd.concat(uncertainty, ignore_index=True).set_index('segment')
    uncertainty['interval_size_diff'] = uncertainty['interval_size_toplevelregr'] - uncertainty['interval_size_independent']
    uncertainty['perc_within_diff'] = uncertainty['perc_within_toplevelregr'] - uncertainty['perc_within_independent']
    return uncertainty
```
```python
# Python
uncertainty = summarise_uncertainty(oos_independent, oos_toplevelregr)
```
```python
# Python
uncertainty[[c for c in uncertainty.columns if c.startswith('interval_size')]].style.format("{:,.0f}")
```



<style  type="text/css" >
</style><table id="T_da0ac_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >interval_size_independent</th>        <th class="col_heading level0 col1" >interval_size_toplevelregr</th>        <th class="col_heading level0 col2" >interval_size_diff</th>    </tr>    <tr>        <th class="index_name level0" >segment</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_da0ac_level0_row0" class="row_heading level0 row0" >Bourke Street Mall (North)</th>
                        <td id="T_da0ac_row0_col0" class="data row0 col0" >11,554</td>
                        <td id="T_da0ac_row0_col1" class="data row0 col1" >8,237</td>
                        <td id="T_da0ac_row0_col2" class="data row0 col2" >-3,317</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row1" class="row_heading level0 row1" >Bourke Street Mall (South)</th>
                        <td id="T_da0ac_row1_col0" class="data row1 col0" >11,806</td>
                        <td id="T_da0ac_row1_col1" class="data row1 col1" >9,712</td>
                        <td id="T_da0ac_row1_col2" class="data row1 col2" >-2,094</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row2" class="row_heading level0 row2" >Princes Bridge</th>
                        <td id="T_da0ac_row2_col0" class="data row2 col0" >11,990</td>
                        <td id="T_da0ac_row2_col1" class="data row2 col1" >10,452</td>
                        <td id="T_da0ac_row2_col2" class="data row2 col2" >-1,538</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row3" class="row_heading level0 row3" >Flinders Street Station Underpass</th>
                        <td id="T_da0ac_row3_col0" class="data row3 col0" >9,809</td>
                        <td id="T_da0ac_row3_col1" class="data row3 col1" >6,759</td>
                        <td id="T_da0ac_row3_col2" class="data row3 col2" >-3,050</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row4" class="row_heading level0 row4" >Webb Bridge</th>
                        <td id="T_da0ac_row4_col0" class="data row4 col0" >1,852</td>
                        <td id="T_da0ac_row4_col1" class="data row4 col1" >1,423</td>
                        <td id="T_da0ac_row4_col2" class="data row4 col2" >-429</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row5" class="row_heading level0 row5" >Southern Cross Station</th>
                        <td id="T_da0ac_row5_col0" class="data row5 col0" >6,745</td>
                        <td id="T_da0ac_row5_col1" class="data row5 col1" >6,341</td>
                        <td id="T_da0ac_row5_col2" class="data row5 col2" >-404</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row6" class="row_heading level0 row6" >Waterfront City</th>
                        <td id="T_da0ac_row6_col0" class="data row6 col0" >5,633</td>
                        <td id="T_da0ac_row6_col1" class="data row6 col1" >5,834</td>
                        <td id="T_da0ac_row6_col2" class="data row6 col2" >202</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row7" class="row_heading level0 row7" >New Quay</th>
                        <td id="T_da0ac_row7_col0" class="data row7 col0" >7,613</td>
                        <td id="T_da0ac_row7_col1" class="data row7 col1" >7,187</td>
                        <td id="T_da0ac_row7_col2" class="data row7 col2" >-426</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row8" class="row_heading level0 row8" >Flagstaff Station</th>
                        <td id="T_da0ac_row8_col0" class="data row8 col0" >11,781</td>
                        <td id="T_da0ac_row8_col1" class="data row8 col1" >11,338</td>
                        <td id="T_da0ac_row8_col2" class="data row8 col2" >-443</td>
            </tr>
            <tr>
                        <th id="T_da0ac_level0_row9" class="row_heading level0 row9" >State Library</th>
                        <td id="T_da0ac_row9_col0" class="data row9 col0" >7,191</td>
                        <td id="T_da0ac_row9_col1" class="data row9 col1" >5,023</td>
                        <td id="T_da0ac_row9_col2" class="data row9 col2" >-2,168</td>
            </tr>
    </tbody></table>



```python
# Python
uncertainty[[c for c in uncertainty.columns if c.startswith('perc_within')]].style.format("{:.2%}")
```



<style  type="text/css" >
</style><table id="T_d24f4_" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >perc_within_independent</th>        <th class="col_heading level0 col1" >perc_within_toplevelregr</th>        <th class="col_heading level0 col2" >perc_within_diff</th>    </tr>    <tr>        <th class="index_name level0" >segment</th>        <th class="blank" ></th>        <th class="blank" ></th>        <th class="blank" ></th>    </tr></thead><tbody>
                <tr>
                        <th id="T_d24f4_level0_row0" class="row_heading level0 row0" >Bourke Street Mall (North)</th>
                        <td id="T_d24f4_row0_col0" class="data row0 col0" >75.71%</td>
                        <td id="T_d24f4_row0_col1" class="data row0 col1" >57.86%</td>
                        <td id="T_d24f4_row0_col2" class="data row0 col2" >-17.86%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row1" class="row_heading level0 row1" >Bourke Street Mall (South)</th>
                        <td id="T_d24f4_row1_col0" class="data row1 col0" >94.29%</td>
                        <td id="T_d24f4_row1_col1" class="data row1 col1" >85.00%</td>
                        <td id="T_d24f4_row1_col2" class="data row1 col2" >-9.29%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row2" class="row_heading level0 row2" >Princes Bridge</th>
                        <td id="T_d24f4_row2_col0" class="data row2 col0" >87.86%</td>
                        <td id="T_d24f4_row2_col1" class="data row2 col1" >82.14%</td>
                        <td id="T_d24f4_row2_col2" class="data row2 col2" >-5.71%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row3" class="row_heading level0 row3" >Flinders Street Station Underpass</th>
                        <td id="T_d24f4_row3_col0" class="data row3 col0" >76.43%</td>
                        <td id="T_d24f4_row3_col1" class="data row3 col1" >65.71%</td>
                        <td id="T_d24f4_row3_col2" class="data row3 col2" >-10.71%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row4" class="row_heading level0 row4" >Webb Bridge</th>
                        <td id="T_d24f4_row4_col0" class="data row4 col0" >90.00%</td>
                        <td id="T_d24f4_row4_col1" class="data row4 col1" >83.57%</td>
                        <td id="T_d24f4_row4_col2" class="data row4 col2" >-6.43%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row5" class="row_heading level0 row5" >Southern Cross Station</th>
                        <td id="T_d24f4_row5_col0" class="data row5 col0" >85.71%</td>
                        <td id="T_d24f4_row5_col1" class="data row5 col1" >87.14%</td>
                        <td id="T_d24f4_row5_col2" class="data row5 col2" >1.43%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row6" class="row_heading level0 row6" >Waterfront City</th>
                        <td id="T_d24f4_row6_col0" class="data row6 col0" >85.00%</td>
                        <td id="T_d24f4_row6_col1" class="data row6 col1" >85.00%</td>
                        <td id="T_d24f4_row6_col2" class="data row6 col2" >0.00%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row7" class="row_heading level0 row7" >New Quay</th>
                        <td id="T_d24f4_row7_col0" class="data row7 col0" >82.86%</td>
                        <td id="T_d24f4_row7_col1" class="data row7 col1" >84.29%</td>
                        <td id="T_d24f4_row7_col2" class="data row7 col2" >1.43%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row8" class="row_heading level0 row8" >Flagstaff Station</th>
                        <td id="T_d24f4_row8_col0" class="data row8 col0" >90.71%</td>
                        <td id="T_d24f4_row8_col1" class="data row8 col1" >91.43%</td>
                        <td id="T_d24f4_row8_col2" class="data row8 col2" >0.71%</td>
            </tr>
            <tr>
                        <th id="T_d24f4_level0_row9" class="row_heading level0 row9" >State Library</th>
                        <td id="T_d24f4_row9_col0" class="data row9 col0" >87.86%</td>
                        <td id="T_d24f4_row9_col1" class="data row9 col1" >73.57%</td>
                        <td id="T_d24f4_row9_col2" class="data row9 col2" >-14.29%</td>
            </tr>
    </tbody></table>



* We can see that in the segments where the size of the uncertainty interval reduced the most, we also saw a much lower proportion of `y` values (actual traffic counts) fall within the uncertainty interval. 

* There are some instances (4 out of 10 segments) where the proportion of `y` values within the interval slightly improves when using the top-level regressor strategy, but in these instances the interval size is roughly the same as under the independent-model strategy.



These results reinforce that if we want reliable uncertainty estimates when using the top-level-regressor strategy, we need to perform full MCMC sampling for $\beta$ and account for the uncertainty of the top-level forecast.


<a id="further-reading"> </a>

## Further reading



We've walked through one strategy for producing segment-level forecasts with Prophet. Other modelling strategies for this use case include:



* Estimating the model parameters for each segment simultaeneously using partial pooling, as described [here](https://github.com/facebook/prophet/issues/49#issuecomment-283254534).

* [Vector Autoregressive (VAR) models](https://otexts.com/fpp3/VAR.html).

* [Forecast reconciliation for hierarchical forecasts](https://otexts.com/fpp3/hierarchical.html).



As with the top-level-regressor strategy, these strategies make assumptions about the data generation process and the relationship between the different segments, and it's recommended to outline a validation scheme (e.g. cross-validation) to assess their suitability for the task at hand.


```python
# Python

```