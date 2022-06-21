# Python
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from importlib import reload
import prophet


# Python
df_in = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv')
df_in.head()


# Python
reload(prophet)
def run_prophet_speedtest(periods=30, include_history=True, holiday=True):
    m = prophet.Prophet(uncertainty_samples=1000)
    if holiday:
        m.add_country_holidays(country_name='US')
    m.fit(df_in)

    # Python
    future = m.make_future_dataframe(periods=periods, include_history=include_history)
    forecast = m.predict(future)
    return m, forecast
%timeit run_prophet_speedtest()

m, forecast = run_prophet_speedtest(periods=100, include_history=False)
future = m.make_future_dataframe(periods=1)
sample1 = m.sample_posterior_predictive(m.setup_dataframe(future))
print(forecast.set_index("ds")[[x for x in forecast.columns if "yhat" in x]].tail())
fig1 = m.plot(forecast)
plt.savefig("MainProphet.png", dpi=300)


DATA = pd.read_csv(
    os.path.join('C:/Users/Colin/Documents/prophet/python/prophet/tests', 'data.csv'),
    parse_dates=['ds'],
)
prophet_obj = Prophet(growth='logistic')
N = DATA.shape[0]
history = DATA.head(N // 2).copy()
history['floor'] = 10.
history['cap'] = 80.
future = DATA.tail(N // 2).copy()
future['cap'] = 80.
future['floor'] = 10.
prophet_obj.fit(history, algorithm='Newton')
if not prophet_obj.logistic_floor:
    raise Exception
if 'floor' not in prophet_obj.history:
    raise Exception
if round(prophet_obj.history['y_scaled'][0], 1) != 1.0:
    raise Exception
if prophet_obj.fit_kwargs != {'algorithm': 'Newton'}:
    raise Exception
fcst1 = prophet_obj.predict(future)

m2 = Prophet(growth='logistic')
history2 = history.copy()
history2['y'] += 10.
history2['floor'] += 10.
history2['cap'] += 10.
future['cap'] += 10.
future['floor'] += 10.
m2.fit(history2, algorithm='Newton')
if round(m2.history['y_scaled'][0], 1) != 1.0:
    raise Exception


m = Prophet()
m.add_regressor('binary_feature', prior_scale=0.2)
m.add_regressor('numeric_feature', prior_scale=0.5)
m.add_regressor(
    'numeric_feature2', prior_scale=0.5, mode='multiplicative'
)
m.add_regressor('binary_feature2', standardize=True)
df = DATA.copy()
df['binary_feature'] = ['0'] * 255 + ['1'] * 255
df['numeric_feature'] = range(510)
df['numeric_feature2'] = range(510)
df['binary_feature2'] = [1] * 100 + [0] * 410
m.fit(df)
# Check that standardizations are correctly set
if m.extra_regressors['binary_feature'] != {
    'prior_scale': 0.2,
    'mu': 0,
    'std': 1,
    'standardize': 'auto',
    'mode': 'additive',
}:
    raise Exception
if m.extra_regressors['numeric_feature']['prior_scale'] != 0.5:
    raise Exception
if m.extra_regressors['numeric_feature']['mu'] != 254.5:
    raise Exception
if round(m.extra_regressors['numeric_feature']['std'], 4) != 147.3686:
    print(round(m.extra_regressors['numeric_feature']['std'], 6))
    print(147.368585)
    raise Exception
if m.extra_regressors['numeric_feature2']['mode'] != 'multiplicative':
    raise Exception
if m.extra_regressors['binary_feature2']['prior_scale'] != 10.0:
    raise Exception
if round(m.extra_regressors['binary_feature2']['mu'], 6) != 0.196078:
    print(round(m.extra_regressors['binary_feature2']['mu'], 6))
    print(0.1960784)
    raise Exception
if round(m.extra_regressors['binary_feature2']['std'], 6) != 0.397418:
    raise Exception
# Check that standardization is done correctly
df2 = m.setup_dataframe(df.copy())
if df2['binary_feature'][0] != 0:
    raise Exception
if round(df2['numeric_feature'][0], 5) != -1.72696:
    raise Exception
if round(df2['binary_feature2'][0], 5) != 2.02286:
    print(round(df2['binary_feature2'][0], 6))
    print(2.022859)
    raise Exception
# Check that feature matrix and prior scales are correctly constructed
seasonal_features, prior_scales, component_cols, modes = (
    m.make_all_seasonality_features(df2)
)
if seasonal_features.shape[1] != 30:
    raise Exception
names = ['binary_feature', 'numeric_feature', 'binary_feature2']
true_priors = [0.2, 0.5, 10.]
for i, name in enumerate(names):
    if name not in seasonal_features:
        raise Exception
    if sum(component_cols[name]) != 1:
        raise Exception
    if sum(np.array(prior_scales) * component_cols[name]) != true_priors[i]:
        raise Exception
# Check that forecast components are reasonable
future = pd.DataFrame({
    'ds': ['2014-06-01'],
    'binary_feature': [0],
    'numeric_feature': [10],
    'numeric_feature2': [10],
})
future['binary_feature2'] = 0
fcst = m.predict(future)
print(fcst.columns)
print(fcst.shape)
if fcst.shape[1] != 37:
    raise Exception
if fcst['binary_feature'][0] != 0:
    raise Exception
(
    fcst['extra_regressors_additive'][0],
    fcst['numeric_feature'][0] + fcst['binary_feature2'][0],
)
(
    fcst['extra_regressors_multiplicative'][0],
    fcst['numeric_feature2'][0],
)
(
    fcst['additive_terms'][0],
    fcst['yearly'][0] + fcst['weekly'][0]
    + fcst['extra_regressors_additive'][0],
)
(
    fcst['multiplicative_terms'][0],
    fcst['extra_regressors_multiplicative'][0],
)
(
    fcst['yhat'][0],
    fcst['trend'][0] * (1 + fcst['multiplicative_terms'][0])
    + fcst['additive_terms'][0],
)
# Check works if constant extra regressor at 0
df['constant_feature'] = 0
m = Prophet()
m.add_regressor('constant_feature')
m.fit(df)
if m.extra_regressors['constant_feature']['std'] != 1:
    raise Exception


from prophet import diagnostics

# Next thing
params = (x for x in ['logistic', 'flat'])
for growth in params:
    df = DATA.head(100)
    if growth == "logistic":
        df['cap'] = 40
    m = Prophet(growth=growth).fit(df)
    df_cv = diagnostics.cross_validation(
        m, horizon='1 days', period='1 days', initial='140 days'
    )
    if len(np.unique(df_cv['cutoff'])) != 2:
        print(len(np.unique(df_cv['cutoff'])))
        raise Exception(f"{growth}")
    if not (df_cv['cutoff'] < df_cv['ds']).all():
        raise Exception(f"{growth}")
    df_merged = pd.merge(df_cv, DATA, 'left', on='ds')
    if not round(np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 2) == 0.0:
        raise Exception(f"{growth}")


# Cross Validation Test
class CustomParallelBackend:
    def map(self, func, *iterables):
        results = [func(*args) for args in zip(*iterables)]
        return results


m = Prophet()
m.fit(DATA.head(100))
# Calculate the number of cutoff points(k)
horizon = pd.Timedelta('4 days')
period = pd.Timedelta('10 days')
initial = pd.Timedelta('115 days')
methods = [None, 'processes', 'threads', CustomParallelBackend()]

try:
    from dask.distributed import Client
    client = Client(processes=False)  # noqa
    methods.append("dask")
except ImportError:
    pass

for parallel in methods:
    print(f"starting: {parallel}")
    df_cv = diagnostics.cross_validation(
        m, horizon='4 days', period='10 days', initial='115 days',
        parallel=parallel)
    print(len(np.unique(df_cv['cutoff'])))
    time.sleep(5)
    # if len(np.unique(df_cv['cutoff'])) != 3:
    #     raise Exception(f"{parallel}")
    if max(df_cv['ds'] - df_cv['cutoff']) != horizon:
        raise Exception(f"{parallel}")
    if not min(df_cv['cutoff']) >= min(DATA['ds']) + initial:
        raise Exception(f"{parallel}")
    dc = df_cv['cutoff'].diff()
    dc = dc[dc > pd.Timedelta(0)].min()
    if not dc >= period:
        raise Exception(f"{parallel}")
    if not (df_cv['cutoff'] < df_cv['ds']).all():
        raise Exception(f"{parallel}")
    # Each y in df_cv and self.__df with same ds should be equal
    df_merged = pd.merge(df_cv, DATA, 'left', on='ds')
    if round(np.sum((df_merged['y_x'] - df_merged['y_y']) ** 2), 2) != 0.0:
        raise Exception(f"{parallel}")
    df_cv = diagnostics.cross_validation(
        m, horizon='4 days', period='10 days', initial='135 days')
    if len(np.unique(df_cv['cutoff'])) != 1:
        raise Exception
