---
layout: docs
docid: "additional_topics"
title: "Additional Topics"
permalink: /docs/additional_topics.html
subsections:
  - title: Saving models
    id: saving-models
  - title: Flat trend and custom trends
    id: flat-trend-and-custom-trends
  - title: Updating fitted models
    id: updating-fitted-models
  - title: External references
    id: external-references
---
<a id="saving-models"> </a>

### Saving models



It is possible to save fitted Prophet models so that they can be loaded and used later.



In R, this is done with `saveRDS` and `readRDS`:


```R
# R
saveRDS(m, file="model.RDS")  # Save model
m <- readRDS(file="model.RDS")  # Load model
```
In Python, models should not be saved with pickle; the Stan backend attached to the model object will not pickle well, and will produce issues under certain versions of Python. Instead, you should use the built-in serialization functions to serialize the model to json:


```python
# Python
import json
from prophet.serialize import model_to_json, model_from_json

with open('serialized_model.json', 'w') as fout:
    json.dump(model_to_json(m), fout)  # Save model

with open('serialized_model.json', 'r') as fin:
    m = model_from_json(json.load(fin))  # Load model
```
The json file will be portable across systems, and deserialization is backwards compatible with older versions of prophet.


<a id="flat-trend-and-custom-trends"> </a>

### Flat trend and custom trends



For time series that exhibit strong seasonality patterns rather than trend changes, it may be useful to force the trend growth rate to be flat. This can be achieved simply by passing `growth=flat` when creating the model:


```R
# R
m <- prophet(df, growth='flat')
```
```python
# Python
m = Prophet(growth='flat')
```
Note that if this is used on a time series that doesn't have a constant trend, any trend will be fit with the noise term and so there will be high predictive uncertainty in the forecast.



To use a trend besides these three built-in trend functions (piecewise linear, piecewise logistic growth, and flat), you can download the source code from github, modify the trend function as desired in a local branch, and then install that local version. [This PR](https://github.com/facebook/prophet/pull/1466/files) provides a good illustration of what must be done to implement a custom trend, as does [this one](https://github.com/facebook/prophet/pull/1794) that implements a step function trend and [this one](https://github.com/facebook/prophet/pull/1778) for a new trend in R.


<a id="updating-fitted-models"> </a>

### Updating fitted models



A common setting for forecasting is fitting models that need to be updated as additional data come in. Prophet models can only be fit once, and a new model must be re-fit when new data become available. In most settings, model fitting is fast enough that there isn't any issue with re-fitting from scratch. However, it is possible to speed things up a little by warm-starting the fit from the model parameters of the earlier model. This code example shows how this can be done in Python:


```python
# Python
def stan_init(m):
    """Retrieve parameters from a trained model.
    
    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.
    
    Parameters
    ----------
    m: A trained model of the Prophet class.
    
    Returns
    -------
    A Dictionary containing retrieved parameters of m.
    
    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

df = pd.read_csv('../examples/example_wp_log_peyton_manning.csv')
df1 = df.loc[df['ds'] < '2016-01-19', :]  # All data except the last day
m1 = Prophet().fit(df1) # A model fit to all data except the last day


%timeit m2 = Prophet().fit(df)  # Adding the last day, fitting from scratch
%timeit m2 = Prophet().fit(df, init=stan_init(m1))  # Adding the last day, warm-starting from m1
```
    1.33 s ± 55.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    185 ms ± 4.46 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)


As can be seen, the parameters from the previous model are passed in to the fitting for the next with the kwarg `init`. In this case, model fitting was about 5x faster when using warm starting. The speedup will generally depend on how much the optimal model parameters have changed with the addition of the new data.



There are few caveats that should be kept in mind when considering warm-starting. First, warm-starting may work well for small updates to the data (like the addition of one day in the example above) but can be worse than fitting from scratch if there are large changes to the data (i.e., a lot of days have been added). This is because when a large amount of history is added, the location of the changepoints will be very different between the two models, and so the parameters from the previous model may actually produce a bad trend initialization. Second, as a detail, the number of changepoints need to be consistent from one model to the next or else an error will be raised because the changepoint prior parameter `delta` will be the wrong size.


<a id="external-references"> </a>

### External references

These github repositories provide examples of building on top of Prophet in ways that may be of broad interest:

* [forecastr](https://github.com/garethcull/forecastr): A web app that provides a UI for Prophet.

* [NeuralProphet](https://github.com/ourownstory/neural_prophet): A Prophet-style model implemented in pytorch, to be more adaptable and extensible.

