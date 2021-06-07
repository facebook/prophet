# Holidays Extension

This repo forks the [Facebook Prophet](https://github.com/facebook/prophet)
library, and serve as an extension of the [Python holidays](https://github.com/dr-prodigy/python-holidays)
library. The holidays are customized and extended.

To use this library, simply do

```python
from holidays_ext.get_holidays import get_holiday
from holidays_ext.get_holidays import get_holiday_df

holidays = get_holiday(
    country_list=["UnitedStates", "Russia"],
    years=[2019, 2020]
)
holidays

holidays_df = get_holiday_df(
    country_list=["UnitedStates", "China", "India"],
    years=[2018, 2019, 2020, 2021]
)
holidays_df
```
