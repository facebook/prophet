import pandas as pd
import pytest
from prophet import Prophet

def test_in_range_ok():
    df = pd.DataFrame({"ds": ["2024-01-01","2024-01-02"], "y": [1.0, 2.0]})
    Prophet().fit(df)  # no error

def test_out_of_range_raises():
    df = pd.DataFrame({"ds": ["3969-12-02","3969-12-03"], "y": [1.0, 2.0]})
    with pytest.raises(ValueError) as e:
        Prophet().fit(df)
    assert "datetime64[ns]" in str(e.value)
