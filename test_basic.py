import pandas as pd
from contextlens.analysis import profile_dataset  # adjust to your actual function name

def test_profile_runs_on_simple_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = profile_dataset(df)
    assert result is not None
