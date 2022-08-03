import pandas as pd
from datetime import datetime

from batch import prepare_data


def test_preprocessing():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    df = pd.DataFrame(data, columns=columns)
    processed_df = prepare_data(
        df.copy(deep=True), categorical=["PUlocationID", "DOlocationID"]
    )
    expected_df = {
        "PUlocationID": {0: "-1", 1: "1"},
        "DOlocationID": {0: "-1", 1: "1"},
        "pickup_datetime": {0: dt(1, 2, 0), 1: dt(1, 2, 0)},
        "dropOff_datetime": {0: dt(1, 10, 0), 1: dt(1, 10, 00)},
        "duration": {0: 8.000000000000002, 1: 8.000000000000002},
    }
    print(processed_df.shape)
    processed_df = processed_df.to_dict()
    assert processed_df == expected_df


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)
