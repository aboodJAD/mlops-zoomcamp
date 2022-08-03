import os
from datetime import datetime

import pandas as pd


def dt(hour, minute, second=0):
    return datetime(2021, 1, 1, hour, minute, second)


def create_df():
    data = [
        (None, None, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, 1, dt(1, 2, 0), dt(1, 2, 50)),
        (1, 1, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ["PUlocationID", "DOlocationID", "pickup_datetime", "dropOff_datetime"]
    df_input = pd.DataFrame(data, columns=columns)
    return df_input


year = 2021
month = 2
input_file = "s3://nyc-duration/taxi_type=fhv_year={year:04d}_month={month:02d}.parquet"
output_file = (
    "s3://nyc-duration/taxi_type=fhv_year={year:04d}_month={month:02d}_results.parquet"
)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")

options = {"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
df_input = create_df()
df_input.to_parquet(
    input_file.format(year=year, month=month),
    engine="pyarrow",
    compression=None,
    index=False,
    storage_options=options,
)
os.system(
    f"export INPUT_FILE_PATTERN={input_file}; export OUTPUT_FILE_PATTERN={output_file}; python batch.py {year} {month}"
)
df_output = pd.read_parquet(
    output_file.format(year=year, month=month),
    storage_options=options,
)
print("Sum predicted durations = ", df_output["predicted_duration"].sum())
