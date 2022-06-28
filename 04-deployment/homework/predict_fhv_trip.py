import argparse
import os
import pickle

import pandas as pd


def main(args):
    features_categorical = ['PUlocationID', 'DOlocationID']
    dv, model = load_model(path='model.bin')
    df = read_data(
        f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{args.year:04d}-{args.month:02d}.parquet',
        features_categorical,)
    predictions = predict(dv, model, df[features_categorical])
    print("Predictions mean = ", predictions.mean())
    df_result = build_results_df(predictions, df, args)

    output_file = f"fhv_tripdata_{args.year:04d}-{args.month:02d}_predictions.parquet"
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def load_model(path):
    with open(path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename, features_categorical):
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[features_categorical] = df[features_categorical].fillna(
        -1).astype('int').astype('str')

    return df


def predict(dv, model, data_df):
    dicts = data_df.to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def build_results_df(predictions, df, args):
    df['ride_id'] = f'{args.year:04d}/{args.month:02d}_' + \
        df.index.astype('str')
    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["prediction"] = predictions
    return df_result


def parse_args():
    year = int(os.getenv("year"))
    month = int(os.getenv("month"))
    args = argparse.Namespace(year=year, month=month)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
