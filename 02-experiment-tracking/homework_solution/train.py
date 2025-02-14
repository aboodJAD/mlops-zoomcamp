import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run(data_path):
    print("tracking uri =", mlflow.get_tracking_uri())
    print("experiments =", mlflow.list_experiments())
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        mlflow.set_tag("hw2", "q3")

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    args = parser.parse_args()
    mlflow.set_experiment("green-trip-jan-march-taxi-experiment")

    run(args.data_path)


# q2=4
# q3=17
# q4=default-artifact-root
# q5=6.628
# q6=6.55
