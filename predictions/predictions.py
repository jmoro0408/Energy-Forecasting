# pylint: disable=W0105
# pylint: disable=C0103
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer  # noqa, isort: skip
from sklearn.impute import IterativeImputer  # isort: skip


class PreprocessingTransformer:
    """
    A custom transformer for X_train and X_test data,
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    # Rounding data

    def round_data(self):
        """Round the Max, Min  temperatures to 2 decimal places.
        The original values have 6 sig figs which are inaccurate, unecessary, and may slow
        down future calculations.
        """
        self.df["Max_Temp"] = self.df["Max_Temp"].round(2)
        self.df["Min_Temp"] = self.df["Min_Temp"].round(2)

    def drop_unused_cols(self):
        """Drops the 'Name' column from the dataframe.
        Each name also has a corresponding numerical PTID which will be used for identification instead.
        """
        self.df = self.df.drop(
            ["Area", "Lat", "Lon", "Grid Zone", "Lat_Lon", "temp", "index"],
            axis=1,
            errors="ignore",
        ).reset_index()  # Each name has a unique PTID

    # Cyclical Transformations
    def encode_cyclical(self):
        def encode_sin_cos(
            data: pd.DataFrame, col: str, max_val: Union[int, float]
        ) -> pd.DataFrame:
            """Create two new columns within a given dataframe to encode specified cols with sin and cos transformations

            Args:
                data (pd.DataFrame): Dataframe containing cols to encode
                col (str): column to encode (month, yeah, minute etc)
                max_val (Union[int, float]): maximum value of the given column

            Returns:
                pd.DataFrame: original dataframe with additional columns
            """
            data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
            data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
            return data

        self.df = encode_sin_cos(self.df, "Month", self.df["Month"].max())
        self.df = encode_sin_cos(self.df, "Day", self.df["Day"].max())
        self.df = encode_sin_cos(self.df, "Minute", self.df["Minute"].max())
        self.df = encode_sin_cos(self.df, "Hour", self.df["Hour"].max())

    # Handling dates
    def convert_dates_to_int(self, date_col: str = "Time Stamp"):
        """Converts a given timestamp column to integers

        Args:
            date_col (str, optional): Name of timestamp column in df. Defaults to "Time Stamp".
        """
        self.df[date_col] = self.df[date_col].astype(int)

    def convert_int_to_date(self, date_col: str = "Time Stamp"):
        """
        Converts a given ineteger column to timestamps
        Args:
            date_col (str, optional): Name of timestamp column in df. Defaults to "Time Stamp".
        """
        self.df[date_col] = pd.to_datetime(self.df[date_col])

    # Imputing
    def impute_missing_vals(self):
        """Imputes missing values using the temperature columns.
        Only missing rows in this dataset are in the min wet bulb column, therefore
        the data is only imputed using the other temperature columns.
        """
        imp = IterativeImputer(max_iter=5, random_state=0)
        df_temp = self.df[["Min_Temp", "Max_Temp", "Min Wet Bulb", "Max Wet Bulb"]]
        df_non_temp_cols = [x for x in self.df.columns.to_list() if x not in df_temp]
        df_temp = df_temp.reset_index(drop=True)
        df_non_temp = self.df[df_non_temp_cols]
        imputed = imp.fit_transform(df_temp)

        df_imputed = pd.DataFrame(imputed, columns=df_temp.columns)
        self.df = pd.concat([df_imputed, df_non_temp], axis=1)
        del df_imputed
        del df_non_temp
        assert self.df.isna().sum().sum() == 0

    # Scaling
    def scale_vals(self):
        """
        Scales all data with the standardscaler trnasformer.
        """
        transformer = StandardScaler()
        df_scaled = transformer.fit_transform(self.df)
        self.df = pd.DataFrame(df_scaled, columns=self.df.columns)
        del df_scaled


def preprocess(forecast_df: pd.DataFrame) -> pd.DataFrame:
    preprocessing = PreprocessingTransformer(forecast_df)
    preprocessing.round_data()
    preprocessing.encode_cyclical()
    preprocessing.convert_dates_to_int()
    preprocessing.drop_unused_cols()
    preprocessing.scale_vals()
    preprocessed_df = preprocessing.df
    preprocessed_df = preprocessed_df.drop("index", axis=1, errors="ignore")
    reorder_cols = [
        "Min_Temp",
        "Max_Temp",
        "Time Stamp",
        "PTID",
        "Year",
        "Month",
        "Day",
        "Minute",
        "Hour",
        "Month_sin",
        "Month_cos",
        "Day_sin",
        "Day_cos",
        "Minute_sin",
        "Minute_cos",
        "Hour_sin",
        "Hour_cos",
    ]
    preprocessed_df = preprocessed_df[reorder_cols]
    return preprocessed_df


def make_predictions():
    SAVE_MODEL_DIR = Path(Path.cwd(), "models", "saved_model")
    SAVED_FORECAST_DIR = Path(
        Path.cwd(),
        "data.nosync",
        "outputs",
        "weather_forecast",
        "forecast_2023_07_23.parquet",
    )
    model = tf.keras.models.load_model(SAVE_MODEL_DIR)
    forecast_df = pd.read_parquet(SAVED_FORECAST_DIR)
    preprocessed_df = preprocess(forecast_df)
    preds = model.predict(preprocessed_df)
    preprocessed_df["pred_load"] = preds
    return preprocessed_df


if __name__ == "__main__":
    pred_df = make_predictions()
    print(pred_df.head())
