# pylint: disable=W0105
# pylint: disable=C0103

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tomli


def get_forecast(
    api_key: str, lat_lon: tuple[float, float], forecast_days: int = 3
) -> dict:
    """Gets a forecase for a given lat lon from the https://www.weatherapi.com/ site.
    Requires an api key to be defined.

    Args:
        lat_lon (tuple[float, float]): lat/lon of the location to be forecasted
        forecast_days (int, optional): number of days to forecast, note free tier weatherapi is restricted to 14 days. Defaults to 3.
        api_key (str, optional): api key for weather api. Defaults to key.

    Raises:
        SystemExit: generic error catch for incorrect request parameters

    Returns:
        dict: json of the returned api call#
    """
    BASE_URL = "https://api.weatherapi.com/v1/forecast.json?"
    str_lat_lon = ",".join([str(x) for x in lat_lon])
    query_params = {"q": str_lat_lon, "days": forecast_days, "key": api_key}
    try:
        response = requests.get(BASE_URL, params=query_params, timeout=10)  # type: ignore
    except (
        requests.exceptions.RequestException
    ) as _error:  # TODO Generic error catching = bad
        raise SystemExit(_error)
    return response.json()


def parse_forcast_response(forecast_response_json: dict) -> dict[str, float]:
    """From a full api response dict, extract the latitude, longitude, forecast dates,
    hourly forecast for dates, with corresponding temperatures

    Args:
        forecast_response_json (dict): full response from get_forecast function

    Returns:
        dict: latitude, longitude, dates,hours, minmax temp dictionary
    """
    num_days_forecasted = range(len(forecast_response_json["forecast"]["forecastday"]))
    num_hours = range(0, 24)
    forecast_day = forecast_response_json["forecast"]["forecastday"]
    lat = forecast_response_json["location"]["lat"]
    lon = forecast_response_json["location"]["lon"]
    tuple([lat, lon])
    days = []
    time = []
    temps = []
    for day in num_days_forecasted:
        days.append(forecast_day[day]["date"])
        for hour in num_hours:
            time.append(forecast_day[day]["hour"][hour]["time"])
            temps.append(forecast_day[day]["hour"][hour]["temp_c"])
    # Extracting time from datetime
    time = [datetime.strptime(x, "%Y-%m-%d %H:%M") for x in time]
    time = [str(x.time()) for x in time]
    response_dict = {}
    # response_dict['lat_lon'] = lat_lon
    step = len(time) / len(forecast_response_json["forecast"]["forecastday"])
    for i in num_days_forecasted:
        step1 = int(i * step)
        step2 = int((i + 1) * step)
        response_dict[days[i]] = dict(zip(time[step1:step2], temps[step1:step2]))
    """
    "Steps" explanation above.
    the time and temps list contain all the time/temps for every day and hour
    To map the correct chunk of each list to the correct day I need to split
    each list into sizes of len(temps)/num_days -> "step".
    I then slice each list based on this step and append to the correct day key in the dict.
    """
    df = pd.json_normalize(response_dict, sep=" ")  # type: ignore
    return df.to_dict(orient="records")[0]


def prepare_mapping_df(
    area_lat_lon_map: dict, grid_zone_map: dict, ptid_area_map: dict
) -> pd.DataFrame:
    # TODO This df should be created an saved earlier in preprocessing,
    # doesnt really belong here
    """Maps seperate information about the state area into a single dataframe.
    Maps: Lat_lon info, grid zone info, ptid number, and area name.
    Resulting df is of the following format:
    +------+------+-------+-----------+-------+---------------+
    | Area | Lat  | Lon   | Grid zone | PTID  | Lat_Lon       |
    +------+------+-------+-----------+-------+---------------+
    | ALB  | 42.5 | -73.5 | CAPITL    | 61575 | (42.5, -73.5) |
    +------+------+-------+-----------+-------+---------------+


    Args:
        area_lat_lon_map (dict): Area names to lat_lon values dict
        grid_zone_map (dict): grid zone to area name mapping dict
        ptid_area_map (dict): area name to ptid value mapping dict

    Returns:
        pd.DataFrame: dataframe as described above
    """
    df = pd.DataFrame.from_dict(area_lat_lon_map).T.rename(columns={0: "Lat", 1: "Lon"})  # type: ignore
    df["Grid Zone"] = df.index.map(grid_zone_map)
    df["PTID"] = df["Grid Zone"].map(ptid_area_map)
    df["Lat_Lon"] = tuple(zip(df["Lat"], df["Lon"]))
    df = df.reset_index(drop=False).rename(columns={"index": "Area"})
    df = df.drop("index", axis=1, errors="ignore")
    return df


def prepare_prediction_df(
    mapping_df: pd.DataFrame, df_row_no: int, api_key: str, num_days_forecast: int = 3
) -> pd.DataFrame:
    """For a single row in the mapping_df (returned prepare_mapping_df function),
    retrieves the weather forecast for that given lat_lon for the number of days specified.
    For each timestamp in the df, splits into year, month, day, hour.
    Then finds the min/max daily temperatures.
    Finally it merges the weather results back with the mapping df, with the mapping info rows
    duplicated down to fill the entire df.
    Final returned df is of the following structure:

    | index | Area | Lat | Lon | Grid Zone | PTID   | Lat_Lon | timestamp               | temp | year | month | day | minute | hour | max_temp | min_temp |
    |-------|------|-----|-----|-----------|--------|---------|-------------------------|------|------|-------|-----|--------|------|----------|----------|
    | 0     | ALB  | 42  | -72 | CAPITL    | 615757 | 42, -72 | 2023-07-23T00:00:00.000 | 20.1 | 2023 | 7     | 23  | 0      | 0    | 30.2     | 16.1     |
    | 1     | ALB  | 42  | -72 | CAPITL    | 615757 | 42, -72 | 2023-07-23T00:00:00.000 | 19.2 | 2023 | 7     | 23  | 0      | 0    | 30.2     | 16.1     |


    Args:
        mapping_df (pd.DataFrame): mapping df from prepare_mapping_df func
        df_row_no (int): corresponding row number in mapping df to pull weather forecast info for
        api_key (str): weather_api api key
        num_days_forecast (int, optional): number of days to forecast. Defaults to 3.

    Returns:
        pd.DataFrame: pandas dataframe of the above structure
    """
    _df = mapping_df.copy()

    df_row_at_num = pd.DataFrame(_df.iloc[df_row_no]).T
    df_row_at_num["forecast_response"] = df_row_at_num["Lat_Lon"].apply(
        lambda x: parse_forcast_response(
            get_forecast(lat_lon=x, api_key=api_key, forecast_days=num_days_forecast)
        )
    )

    df_row_forecast = pd.json_normalize(df_row_at_num["forecast_response"]).T.reset_index()  # type: ignore
    df_row_forecast = df_row_forecast.rename(columns={0: "temp", "index": "timestamp"})
    df_row_forecast["timestamp"] = pd.to_datetime(df_row_forecast["timestamp"])  # type: ignore
    df_row_forecast["year"] = df_row_forecast["timestamp"].dt.year
    df_row_forecast["month"] = df_row_forecast["timestamp"].dt.month
    df_row_forecast["day"] = df_row_forecast["timestamp"].dt.day
    df_row_forecast["minute"] = df_row_forecast["timestamp"].dt.minute
    df_row_forecast["hour"] = df_row_forecast["timestamp"].dt.hour

    daily_min = (
        df_row_forecast.groupby("day", as_index=False)
        .min()[["day", "temp"]]
        .rename(columns={"temp": "min_temp"})
    )
    daily_max = (
        df_row_forecast.groupby("day", as_index=False)
        .max()[["day", "temp"]]
        .rename(columns={"temp": "max_temp"})
    )

    df_row_max_merge = df_row_forecast.merge(daily_max, on="day")
    df_row_min_merge = df_row_max_merge.merge(daily_min, on="day")

    df_row_at_num = df_row_at_num.drop("forecast_response", axis=1, errors="ignore")  # type: ignore
    df_row_at_num = df_row_at_num.loc[
        np.repeat(df_row_at_num.index, df_row_min_merge.shape[0])  # type: ignore
    ].reset_index(drop=True)

    df_row_final = pd.concat([df_row_at_num, df_row_min_merge], axis=1)
    df_row_final = df_row_final.dropna()
    return df_row_final


def merge_prediction_dfs(mapping_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """For each row in the mapping_df:
    1. Creates weather forecast df via prepare_prediction_df func
    2. appends to a list
    3. merges all weather forecast dfs

    Args:
        mapping_df (pd.DataFrame): mapping df from prepare_mapping_df func

    Returns:
        pd.DataFrame: combined df containing all date/time forecast info for all rows in mapping_df
    """
    prediction_dfs = []
    for row_num in range(len(mapping_df)):
        prediction_dfs.append(prepare_prediction_df(mapping_df, row_num, **kwargs))
    return pd.concat(prediction_dfs, axis=0)


def get_weather_forecast() -> pd.DataFrame:
    """Pulls in local info (api key, mapping info, location info).
    Maps the area, PTID, and lat_lon info into a single dataframe and creates
    weather forecast df for all times/dates for each NY state area as defined in
    weather_location_mapping.json

    Returns:
        pd.DataFrame: Weather forecast info for each NY state site
    """
    API_TOML_DIR = Path("api_creds.toml")
    WEATHER_LOCATION_MAPPING_DIR = Path(
        "data_information", "weather_location_mapping.json"
    )
    PTID_AREA_MAPPING_DIR = Path("data_information", "PTID_name_mapping.json")
    with open(API_TOML_DIR, "rb") as f:
        api_key = tomli.load(f)["api_key"]
    mapping_json = json.loads(
        Path(WEATHER_LOCATION_MAPPING_DIR).read_text(encoding="UTF-8")
    )
    ptid_area_mapping = json.loads(
        Path(PTID_AREA_MAPPING_DIR).read_text(encoding="UTF-8")
    )
    area_info_df = prepare_mapping_df(
        area_lat_lon_map=mapping_json["lat_lon"],
        grid_zone_map=mapping_json["grid_zone"],
        ptid_area_map=ptid_area_mapping,
    )

    return merge_prediction_dfs(mapping_df=area_info_df, api_key=api_key)


def save_weather_forecast(weather_predictions_df: pd.DataFrame, save_dir: Path) -> None:
    """Saved a df as a parquet to a given file dir
    Appends todays date to the filename before saving

    Args:
        weather_predictions_df (pd.DataFrame): df to save
        save_dir (Path): directory to save in
    """
    today_str = date.today().strftime("%Y_%m_%d")
    SAVE_FNAME = Path(save_dir, f"forecast_{today_str}.parquet")
    weather_predictions_df.to_parquet(SAVE_FNAME)


if __name__ == "__main__":

    SAVE_PREDICTIONS_DIR = Path(
        Path.cwd(), "data.nosync", "outputs", "weather_forecast"
    )
    weather_predictions = get_weather_forecast()
    save_weather_forecast(weather_predictions, SAVE_PREDICTIONS_DIR)
    print("saved")
