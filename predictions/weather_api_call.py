import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import tomli


@dataclass
class PredResult:
    area: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    lat_lon: Optional[tuple[float, float]] = None
    grid_zone: Optional[str] = None
    ptid: Optional[int] = None
    date: Optional[str] = None
    max_temp: Optional[float] = None
    min_temp: Optional[float] = None
    predicted_load: Optional[float] = None


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
        response = requests.get(BASE_URL, params=query_params, timeout=10) #type: ignore
    except (
        requests.exceptions.RequestException
    ) as _error:  # TODO Generic error catching = bad
        raise SystemExit(_error)
    return response.json()


def parse_forcast_response(forecast_response_json: dict) -> dict:
    """From a full api response dict, extract the latitude, longitude, forecast dates and corresposnding
    min and max termperatures forecasted,

    Args:
        forecast_response_json (dict): full response from get_forecast function

    Returns:
        dict: latitude, longitude, dates, minmax temp dictionary
    """
    num_days_forecasted = len(forecast_response_json["forecast"]["forecastday"])
    lat = forecast_response_json["location"]["lat"]
    lon = forecast_response_json["location"]["lon"]
    dates = []
    max_temps = []
    min_temps = []
    for day in range(num_days_forecasted):
        dates.append(forecast_response_json["forecast"]["forecastday"][day]["date"])
        max_temps.append(
            forecast_response_json["forecast"]["forecastday"][day]["day"]["maxtemp_c"]
        )
        min_temps.append(
            forecast_response_json["forecast"]["forecastday"][day]["day"]["mintemp_c"]
        )
    return {
        "lat": lat,
        "lon": lon,
        "dates": dates,
        "max_temps": max_temps,
        "min_temps": min_temps,
    }


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
    df = pd.DataFrame.from_dict(area_lat_lon_map).T.rename(columns={0: "Lat", 1: "Lon"}) #type: ignore
    df["Grid Zone"] = df.index.map(grid_zone_map)
    df["PTID"] = df["Grid Zone"].map(ptid_area_map)
    df["Lat_Lon"] = tuple(zip(df["Lat"], df["Lon"]))
    df = df.reset_index(drop=False).rename(columns={"index": "Area"})
    df = df.drop("index", axis=1, errors="ignore")
    return df


def convert_mapping_df_to_dataclasses(mapping_df: pd.DataFrame) -> list[PredResult]:
    """converts a pandas dataframe with gridzone, ptid, area, lat, lon info
    into a list of dataclasses as described by the PredResult class.

    Args:
        mapping_df (pd.DataFrame): output from prepare_mapping_df() function

    Returns:
        list[PredResult]: list of PredResult classes. Class should be fulling populated
        excep for the predicted load values.
    """
    # Converting dataframe to list of dataclasses
    area_info_list = []  # TODO This is a poor name
    for row in mapping_df.itertuples():
        result_class = PredResult()
        result_class.area = row.Area  #type: ignore
        result_class.lat = float(row.Lat) #type: ignore
        result_class.lon = float(row.Lon) #type: ignore
        result_class.lat_lon = tuple([float(row.Lat), float(row.Lon)]) #type: ignore
        result_class.grid_zone = (
            row._4 #type: ignore
        )  # BUG Not sure why grid_zone col name hasnt flowed through
        result_class.ptid = float(row.PTID) #type: ignore
        area_info_list.append(result_class)
    return area_info_list


def write_forecast_results(
    results_list: list[PredResult], **kwargs
) -> list[PredResult]:
    """Gets min/max temperatures for given number of days as specified by
    "forecast_days" kwarg passed to get_forecast function

    Args:
        results_list (list[PredResult]): list of PredResult dataclasses holding area info:
        area name, lat_lon, grid_zone, ptid

    Returns:
        list[PredResult]: same list of PredResults with additional date and min/max
        forecasted temperature data
    """
    for res in results_list:
        forecast = parse_forcast_response(get_forecast(lat_lon=res.lat_lon, **kwargs)) #type: ignore
        res.date = forecast["dates"]
        res.max_temp = forecast["max_temps"]
        res.min_temp = forecast["min_temps"]
    return results_list


def get_weather_forecast() -> list[PredResult]:
    DAYS_FORECAST = 3
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
    df = prepare_mapping_df(
        area_lat_lon_map=mapping_json["lat_lon"],
        grid_zone_map=mapping_json["grid_zone"],
        ptid_area_map=ptid_area_mapping,
    )
    area_info = convert_mapping_df_to_dataclasses(df)
    return write_forecast_results(
        area_info, api_key=api_key, forecast_days=DAYS_FORECAST
    )


if __name__ == "__main__":

    forecast_results = get_weather_forecast()
    print(forecast_results)
    print()
    print()
