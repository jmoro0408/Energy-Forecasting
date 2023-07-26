import datetime
from pathlib import Path

import pandas as pd

from flask import (Blueprint, jsonify, redirect, render_template, request,
                   url_for)

views = Blueprint(__name__, "views")

@views.route("/views/")
def home():
    return render_template("index.html", name = "James") # Can pass multiple vals to render template which are passed to {{}} in index.html

@views.route("/profile/")
def profile():
    return render_template("profile.html")

def get_latest_forecast_date(saved_weather_forecast_dir:Path) -> str:
    """returns the latest date present in the folder of saved weather forecasts

    Args:
        saved_weather_forecast_dir (Path): directory holding saved weather forecast parquet files

    Returns:
        str: latest forecast date
    """
    files = Path(saved_weather_forecast_dir).glob("*.parquet")
    files = [p.stem for p in files]
    files = [p[11:21] for p in files]
    return sorted(files, key=lambda x: datetime.datetime.strptime(x, '%Y_%m_%d'),  reverse=True)[0]

@views.route("/results/")
def results():
    PRED_SAVE_DIR = Path(Path.cwd().parent, "data.nosync", "outputs", "load_predictions")
    latest_pred_date = get_latest_forecast_date(PRED_SAVE_DIR)
    fname = f"prediction_{latest_pred_date}.parquet"
    df = pd.read_parquet(Path(PRED_SAVE_DIR, fname))
    df['Date'] = df['Year'].astype(str) + "-" + df['Month'].astype(str) + "-" + df['Day'].astype(str)
    df['Hour'] = df['Hour'].astype(str)
    df['Minute']= df['Minute'].astype(str)
    df['Minute'] = df['Minute'].str.zfill(2) #padding with leading zeros
    df['Hour'] = df['Hour'].str.zfill(2)
    df['Time'] = df['Hour'] + ":" + df['Minute']
    df['pred_load'] = df['pred_load'].astype(float).round(2)
    df['PTID'] = df['PTID'].astype(float).round(0).astype(int)
    cols_to_display = ['Area', 'Lat', 'Lon', 'Grid Zone', 'PTID',
                       'Date', 'Time', 'temp','Max_Temp', 'Min_Temp','pred_load']
    df['Lat'] = df['Lat'].astype(float).round(2)
    df['Lon'] = df['Lon'].astype(float).round(2)
    df = df[cols_to_display]
    df = df.rename(columns = {'Max_Temp': 'Max Daily Temp (C)', 'Min_Temp':'Min Daily Temp (C)',
                              'temp':'Temp (C)', 'pred_load':'Predicted Load (kW)'})
    # return render_template("dataframe.html",
    #                        column_names=df.columns.values,
    #                        row_data=list(df.values.tolist()),
    #                        zip=zip)
    return render_template("dataframe.html", data=df.to_html(table_id="df_output", index = False))


@views.route("/json/")
def get_json():
    return jsonify({'name':'james', 'coolness':10})


@views.route("/data/")
def get_data():
    data = request.json
    return jsonify(data)

@views.route("/go_to_home/")
def go_to_home():
    return redirect(url_for("views.home"))
