import azure.functions as func
import logging
from retry_requests import retry
import requests
import openmeteo_requests
from datetime import datetime, timedelta
import pandas as pd
import jpholiday
import json
import joblib

# 学習済みモデルと特徴量リストの読み込み
multi_model = joblib.load("demand_prediction/model.pkl")
X_columns = joblib.load("demand_prediction/x_columns.pkl")

# ----------天気APIを取得する
retry_session = retry(requests.Session(), retries=5, backoff_factor=0.2)
openmeteo     = openmeteo_requests.Client(session=retry_session)
tz        = "Asia/Tokyo"
today_jst = datetime.now().astimezone().date()
date_list = []
for i in range(8):
  date_list.append(today_jst + timedelta(days=i-1))

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude"   : 35.658,
    "longitude"  : 139.778,
    "daily"      : ["weather_code", "temperature_2m_mean", "precipitation_sum"],
    "timezone"   : tz,
    "past_days": 1,
    "forecast_days":7
}

resp   = openmeteo.weather_api(url, params=params)[0]
daily  = resp.Daily()
weather_code = daily.Variables(0).ValuesAsNumpy()
temperature__2m_mean = daily.Variables(1).ValuesAsNumpy()
precipitation_sum = daily.Variables(2).ValuesAsNumpy()

forecast = pd.DataFrame({
    "date":date_list,
    "weather_code":weather_code,
    "temperature_2m_mean":temperature__2m_mean,
    "precipitation_sum":precipitation_sum
})

forecast["weather_code"] = forecast["weather_code"].astype(int)
forecast["temperature_2m_mean"] = (forecast["temperature_2m_mean"].round(0).astype(int))
forecast["sunny or not"] = (forecast["weather_code"] < 3).astype(int)
forecast['date'] = pd.to_datetime(forecast['date'])
forecast['is_friday'] = (forecast['date'].dt.weekday == 4).astype(int)
forecast['is_holiday'] = forecast['date'].apply(lambda x: int(jpholiday.is_holiday(x)))

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
print(forecast)

#予測天気データをinputする、予測結果をoutput
visitors_default = 20 # 来店数を20名とおく（ダミーデータ）

def predict_sales(sample_dict):
    sample_dict = sample_dict.copy()
    sample_dict.setdefault('visitors', visitors_default)
    df = pd.DataFrame([sample_dict])[X_columns]
    pred = multi_model.predict(df)[0]
    pred = [round(float(x), 2) for x in pred]
    return dict(zip([
        "pale_ale_bottles",
        "lager_bottles",
        "ipa_bottles",
        "white_beer_bottles",
        "black_beer_bottles",
        "fruit_beer_bottles"
    ], pred))

for _,row in forecast.iterrows():
  row_dict = row.to_dict()
  input_data = {
    "visitors":visitors_default,
    "temperature_2m_mean (°C)": row_dict["temperature_2m_mean"],
    "precipitation_sum (mm)": row_dict["precipitation_sum"],
    "sunny or not":row_dict["sunny or not"],
    "is_friday":row_dict["is_friday"],
    "is_holiday":row_dict["is_holiday"]
  }

  result = predict_sales(input_data)
  print(f"=== {row_dict['date']}の予測")
  print(json.dumps(result,ensure_ascii=False,indent=2))
# ------------

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="http_trigger_teamJ")
def http_trigger_teamJ(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )