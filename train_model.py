import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
import requests
from retry_requests import retry
import openmeteo_requests
import numpy as np
import jpholiday
import joblib
import os
os.makedirs("demand_prediction", exist_ok=True)

# データ読み込み
weather = pd.read_csv("open-meteo-35.68N139.81E6m.csv", index_col=0,parse_dates=True)
dataset = pd.read_csv("beers.csv", index_col=0,parse_dates=True)

# 結合・前処理
df = pd.merge(dataset, weather, left_index=True, right_index=True)
df = df.fillna(0)
#金曜日かどうかの列を追加
df.index = pd.to_datetime(df.index)
df["is_friday"] = (df.index.weekday == 4).astype(int)
#祝日かどうかの列を追加
df["is_holiday"] = df.index.to_series().apply(lambda d: 1 if jpholiday.is_holiday(d) else 0)

# ターゲット列（すべてのビールの本数）
target_columns = [
    "pale_ale_bottles",
    "lager_bottles",
    "ipa_bottles",
    "white_beer_bottles",
    "black_beer_bottles",
    "fruit_beer_bottles"
]
target_sales = [
    "pale_ale_yen",
    "lager_yen",
    "ipa_yen",
    "white_beer_yen",
    "black_beer_yen",
    "fruit_beer_yen"
]
# 特徴量と目的変数に分割
# 特徴量に他のビールの売上を含めないようにする
drop_columns = target_columns + target_sales + ["day of the week (1: sunday)","weather_code (wmo code)", "weather", "total_cups", "total_sales_yen"]
X = df.drop(columns=drop_columns)
# y はそのままターゲットビール群
y = df[target_columns]

# 学習用・テスト用データに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# モデルの定義（回帰木 + マルチターゲット）
base_model = DecisionTreeRegressor(max_depth=5, random_state=0)
multi_model = MultiOutputRegressor(base_model)
multi_model.fit(X_train, y_train)

# 予測
y_pred = multi_model.predict(X_test)
# 評価表示
print("▼ 各ビールの予測精度（R²スコア）")
# 予測精度を出力
for i, col in enumerate(target_columns):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print(f"{col}: R² = {r2:.3f}, RMSE = {rmse:.2f}")



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
visitors_default = X_train['visitors'].mean()

def predict_sales(sample_dict):
    sample_dict = sample_dict.copy()
    sample_dict.setdefault('visitors', visitors_default)
    df = pd.DataFrame([sample_dict])[X_train.columns]
    pred = multi_model.predict(df)[0]
    pred = [round(float(x), 2) for x in pred]
    return dict(zip(y_train.columns, pred))

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

# 保存
joblib.dump(multi_model, "demand_prediction/model.pkl")
joblib.dump(X.columns.tolist(), "demand_prediction/x_columns.pkl")
with open("demand_prediction/target_columns.json", "w") as f:
    json.dump(target_columns, f, ensure_ascii=False)