import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import jpholiday
import joblib
import os
os.makedirs("demand_prediction", exist_ok=True)

# データ読み込み
weather = pd.read_csv("open-meteo-35.68N139.81E6m.csv", index_col=0,parse_dates=True)
beer_dataset = pd.read_csv("beers.csv", index_col=0,parse_dates=True)

# 結合・前処理
df = pd.merge(beer_dataset, weather, left_index=True, right_index=True)
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

"""（消去）
# 予測
y_pred = multi_model.predict(X_test)
# 評価表示
print("▼ 各ビールの予測精度（R²スコア）")
# 予測精度を出力
for i, col in enumerate(target_columns):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print(f"{col}: R² = {r2:.3f}, RMSE = {rmse:.2f}")

"""

# 保存
joblib.dump(multi_model, "demand_prediction/model.pkl")
joblib.dump(X.columns.tolist(), "demand_prediction/x_columns.pkl")
with open("demand_prediction/target_columns.json", "w") as f:
    json.dump(target_columns, f, ensure_ascii=False)