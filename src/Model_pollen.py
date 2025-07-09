import joblib
import requests
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    TimeSeriesSplit,
    train_test_split,
    RandomizedSearchCV,
)


# TODO: test
# TODO: write documentation
# TODO: adapt for short actual prediction (for a week)
# TODO: rewrite with piplines and trnasformers
# TODO: add into pipline standard scaler
""" ~_~ """
class Model_poll:
    def __init__(self, data: str):
        # self.data_weater = pd.read_csv(path_to_weater, sep=s1)
        # self.data_pollen = pd.read_csv(path_to_pollen, sep=s2)
        self.data = data
        self.X_train = ...
        self.y_train = ...
        self.X_test = ...
        self.y_test = ...
        self.model = ...
    
    def learn_xgb_model(self, train_years: list=[2017, 2019, 2020, 2021, 2023],
                        test_year: int = 2022) -> None:
        xgboost_data = self.data.copy()
        xgboost_data['date'] = pd.to_datetime(xgboost_data['date'])
        xgboost_data = xgboost_data.set_index("date")
        # xgboost_data.reset_index()
    
        for lag in [1, 2, 3, 7]:
            xgboost_data[f"pollen_lag_{lag}"] = xgboost_data["concentration"].shift(lag)
            xgboost_data[f"temp_mean_lag_{lag}"] = xgboost_data["temp_mean"].shift(lag)
            xgboost_data[f"temp_min_lag_{lag}"] = xgboost_data["temp_min"].shift(lag)
            xgboost_data[f"temp_max_lag_{lag}"] = xgboost_data["temp_max"].shift(lag)
        xgboost_data['day_of_year'] = xgboost_data.index.dayofyear
        xgboost_data['day_sin'] = np.sin(2 * np.pi * xgboost_data['day_of_year'] / 365)
        xgboost_data['day_cos'] = np.cos(2 * np.pi * xgboost_data['day_of_year'] / 365)
        xgboost_data['pollen_7d_avg'] = xgboost_data['concentration'].rolling(7).mean()
        xgboost_data['pollen_7d_max'] = xgboost_data['concentration'].rolling(7).max()

        xgboost_data = xgboost_data.dropna()

        train_mask = xgboost_data.index.year.isin(train_years)
        test_mask = xgboost_data.index.year == test_year

        X = xgboost_data.drop("concentration", axis=1)
        y = xgboost_data["concentration"]

        self.X_train = xgboost_data[train_mask].drop("concentration", axis=1) # TODO: think about that
        self.y_train = xgboost_data[train_mask]["concentration"]
        self.X_test = xgboost_data[test_mask].drop("concentration", axis=1)
        self.y_test = xgboost_data[test_mask]["concentration"]

        weights = np.where(self.y_train == 0, 1, 3)

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42
        )

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 1, 10]
        }

        search = RandomizedSearchCV(model, param_grid, n_iter=50, cv=3, scoring='neg_mean_absolute_error')
        search.fit(self.X_train, self.y_train, sample_weight=weights)
        self.model = search.best_estimator_

    def _culc_features(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index("date")
    
        for lag in [1, 2, 3, 7]:
            result[f"pollen_lag_{lag}"] = result["concentration"].shift(lag)
            result[f"temp_mean_lag_{lag}"] = result["temp_mean"].shift(lag)
            result[f"temp_min_lag_{lag}"] = result["temp_min"].shift(lag)
            result[f"temp_max_lag_{lag}"] = result["temp_max"].shift(lag)
        result['day_of_year'] = result.index.dayofyear
        result['day_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
        result['day_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
        result['pollen_7d_avg'] = result['concentration'].rolling(7).mean()
        result['pollen_7d_max'] = result['concentration'].rolling(7).max()

        result = result.dropna()
        return result.drop("concentration", axis=1)

    def learn_hybrid(self, merged_data: pd.DataFrame,
                        train_years: list=[2017, 2019, 2020, 2021, 2023],
                        test_year: int = 2022) -> None:
        hybrid_data = merged_data.copy()

        hybrid_data['day_of_year'] = hybrid_data.index.dayofyear
        hybrid_data['temp_5d_avg'] = hybrid_data['temp_mean'].rolling(5).mean()
        hybrid_data['warming_flag'] = (hybrid_data['temp_mean'].rolling(3).min() > 12).astype(int)

        hybrid_data['is_season'] = (hybrid_data['concentration'] > 0).astype(int)

        for lag in [1, 2, 3, 7]:
            hybrid_data[f"pollen_lag_{lag}"] = hybrid_data["concentration"].shift(lag)
            hybrid_data[f"temp_mean_lag_{lag}"] = hybrid_data["temp_mean"].shift(lag)
            hybrid_data[f"precip_lag_{lag}"] = hybrid_data["precip_sum"].shift(lag)
            hybrid_data[f"temp_min_lag_{lag}"] = hybrid_data["temp_min"].shift(lag)
            hybrid_data[f"temp_max_lag_{lag}"] = hybrid_data["temp_max"].shift(lag)
        hybrid_data['pollen_7d_max'] = hybrid_data['concentration'].rolling(7).max()
        hybrid_data['days_in_season'] = hybrid_data.groupby(
                (hybrid_data['is_season'] == 0).cumsum()
            ).cumcount()
        hybrid_data['humidity_3d_avg'] = hybrid_data['humidity_mean'].rolling(3).mean()
        hybrid_data['precip_3d_sum'] = hybrid_data['precip_sum'].rolling(3).sum()
        hybrid_data['temp_amplitude'] = hybrid_data['temp_max'] - hybrid_data['temp_min']
        hybrid_data['season_progress'] = (hybrid_data['day_of_year'] - 100) / 60
        hybrid_data['pollen_3d_trend'] = hybrid_data['concentration'].rolling(3).apply(
            lambda x: np.polyfit(range(3), x, 1)[0], raw=True
        )
        hybrid_data['temp_humidity'] = hybrid_data['temp_mean'] * hybrid_data['humidity_mean']

        hybrid_data = hybrid_data.dropna()

        train_mask = hybrid_data.index.year.isin(train_years)
        test_mask = hybrid_data.index.year == test_year

        classifier_features = [
            'day_of_year', 
            'temp_5d_avg',
            'warming_flag',
            'temp_min',
            'temp_max',
            'humidity_3d_avg',
            'precip_3d_sum',
            'temp_amplitude',
            'season_progress'
        ]

        regression_features = [
            'temp_mean',
            'precip_sum',
            'humidity_mean',
            'pollen_lag_1',
            'pollen_lag_3',
            'days_in_season',
            'temp_mean_lag_3',
            'precip_lag_7',
            'temp_min_lag_1',
            'temp_min_lag_7',
            'temp_max_lag_1',
            'temp_max_lag_7',
            'pollen_7d_max',
            'pollen_3d_trend',
            'temp_humidity',
            'temp_amplitude'
        ]

        clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=5,
            verbose=0
        )

        season_mask = hybrid_data.index.month.isin([4,5,6])
        clf.fit(
            hybrid_data.loc[train_mask & season_mask, classifier_features],
            hybrid_data.loc[train_mask & season_mask, 'is_season'],
            eval_set=(hybrid_data.loc[test_mask & season_mask, classifier_features],
                    hybrid_data.loc[test_mask & season_mask, 'is_season']),
            early_stopping_rounds=50
        )

        reg_data = hybrid_data[hybrid_data['is_season'] == 1]

        train_mask_reg = reg_data.index.year.isin(train_years)

        X_reg = reg_data.loc[train_mask_reg, regression_features]
        y_reg = reg_data.loc[train_mask_reg, 'concentration']

        weights_reg = np.interp(y_reg, 
                            [0, y_reg.quantile(0.9), y_reg.max()], 
                            [1, 3, 5])

        model = xgb.XGBRegressor(objective='reg:squarederror')
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        search = RandomizedSearchCV(
            model, 
            param_grid, 
            n_iter=30, 
            cv=TimeSeriesSplit(n_splits=3),
            scoring='neg_mean_absolute_error'
        )
        search.fit(X_reg, y_reg, sample_weight=weights_reg)
        best_reg = search.best_estimator_

        test_data = hybrid_data[test_mask]
        test_data['predicted_season'] = clf.predict(test_data[classifier_features])

        X_test_reg = test_data[test_data['predicted_season'] == 1][regression_features]
        y_test_reg = test_data[test_data['predicted_season'] == 1]['concentration']         # TODO: write into class field 

        if len(X_test_reg) > 0:
            test_preds = best_reg.predict(X_test_reg)
            final_preds = pd.Series(test_preds, index=X_test_reg.index)
        else:
            final_preds = pd.Series(0, index=test_data.index)

        return final_preds.reindex(test_data.index).fillna(0)

    def get_predict(self, weather):
        return self.model.predict(weather)
    
    def predict_future(self, weather):
        """
        Get weater predict
        Return pollen predict

        Inside predict one day fill concentration data and get another predict until end
        """
        result = pd.DataFrame([[weather.index[0], 0]], columns=['date', 'concentration'])
        for ind in len(weather):
            
            result = self.model.predict(self._culc_features(pd.concat(weather[:ind], result)))

        return result

    def show_result(self, preds) -> None:
        self.Tools.draw_plot(self.y_test, preds)
        self.Tools.print_stats(self.y_test, preds)
       

class Tools:
    def get_weater(name: str, start: str, end: str, 
            latitude: float=55.7558,
            longitude: float=37.6176,
            create_csv=False) -> pd.DataFrame:
        # url = "https://archive-api.open-meteo.com/v1/archive"
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m",
            "timezone": "Europe/Moscow"
        }
        
        response = requests.get(url, params=params)
        weather_data = response.json()

        df = pd.DataFrame({
            "time": weather_data["hourly"]["time"],
            "temperature": weather_data["hourly"]["temperature_2m"],
            "precipitation": weather_data["hourly"]["precipitation"],
            "humidity": weather_data["hourly"]["relative_humidity_2m"],
            "wind_speed": weather_data["hourly"]["wind_speed_10m"]
        })
        if create_csv: df.to_csv(f"{name}_{start}-{end}.csv", index=False)
        return df
        # return weather_data

    def group_weather_by_day(weather: pd.DataFrame,
                                create_csv=False) -> pd.DataFrame:
        date = pd.to_datetime(weather['time'])
        weather["year"] = date.dt.year
        weather["month"] = date.dt.month
        weather["day"] = date.dt.day
        weather = weather.groupby(['year', 'month', 'day']
                                    ).agg(
                                        {"temperature": ["mean", "min", "max"],
                                        "precipitation": "sum",
                                        "humidity": "mean",
                                        "wind_speed": "mean"}
                                        ).reset_index()

        weather.columns = [
            'year', 'month', 'day', 
            'temp_mean', 'temp_min', 'temp_max', 
            'precip_sum', 
            'humidity_mean', 
            'wind_speed_mean'
        ]
        weather['date'] = pd.to_datetime(weather[['year', 'month', 'day']])
        weather = weather.drop(['year', 'month', 'day'], axis=1)
        weather = weather[['date', 'temp_mean', 'temp_min', 'temp_max',
                        'precip_sum', 'humidity_mean', 'wind_speed_mean']]
        if create_csv: weather.to_csv("weather.csv", index=False)
        return weather
    
    def join_datasets(weather: pd.DataFrame, pollen: pd.DataFrame, create_csv=False) -> pd.DataFrame:
        merged_data = weather.join(pollen, how="left")
        merged_data = merged_data.fillna(0)
        if create_csv: merged_data.to_csv("weather_and_pollen.csv", index=False)
        return merged_data

    def draw_plot(y_test, y_pred) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="Реальные значения", marker="o")
        plt.plot(y_test.index, y_pred, label="Прогноз", linestyle="--", marker="x")
        plt.title("Прогноз концентрации пыльцы")
        plt.xlabel("Дата")
        plt.ylabel("Концентрация")
        plt.legend()
        plt.grid(True)
        plt.show()

    def print_stats(y_test, y_pred) -> None:
        print("MAE:", mean_absolute_error(y_test, y_pred))
        print("R²:", r2_score(y_test, y_pred))
        print("RMSE:", root_mean_squared_error(y_test, y_pred))
        print("MSE:", mean_squared_error(y_test, y_pred))


class PollenModel:
    class Preprocessing:
        class ExtractWeatherFeature(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X: pd.DataFrame):
                """
                X: DataFrame with meteo data
                (time, temperature, precipitation, humidity, wind_speed)

                -> DataFrame with meteo data
                """
                df = X.copy()
                df.time = pd.to_datetime(df['time'])
                df = df.groupby('time'
                                ).agg(
                                    {"temperature": ["mean", "min", "max"],
                                    "precipitation": "sum",
                                    "humidity": "mean",
                                    "wind_speed": "mean"}
                                    ).reset_index()
                return df
            
        class ExtractWeatherAndPollenFeature(BaseEstimator, TransformerMixin):
            def __init__(self, concentration: pd.DataFrame):
                self.concentration = concentration
            
            def fit(self, X, y=None):
                return self

            def transform(self, X: pd.DataFrame):
                """
                X: DataFrame with meteo data
                (date, temp_mean, temp_min, temp_max, precip_sum, humidity_mean, wind_speed_mean)

                -> DataFrame with meteo data + pollen concentration by day
                """
                df = X.copy()
                df = df.join(self.concentration, how="left").fillna(0)
                return df

        class TrainValidationTest(BaseEstimator, TransformerMixin):
            def __init__(self, test_size=0.2, random_state=21):
                self.test_size = test_size
                self.random_state = random_state

            def fit(self, X, y):
                X_train_valid, X_test, y_train_valid, y_test = train_test_split(
                    X, y,
                    test_size=self.test_size,
                    random_state=self.random_state,
                    stratify=y
                )
                
                valid_size = self.test_size / (1 - self.test_size)
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X_train_valid, y_train_valid,
                    test_size=valid_size,
                    random_state=self.random_state,
                    stratify=y_train_valid
                )

                return X_train, X_valid, X_test, y_train, y_valid, y_test

    class FitModel:
        # TODO: fit Classifier
        # TODO: fit Regressor
        class CatBoostFit:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                return self

            def predict(self, X):
                pass
        
        class RGBoostFit:
            def __init__(self, model):
                self.model = model

            def fit(self, X, y):
                return self

            def predict(self, X):
                pass

    class ModelSelection:
        # TODO: write crosval selection
        pass

    class Finalize:
        def __init__(self, estimator):
            self.estimator = estimator
            self.metrics: pd.DataFrame = ...

        def final_score(self, X_train, y_train, X_test, y_test):
            y_pred = self.estimator.predict(X_test)
            # TODO: get metrics
            return y_pred

        def save_model(self, path):
            joblib.dump(self.estimator, path)
            print(f"Model successfully saved to {path}")