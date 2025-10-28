# features_forecasting.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ========================
# Feature Engineering: RFM
# ========================
def create_rfm_features(df, reference_date=None):
    """
    Tạo các đặc trưng RFM + CLV + purchase/churn trong 90 ngày.
    """
    if reference_date is None:
        reference_date = df["DATE_"].max() - pd.Timedelta(days=365)
    
    # Quá khứ
    past_df = df[df["DATE_"] <= reference_date].copy()
    
    rfm = past_df.groupby("USERID").agg(
        Recency=("DATE_", lambda x: (reference_date - x.max()).days),
        Frequency=("ORDERID", "nunique"),
        Monetary=("TOTALPRICE", "sum"),
        AvgBasketSize=("TOTALPRICE", "mean"),
        NumCategories=("CATEGORY1", "nunique")
    ).reset_index()
    
    # Tương lai 90 ngày
    future_df = df[(df["DATE_"] > reference_date) & 
                   (df["DATE_"] <= reference_date + pd.Timedelta(days=90))].copy()
    future_summary = future_df.groupby("USERID", as_index=False).agg(
        FutureOrders90=("ORDERID", "nunique"),
        FutureTotal90=("TOTALPRICE", "sum")
    )
    rfm = rfm.merge(future_summary, on="USERID", how="left").fillna(0)
    rfm["PurchaseNext90"] = (rfm["FutureOrders90"] > 0).astype(int)
    rfm["Churn90"] = (rfm["FutureOrders90"] == 0).astype(int)
    
    # CLV
    rfm["Frequency_adj"] = rfm["Frequency"].replace(0,1)
    rfm["CLV90_raw"] = rfm["FutureTotal90"]
    rfm["CLV90_capped"] = np.clip(rfm["CLV90_raw"], 0, rfm["CLV90_raw"].quantile(0.99))
    rfm["CLV90_log"] = np.log1p(rfm["CLV90_capped"])
    
    return rfm, future_df


def cluster_rfm(rfm, features):
    """
    Cluster khách hàng dựa trên RFM features.
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    best_score, best_k = -1, 2
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(rfm_scaled)
        score = silhouette_score(rfm_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = km.fit_predict(rfm_scaled)
    rfm = pd.get_dummies(rfm, columns=["Cluster"], prefix="Cluster")
    features_model = features + [col for col in rfm.columns if col.startswith("Cluster_")]

    return rfm, features_model, scaler, best_k, best_score


# ========================
# Forecasting
# ========================
def prepare_time_series(df, category_col="CATEGORY1"):
    """
    Tạo chuỗi thời gian tổng doanh thu theo vùng & tháng.
    """
    df = df[df[category_col] == "home"].copy()
    df["MONTH"] = pd.to_datetime(df["DATE_"].dt.to_period("M").dt.to_timestamp())
    monthly = df.groupby(["REGION","MONTH"])["TOTALPRICE"].sum().reset_index().rename(columns={"TOTALPRICE":"TOTAL_REVENUE"})
    return monthly


def train_xgb_forecast(region_df, n_future_months=6):
    """
    Huấn luyện XGBRegressor dự báo 6 tháng tới, trả về model + metrics + future forecast.
    """
    region_df = region_df.sort_values("MONTH").reset_index(drop=True)
    region_df['month'] = region_df['MONTH'].dt.month
    region_df['year'] = region_df['MONTH'].dt.year
    region_df['lag1'] = region_df['TOTAL_REVENUE'].shift(1)
    region_df['lag2'] = region_df['TOTAL_REVENUE'].shift(2)
    region_df['lag3'] = region_df['TOTAL_REVENUE'].shift(3)
    region_df['rolling_mean3'] = region_df['TOTAL_REVENUE'].rolling(3).mean()
    region_df = region_df.dropna().reset_index(drop=True)

    # Train/Test Split
    split_idx = int(len(region_df) * 0.8)
    train_df = region_df.iloc[:split_idx]
    test_df = region_df.iloc[split_idx:]

    X_cols = ['month','year','lag1','lag2','lag3','rolling_mean3']
    X_train, y_train = train_df[X_cols], train_df['TOTAL_REVENUE']
    X_test, y_test = test_df[X_cols], test_df['TOTAL_REVENUE']

    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    safe_y = np.where(y_test==0, np.nan, y_test)
    mape = np.nanmean(np.abs((safe_y - y_pred)/safe_y))*100

    # Dự báo n_future_months tiếp theo
    future_preds = []
    last_data = region_df.copy()
    last_month = last_data['MONTH'].max()

    for i in range(1, n_future_months+1):
        next_month = (last_month + pd.DateOffset(months=i)).replace(day=1)
        lag1 = last_data['TOTAL_REVENUE'].iloc[-1]
        lag2 = last_data['TOTAL_REVENUE'].iloc[-2]
        lag3 = last_data['TOTAL_REVENUE'].iloc[-3]
        rolling_mean3 = last_data['TOTAL_REVENUE'].iloc[-3:].mean()
        X_future = pd.DataFrame([{
            'month': next_month.month,
            'year': next_month.year,
            'lag1': lag1,
            'lag2': lag2,
            'lag3': lag3,
            'rolling_mean3': rolling_mean3
        }])
        y_future = model.predict(X_future)[0]
        y_future = max(0, y_future)
        future_preds.append({'MONTH': next_month, 'TOTAL_REVENUE': y_future, 'REGION': region_df['REGION'].iloc[0]})
        new_row = {'MONTH': next_month, 'TOTAL_REVENUE': y_future}
        last_data = pd.concat([last_data, pd.DataFrame([new_row])], ignore_index=True)

    future_df = pd.DataFrame(future_preds)
    return model, mae, rmse, mape, future_df
