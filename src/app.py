# app.py
import pandas as pd
import plotly.graph_objects as go

# Preprocessing & Feature Engineering
from preprocessing_polars import load_and_clean_data_polars
from features_forecasting import create_rfm_features, cluster_rfm, prepare_time_series, train_xgb_forecast

# Mô hình khách hàng
from model_training import build_purchase_model, build_clv_model, build_churn_model
from evaluation import evaluate_models

def main():
    # ================================
    # 1️⃣ Load & clean data
    # ================================
    print("🔹 Load và làm sạch dữ liệu...")
    df = load_and_clean_data_polars()
    print(f"✅ Dữ liệu sau làm sạch: {df.shape[0]:,} dòng")

    # ================================
    # 2️⃣ RFM & Feature Engineering
    # ================================
    print("🔹 Tạo RFM features...")
    rfm, future_df = create_rfm_features(df.to_pandas())
    features = ["Recency", "Frequency", "Monetary", "AvgBasketSize", "NumCategories"]
    rfm, features_model, scaler, best_k, best_score = cluster_rfm(rfm, features)
    print(f"✅ RFM clustering hoàn tất: K={best_k}, Silhouette={best_score:.2f}")

    # ================================
    # 3️⃣ Train customer models
    # ================================
    print("🔹 Huấn luyện mô hình khách hàng (Purchase, CLV, Churn)...")
    purchase_model, _, _ = build_purchase_model(rfm, features_model, scaler)
    clv_model, _, _ = build_clv_model(rfm, features_model, scaler)
    churn_model, _, _ = build_churn_model(rfm, features_model, scaler)

    # ================================
    # 4️⃣ Evaluate models
    # ================================
    print("🔹 Đánh giá mô hình...")
    rfm, metrics = evaluate_models(rfm, future_df, features_model, scaler, purchase_model, clv_model, churn_model)
    print("=== Metrics tổng quan ===")
    for k,v in metrics.items():
        print(f"{k}: {v*100:.2f}%" if "Accuracy" in k else f"{k}: {v:.2f}")

    # ================================
    # 5️⃣ Forecast doanh thu Home
    # ================================
    print("🔹 Chuẩn bị dữ liệu dự báo cho danh mục Home...")
    df_pd = df.to_pandas()
    home_monthly = prepare_time_series(df_pd, category_col="CATEGORY1")

    forecast_all = pd.DataFrame()
    metrics_all = []

    for region in home_monthly['REGION'].unique():
        region_df = home_monthly[home_monthly['REGION'] == region].copy()
        model, mae, rmse, mape, future_df = train_xgb_forecast(region_df, n_future_months=6)
        metrics_all.append({
            "REGION": region,
            "MAE": round(mae,2),
            "RMSE": round(rmse,2),
            "MAPE (%)": round(mape,2)
        })
        forecast_all = pd.concat([forecast_all, future_df], ignore_index=True)

        # Vẽ biểu đồ thực tế + dự báo
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=region_df['MONTH'], y=region_df['TOTAL_REVENUE'],
                                 mode='lines+markers', name='Thực tế', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=future_df['MONTH'], y=future_df['TOTAL_REVENUE'],
                                 mode='lines+markers', name='Dự báo 6 tháng', line=dict(color='green', dash='dot')))
        fig.update_layout(title=f"📊 Dự báo Doanh thu Home - {region} (6 tháng tới)",
                          xaxis_title="Tháng", yaxis_title="Tổng Doanh Thu",
                          template="plotly_white", height=500)
        fig.show()

    metrics_df = pd.DataFrame(metrics_all).sort_values("MAPE (%)")
    print("📈 Độ chính xác mô hình XGBoost (Test gần nhất):")
    display(metrics_df.style.background_gradient(subset=["MAPE (%)"], cmap="RdYlGn_r"))

    print("\n🔮 Dự báo 6 tháng tới cho từng khu vực:")
    display(forecast_all.sort_values(["REGION","MONTH"]))

if __name__ == "__main__":
    main()
