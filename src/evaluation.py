# evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_squared_error

def smape(a, f):
    return np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-8)) * 100

def evaluate_models(rfm, future_df, features_model, scaler, purchase_model, clv_model, churn_model):
    X_all = scaler.transform(rfm[features_model])
    rfm['PurchaseProb'] = purchase_model.predict_proba(X_all)[:,1]
    rfm['PredLogCLV'] = clv_model.predict(X_all)
    rfm['PredCLV'] = np.expm1(rfm['PredLogCLV'])

    if len(churn_model.classes_) == 1:
        only_class = churn_model.classes_[0]
        rfm['ChurnProb'] = 1.0 if only_class == 1 else 0.0
    else:
        rfm['ChurnProb'] = churn_model.predict_proba(X_all)[:,1]

    rfm['PurchaseLabel'] = (rfm['PurchaseProb'] >= 0.5).astype(int)
    rfm['ChurnLabel'] = (rfm['ChurnProb'] >= 0.5).astype(int)

    actuals = future_df.groupby('USERID').agg(
        actual_orders=('ORDERID','nunique'),
        actual_total=('TOTALPRICE','sum')
    ).reset_index()

    rfm = rfm.merge(actuals, on='USERID', how='left')
    rfm[['actual_orders','actual_total']] = rfm[['actual_orders','actual_total']].fillna(0)
    rfm['ActualPurchase'] = (rfm['actual_orders'] > 0).astype(int)
    rfm['ActualChurn'] = (rfm['actual_orders'] == 0).astype(int)

    metrics = {
        "PurchaseAccuracy": accuracy_score(rfm['ActualPurchase'], rfm['PurchaseLabel']),
        "ChurnAccuracy": accuracy_score(rfm['ActualChurn'], rfm['ChurnLabel']),
        "CLV_MAPE": mean_absolute_percentage_error(rfm['actual_total'] + 1e-6, rfm['PredCLV'] + 1e-6) * 100,
        "CLV_SMAPE": smape(rfm['actual_total'], rfm['PredCLV']),
        "CLV_RMSE": np.sqrt(mean_squared_error(rfm['actual_total'], rfm['PredCLV']))
    }

    return rfm, metrics
