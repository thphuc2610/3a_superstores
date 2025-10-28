# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import numpy as np

def build_purchase_model(rfm, features, scaler):
    X_scaled = scaler.transform(rfm[features])
    y = rfm["PurchaseNext90"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def build_clv_model(rfm, features, scaler):
    X_scaled = scaler.transform(rfm[features])
    y = rfm["CLV90_log"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    w = 1 / (1 + rfm.loc[y_train.index, "CLV90_capped"].values)
    w = w / np.mean(w)
    
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=42)
    model.fit(X_train, y_train, sample_weight=w)
    
    return model, X_test, y_test

def build_churn_model(rfm, features, scaler):
    X_scaled = scaler.transform(rfm[features])
    y = rfm["Churn90"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test
