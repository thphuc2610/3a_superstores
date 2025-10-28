# preprocessing.py

import pandas as pd
import numpy as np
import polars as pl

def load_and_clean_data_pandas(data_path="/kaggle/input/3a-superstore"):
    """
    Load dữ liệu bằng Pandas và làm sạch cơ bản.
    """
    # ========================
    # Load dữ liệu
    # ========================
    customers = pd.read_csv(f"{data_path}/Customers.csv", sep=";")[["USERID", "REGION"]]
    orders = pd.read_csv(f"{data_path}/Orders.csv")[["ORDERID", "USERID", "DATE_", "TOTALBASKET"]]
    order_details = pd.read_csv(f"{data_path}/Order_Details.csv")[["ORDERID", "ITEMID", "TOTALPRICE"]]
    categories = pd.read_csv(f"{data_path}/Categories.csv")[["ITEMID", "CATEGORY1"]]

    # ========================
    # Merge dữ liệu
    # ========================
    df = (
        orders
        .merge(order_details, on="ORDERID", how="left")
        .merge(categories, on="ITEMID", how="left")
        .merge(customers, on="USERID", how="left")
    )

    # ========================
    # Chuyển kiểu dữ liệu
    # ========================
    df["DATE_"] = pd.to_datetime(df["DATE_"], errors="coerce")
    for col in ["TOTALBASKET", "TOTALPRICE"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    df["USERID"] = df["USERID"].astype(str)
    df["ORDERID"] = df["ORDERID"].astype(str)
    df["ITEMID"] = df["ITEMID"].astype(str)
    df["REGION"] = df["REGION"].astype("category")
    df["CATEGORY1"] = df["CATEGORY1"].astype("category")

    # ========================
    # Loại bỏ trùng, thiếu, logic sai
    # ========================
    df.drop_duplicates(subset=["USERID", "ORDERID", "ITEMID"], inplace=True)
    essential_cols = ["USERID", "ORDERID", "DATE_", "TOTALBASKET", "ITEMID", "TOTALPRICE"]
    df.dropna(subset=essential_cols, inplace=True)
    df = df[(df["TOTALPRICE"] > 0) & (df["TOTALBASKET"] > 0)]
    df = df[df["TOTALPRICE"] <= df["TOTALBASKET"]]
    today = pd.Timestamp.today()
    df = df[(df["DATE_"] > "2000-01-01") & (df["DATE_"] <= today)]

    # ========================
    # Winsorizing
    # ========================
    for col in ["TOTALPRICE", "TOTALBASKET"]:
        lower, upper = df[col].quantile([0.01, 0.99])
        df[col] = np.clip(df[col], lower, upper)

    # Lọc khách hàng có đơn hợp lệ
    df = df.groupby("USERID").filter(lambda x: x["TOTALPRICE"].sum() > 0)

    return df


def load_and_clean_data_polars(base_path="/kaggle/input/3a-superstore"):
    """
    Load dữ liệu bằng Polars (lazy) và làm sạch.
    """
    # ========================
    # Load dữ liệu
    # ========================
    orders = (
        pl.scan_csv(f"{base_path}/Orders.csv", try_parse_dates=True)
        .select(["USERID", "ORDERID", "BRANCH_ID", "TOTALBASKET", "DATE_"])
    )

    details = (
        pl.scan_csv(f"{base_path}/Order_Details.csv")
        .select(["ORDERID", "ITEMID", "TOTALPRICE"])
        .unique()
    )

    categories = (
        pl.read_csv(f"{base_path}/Categories_ENG.csv", separator=';')
        .select(["ITEMID", "CATEGORY1", "ITEMNAME"])
        .unique(subset=["ITEMID"])
    )

    branches = (
        pl.read_csv(f"{base_path}/Branches_ENG.csv", separator=';')
        .select(["BRANCH_ID", "REGION"])
        .unique(subset=["BRANCH_ID"])
    )

    # ========================
    # Merge dữ liệu
    # ========================
    merged = (
        orders
        .join(details, on="ORDERID", how="left")
        .join(categories.lazy(), on="ITEMID", how="left")
        .join(branches.lazy(), on="BRANCH_ID", how="left")
        .select(["USERID", "ORDERID", "ITEMID", "TOTALPRICE", "TOTALBASKET",
                 "CATEGORY1","ITEMNAME", "REGION", "DATE_"])
        .collect(streaming=True)
    )

    # ========================
    # Chuyển kiểu dữ liệu
    # ========================
    merged = (
        merged.with_columns([
            pl.col("TOTALPRICE").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64),
            pl.col("TOTALBASKET").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64),
            pl.col("REGION").fill_null("Unknown")
        ])
    )

    # ========================
    # Winsorizing nhẹ
    # ========================
    q = merged.select([
        pl.col("TOTALPRICE").quantile(0.01).alias("p1_price"),
        pl.col("TOTALPRICE").quantile(0.99).alias("p99_price"),
        pl.col("TOTALBASKET").quantile(0.01).alias("p1_basket"),
        pl.col("TOTALBASKET").quantile(0.99).alias("p99_basket"),
    ]).to_dicts()[0]

    merged = merged.filter(
        (pl.col("TOTALPRICE").is_between(q["p1_price"], q["p99_price"])) &
        (pl.col("TOTALBASKET").is_between(q["p1_basket"], q["p99_basket"]))
    ).filter(
        (pl.col("TOTALPRICE") > 0) & (pl.col("TOTALBASKET") > 0)
    )

    # ========================
    # Chuẩn hóa text
    # ========================
    merged = merged.with_columns([
        pl.col("REGION").str.strip_chars().str.to_lowercase(),
        pl.col("CATEGORY1").str.strip_chars().str.to_lowercase(),
        pl.col("ITEMNAME").str.strip_chars().str.to_titlecase()
    ])

    return merged
