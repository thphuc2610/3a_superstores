# eda_polars.py
import plotly.express as px
import polars as pl

def add_time_features(df):
    df = df.lazy().with_columns([
        pl.col("DATE_").dt.year().alias("YEAR"),
        pl.col("DATE_").dt.month().alias("MONTH"),
        pl.col("DATE_").dt.weekday().alias("WEEKDAY")
    ])
    return df.collect()

def plot_revenue_over_time(df_pd):
    # Doanh thu theo tháng
    monthly_revenue = df_pd.groupby(["YEAR", "MONTH"])["TOTALPRICE"].sum().reset_index()
    fig = px.line(monthly_revenue, x="MONTH", y="TOTALPRICE", color="YEAR",
                  title="Doanh thu theo tháng theo từng năm", markers=True)
    fig.show()

    # Doanh thu theo ngày
    daily_revenue = df_pd.groupby("DATE_")["TOTALPRICE"].sum().reset_index()
    fig2 = px.line(daily_revenue, x="DATE_", y="TOTALPRICE", title="Doanh thu theo ngày")
    fig2.show()

    # Top 5 REGION
    region_revenue = df_pd.groupby("REGION")["TOTALPRICE"].sum().reset_index().sort_values("TOTALPRICE", ascending=False).head(5)
    fig3 = px.bar(region_revenue, x="REGION", y="TOTALPRICE", title="Top 5 REGION theo doanh thu", text="TOTALPRICE")
    fig3.show()

    # Top 5 CATEGORY1
    category_revenue = df_pd.groupby("CATEGORY1")["TOTALPRICE"].sum().reset_index().sort_values("TOTALPRICE", ascending=False).head(5)
    fig4 = px.bar(category_revenue, x="CATEGORY1", y="TOTALPRICE", title="Top 5 CATEGORY1 theo doanh thu", text="TOTALPRICE")
    fig4.show()

    # Top 5 sản phẩm
    product_revenue = df_pd.groupby("ITEMNAME")["TOTALPRICE"].sum().reset_index().sort_values("TOTALPRICE", ascending=False).head(5)
    fig5 = px.bar(product_revenue, x="ITEMNAME", y="TOTALPRICE", title="Top 5 sản phẩm theo doanh thu", text="TOTALPRICE")
    fig5.show()

    return monthly_revenue, region_revenue, category_revenue, product_revenue
