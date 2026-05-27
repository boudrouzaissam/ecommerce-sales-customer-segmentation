import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Sales & Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx", engine="openpyxl")
    return df


df_raw = load_data()


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

df = df_raw.copy()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df["CustomerID"] = df["CustomerID"].astype("Int64")

# Keep completed purchase transactions
df = df.dropna(subset=["CustomerID", "InvoiceDate", "Description"])
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Revenue
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Time variables
df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["InvoiceDateOnly"] = df["InvoiceDate"].dt.date
df["InvoiceDay"] = df["InvoiceDate"].dt.day_name()
df["InvoiceHour"] = df["InvoiceDate"].dt.hour

# Order-level table
order_df = df.groupby("InvoiceNo").agg(
    Order_Revenue=("Revenue", "sum"),
    Order_Quantity=("Quantity", "sum"),
    CustomerID=("CustomerID", "first"),
    Country=("Country", "first"),
    InvoiceDate=("InvoiceDate", "first")
).reset_index()

order_df["Average_Item_Value"] = order_df["Order_Revenue"] / order_df["Order_Quantity"]


# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------

st.sidebar.title("Filters")

country_options = sorted(df["Country"].dropna().unique())

selected_countries = st.sidebar.multiselect(
    "Country",
    options=country_options,
    default=country_options
)

if len(selected_countries) == 0:
    selected_countries = country_options

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

date_range = st.sidebar.date_input(
    "Invoice date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

df_filtered = df[
    (df["Country"].isin(selected_countries))
    & (df["InvoiceDate"].dt.date >= start_date)
    & (df["InvoiceDate"].dt.date <= end_date)
].copy()

order_filtered = order_df[
    (order_df["Country"].isin(selected_countries))
    & (order_df["InvoiceDate"].dt.date >= start_date)
    & (order_df["InvoiceDate"].dt.date <= end_date)
].copy()

if df_filtered.empty:
    st.error("No observations are available with the selected filters. Please adjust the filters.")
    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("E-Commerce Sales & Customer Segmentation Dashboard")
st.subheader("Sales Performance, Product Analysis, Market Analysis and RFM Segmentation")

st.markdown("""
This project analyzes the **Online Retail** dataset from the **UCI Machine Learning Repository**.

The dataset contains transaction-level sales data for a **UK-based non-store online retail company**
between December 2010 and December 2011. The company mainly sells unique all-occasion gift products,
and many customers are wholesalers.

Because the retailer is based in the United Kingdom, the UK is expected to dominate the dataset.
The country analysis should therefore be interpreted as a company-level market structure, not as a global
ranking of e-commerce markets.
""")


# --------------------------------------------------
# Key indicators
# --------------------------------------------------

st.markdown("---")
st.subheader("Key Sales Indicators")

total_revenue = df_filtered["Revenue"].sum()
total_orders = df_filtered["InvoiceNo"].nunique()
total_customers = df_filtered["CustomerID"].nunique()
total_products = df_filtered["StockCode"].nunique()
total_quantity = df_filtered["Quantity"].sum()
average_order_value = total_revenue / total_orders if total_orders > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Orders", f"{total_orders:,}")
col3.metric("Customers", f"{total_customers:,}")
col4.metric("Products", f"{total_products:,}")
col5.metric("Average Order Value", f"${average_order_value:,.2f}")

col1, col2, col3 = st.columns(3)

col1.metric("Total Quantity Sold", f"{total_quantity:,.0f}")
col2.metric("Countries", f"{df_filtered['Country'].nunique():,}")
col3.metric("Transaction Lines", f"{len(df_filtered):,}")


# --------------------------------------------------
# Research design
# --------------------------------------------------

st.markdown("---")
st.subheader("Research Design")

st.markdown("""
This e-commerce analytics project is structured around seven main questions:

1. **How are overall sales performing?**
2. **How do sales evolve over time?**
3. **Which products generate the most revenue?**
4. **Which countries are the most important markets?**
5. **Who are the most valuable customers?**
6. **Can customers be segmented using RFM analysis?**
7. **Can customers be segmented using K-means clustering?**
""")


# --------------------------------------------------
# Dataset overview
# --------------------------------------------------

st.markdown("---")
st.subheader("Dataset Overview")

st.markdown("""
The dataset contains invoice, product, quantity, date, price, customer, and country information.

The cleaning process removes missing customer IDs, cancellations, negative quantities, and non-positive prices
to focus on completed purchase transactions.
""")

with st.expander("View first rows of the cleaned dataset"):
    st.dataframe(df_filtered.head(100))

with st.expander("View variable types"):
    st.write(df_filtered.dtypes)

dataset_summary = pd.DataFrame({
    "Indicator": [
        "Raw transaction lines",
        "Cleaned transaction lines after filters",
        "Unique invoices",
        "Unique customers",
        "Unique products",
        "Countries",
        "Start date",
        "End date"
    ],
    "Value": [
        f"{len(df_raw):,}",
        f"{len(df_filtered):,}",
        f"{df_filtered['InvoiceNo'].nunique():,}",
        f"{df_filtered['CustomerID'].nunique():,}",
        f"{df_filtered['StockCode'].nunique():,}",
        f"{df_filtered['Country'].nunique():,}",
        str(df_filtered["InvoiceDate"].min().date()),
        str(df_filtered["InvoiceDate"].max().date())
    ]
})

st.markdown("### Results: Dataset Summary")
st.dataframe(dataset_summary)


# --------------------------------------------------
# Question 1
# --------------------------------------------------

st.markdown("---")
st.header("1. How are overall sales performing?")

st.markdown("""
### Research Question
How are overall sales performing in terms of revenue, orders, customers, products, and order size?

### Objective
The objective is to summarize the commercial performance of the online retailer.

### Method
We calculate sales KPIs and analyze order revenue distribution, order quantity distribution,
revenue by hour, and revenue by weekday.
""")

sales_summary = pd.DataFrame({
    "Indicator": [
        "Total Revenue",
        "Number of Orders",
        "Number of Customers",
        "Number of Products",
        "Total Quantity Sold",
        "Average Order Value"
    ],
    "Value": [
        total_revenue,
        total_orders,
        total_customers,
        total_products,
        total_quantity,
        average_order_value
    ]
})

st.markdown("### Results: Sales Performance Summary")
st.dataframe(sales_summary)

# Order-level distributions
top_orders_revenue = order_filtered.sort_values("Order_Revenue", ascending=False).head(15)
top_orders_quantity = order_filtered.sort_values("Order_Quantity", ascending=False).head(15)

order_hist = order_filtered[
    (order_filtered["Order_Revenue"] <= order_filtered["Order_Revenue"].quantile(0.99))
    & (order_filtered["Order_Quantity"] <= order_filtered["Order_Quantity"].quantile(0.99))
].copy()

col1, col2 = st.columns(2)

with col1:
    fig_order_revenue_hist = px.histogram(
        order_hist,
        x="Order_Revenue",
        nbins=40,
        title="Distribution of Order Revenue",
        labels={"Order_Revenue": "Order Revenue"}
    )
    fig_order_revenue_hist.update_layout(
        yaxis_title="Number of Orders",
        xaxis_title="Order Revenue"
    )
    st.plotly_chart(fig_order_revenue_hist, use_container_width=True)

with col2:
    fig_order_quantity_hist = px.histogram(
        order_hist,
        x="Order_Quantity",
        nbins=40,
        title="Distribution of Order Quantity",
        labels={"Order_Quantity": "Order Quantity"}
    )
    fig_order_quantity_hist.update_layout(
        yaxis_title="Number of Orders",
        xaxis_title="Order Quantity"
    )
    st.plotly_chart(fig_order_quantity_hist, use_container_width=True)

top_order_id = top_orders_revenue.iloc[0]["InvoiceNo"]
top_order_revenue = top_orders_revenue.iloc[0]["Order_Revenue"]
top_order_country = top_orders_revenue.iloc[0]["Country"]

top_quantity_order_id = top_orders_quantity.iloc[0]["InvoiceNo"]
top_quantity_value = top_orders_quantity.iloc[0]["Order_Quantity"]
top_quantity_country = top_orders_quantity.iloc[0]["Country"]

col1, col2 = st.columns(2)

with col1:
    revenue_by_hour = df_filtered.groupby("InvoiceHour").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique")
    ).reset_index()

    best_hour_row = revenue_by_hour.sort_values("Revenue", ascending=False).iloc[0]
    best_hour = int(best_hour_row["InvoiceHour"])
    best_hour_revenue = best_hour_row["Revenue"]

    fig_hour = px.bar(
        revenue_by_hour,
        x="InvoiceHour",
        y="Revenue",
        title="Revenue by Hour of the Day"
    )

    fig_hour.add_scatter(
        x=[best_hour],
        y=[best_hour_revenue],
        mode="markers+text",
        text=[f"Peak: {best_hour}:00"],
        textposition="top center",
        marker=dict(size=16)
    )

    st.plotly_chart(fig_hour, use_container_width=True)

with col2:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    revenue_by_day = df_filtered.groupby("InvoiceDay").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique")
    ).reset_index()

    revenue_by_day["InvoiceDay"] = pd.Categorical(
        revenue_by_day["InvoiceDay"],
        categories=day_order,
        ordered=True
    )

    revenue_by_day = revenue_by_day.sort_values("InvoiceDay")

    best_day_row = revenue_by_day.sort_values("Revenue", ascending=False).iloc[0]
    best_day = best_day_row["InvoiceDay"]
    best_day_revenue = best_day_row["Revenue"]

    fig_day = px.bar(
        revenue_by_day,
        x="InvoiceDay",
        y="Revenue",
        title="Revenue by Day of the Week"
    )

    fig_day.add_scatter(
        x=[best_day],
        y=[best_day_revenue],
        mode="markers+text",
        text=[f"Peak: {best_day}"],
        textposition="top center",
        marker=dict(size=16)
    )

    st.plotly_chart(fig_day, use_container_width=True)

st.markdown(f"""
### Interpretation

The retailer generated **${total_revenue:,.0f}** from **{total_orders:,} orders** and **{total_customers:,} customers**.
The average order value is **${average_order_value:,.2f}**.

The order revenue distribution shows how invoice values are spread across the business. Most orders are concentrated
in lower revenue ranges, while a smaller number of large orders appear in the upper tail.

The largest order is invoice **{top_order_id}**, from **{top_order_country}**, with revenue of **${top_order_revenue:,.0f}**.
The largest order by quantity is invoice **{top_quantity_order_id}**, from **{top_quantity_country}**, with **{top_quantity_value:,.0f} units**.

These large orders may represent wholesale purchases. Since the company has many wholesale customers,
a small number of large invoices can strongly affect revenue performance.

The strongest revenue hour is around **{best_hour}:00**, generating **${best_hour_revenue:,.0f}**.
This concentration may reflect business-hour ordering behavior, especially if many customers are wholesalers.

The strongest revenue day is **{best_day}**, generating **${best_day_revenue:,.0f}**.
This may reflect weekly ordering routines, inventory restocking cycles, or operational patterns in the retailer's order processing.
""")


# --------------------------------------------------
# Question 2
# --------------------------------------------------

st.markdown("---")
st.header("2. How do sales evolve over time?")

st.markdown("""
### Research Question
How do revenue, orders, and customer activity evolve over time?

### Objective
The objective is to identify sales trends, seasonality, and periods of high commercial activity.

### Method
We aggregate revenue, orders, customers, and average order value by month.
""")

monthly_sales = df_filtered.groupby("InvoiceMonth").agg(
    Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Customers=("CustomerID", "nunique")
).reset_index()

monthly_sales["Average_Order_Value"] = monthly_sales["Revenue"] / monthly_sales["Orders"]

st.markdown("### Results: Monthly Sales")
st.dataframe(monthly_sales)

best_month_row = monthly_sales.sort_values("Revenue", ascending=False).iloc[0]
best_month = best_month_row["InvoiceMonth"]
best_month_revenue = best_month_row["Revenue"]

fig_monthly_revenue = px.line(
    monthly_sales,
    x="InvoiceMonth",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend"
)

fig_monthly_revenue.add_scatter(
    x=[best_month],
    y=[best_month_revenue],
    mode="markers+text",
    text=[f"Peak: {best_month}"],
    textposition="top center",
    marker=dict(size=16)
)

st.plotly_chart(fig_monthly_revenue, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    best_order_month_row = monthly_sales.sort_values("Orders", ascending=False).iloc[0]
    best_order_month = best_order_month_row["InvoiceMonth"]
    best_order_value = best_order_month_row["Orders"]

    fig_monthly_orders = px.line(
        monthly_sales,
        x="InvoiceMonth",
        y="Orders",
        markers=True,
        title="Monthly Orders Trend"
    )

    fig_monthly_orders.add_scatter(
        x=[best_order_month],
        y=[best_order_value],
        mode="markers+text",
        text=[f"Peak: {best_order_month}"],
        textposition="top center",
        marker=dict(size=16)
    )

    st.plotly_chart(fig_monthly_orders, use_container_width=True)

with col2:
    best_customer_month_row = monthly_sales.sort_values("Customers", ascending=False).iloc[0]
    best_customer_month = best_customer_month_row["InvoiceMonth"]
    best_customer_value = best_customer_month_row["Customers"]

    fig_monthly_customers = px.line(
        monthly_sales,
        x="InvoiceMonth",
        y="Customers",
        markers=True,
        title="Monthly Active Customers"
    )

    fig_monthly_customers.add_scatter(
        x=[best_customer_month],
        y=[best_customer_value],
        mode="markers+text",
        text=[f"Peak: {best_customer_month}"],
        textposition="top center",
        marker=dict(size=16)
    )

    st.plotly_chart(fig_monthly_customers, use_container_width=True)

fig_monthly_aov = px.line(
    monthly_sales,
    x="InvoiceMonth",
    y="Average_Order_Value",
    markers=True,
    title="Monthly Average Order Value"
)
st.plotly_chart(fig_monthly_aov, use_container_width=True)

st.markdown(f"""
### Interpretation

The highest revenue month is **{best_month}**, with total revenue of **${best_month_revenue:,.0f}**.
The highest order volume occurs in **{best_order_month}**, while the highest number of active customers occurs in **{best_customer_month}**.

For a gift-products retailer, this seasonal pattern can be linked to end-of-year demand, Christmas preparation,
holiday gifting, promotional campaigns, or wholesale restocking before peak retail periods.

If revenue and orders rise together, the increase is likely driven by stronger demand.
If revenue rises more than orders, it may indicate larger order sizes, higher-value products, or more wholesale activity.
""")


# --------------------------------------------------
# Question 3
# --------------------------------------------------

st.markdown("---")
st.header("3. Which products generate the most revenue?")

st.markdown("""
### Research Question
Which products contribute the most to sales performance?

### Objective
The objective is to identify products that generate the highest revenue and products sold in the largest quantities.

### Method
We aggregate transactions by product description and calculate total revenue, quantity sold, number of orders,
and average unit price.
""")

product_summary = df_filtered.groupby("Description").agg(
    Revenue=("Revenue", "sum"),
    Quantity=("Quantity", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Average_UnitPrice=("UnitPrice", "mean")
).reset_index()

product_summary["Revenue_Share"] = product_summary["Revenue"] / product_summary["Revenue"].sum()

top_revenue_products = product_summary.sort_values("Revenue", ascending=False).head(15)
top_quantity_products = product_summary.sort_values("Quantity", ascending=False).head(15)

st.markdown("### Results: Top Products by Revenue")
st.dataframe(top_revenue_products)

fig_top_revenue_products = px.bar(
    top_revenue_products.sort_values("Revenue", ascending=True),
    x="Revenue",
    y="Description",
    orientation="h",
    title="Top 15 Products by Revenue"
)
st.plotly_chart(fig_top_revenue_products, use_container_width=True)

st.markdown("### Results: Top Products by Quantity Sold")
st.dataframe(top_quantity_products)

fig_top_quantity_products = px.bar(
    top_quantity_products.sort_values("Quantity", ascending=True),
    x="Quantity",
    y="Description",
    orientation="h",
    title="Top 15 Products by Quantity Sold"
)
st.plotly_chart(fig_top_quantity_products, use_container_width=True)

top_30_products = product_summary.sort_values("Revenue", ascending=False).head(30)

fig_product_treemap = px.treemap(
    top_30_products,
    path=["Description"],
    values="Revenue",
    color="Revenue",
    title="Product Revenue Treemap: Top 30 Products"
)
st.plotly_chart(fig_product_treemap, use_container_width=True)

fig_product_quantity_revenue = px.scatter(
    top_30_products,
    x="Quantity",
    y="Revenue",
    size="Average_UnitPrice",
    size_max=55,
    hover_name="Description",
    title="Top Products: Quantity Sold vs Revenue"
)
st.plotly_chart(fig_product_quantity_revenue, use_container_width=True)

top_product = top_revenue_products.iloc[0]
top_product_name = top_product["Description"]
top_product_revenue = top_product["Revenue"]
top_product_quantity = top_product["Quantity"]
top_product_share = top_product["Revenue_Share"] * 100

top_quantity_product = top_quantity_products.iloc[0]
top_quantity_product_name = top_quantity_product["Description"]
top_quantity_product_quantity = top_quantity_product["Quantity"]
top_quantity_product_revenue = top_quantity_product["Revenue"]

top5_product_share = product_summary.sort_values("Revenue", ascending=False).head(5)["Revenue"].sum() / product_summary["Revenue"].sum() * 100

st.markdown(f"""
### Interpretation

The strongest product by revenue is **{top_product_name}**, generating **${top_product_revenue:,.0f}**.
It sold **{top_product_quantity:,.0f} units** and represents approximately **{top_product_share:.1f}%** of total revenue.

The product with the highest quantity sold is **{top_quantity_product_name}**, with **{top_quantity_product_quantity:,.0f} units** sold
and revenue of **${top_quantity_product_revenue:,.0f}**.

This distinction is important: **{top_product_name}** is the strongest revenue product, while **{top_quantity_product_name}**
is the strongest volume product. The first one matters more for revenue concentration, while the second one matters more
for inventory movement and operational planning.

The top five products generate approximately **{top5_product_share:.1f}%** of total revenue.
If this share is high, the retailer depends strongly on a small group of products. This can be useful for focused promotion,
but it also creates risk if demand for these specific products declines.
""")


# --------------------------------------------------
# Question 4
# --------------------------------------------------

st.markdown("---")
st.header("4. Which countries are the most important markets?")

st.markdown("""
### Research Question
Which countries contribute most to revenue, orders, and customer base?

### Objective
The objective is to identify the most important geographic markets for the e-commerce business.

### Method
We aggregate revenue, orders, customers, quantity, average order value, and average revenue per customer by country.
""")

country_summary = df_filtered.groupby("Country").agg(
    Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Customers=("CustomerID", "nunique"),
    Quantity=("Quantity", "sum")
).reset_index()

country_summary["Average_Revenue_Per_Customer"] = country_summary["Revenue"] / country_summary["Customers"]
country_summary["Average_Order_Value"] = country_summary["Revenue"] / country_summary["Orders"]
country_summary["Revenue_Share"] = country_summary["Revenue"] / country_summary["Revenue"].sum()

country_summary = country_summary.sort_values("Revenue", ascending=False)

st.markdown("### Results: Sales by Country")
st.dataframe(country_summary)

top_countries = country_summary.head(15)

fig_country_revenue = px.bar(
    top_countries.sort_values("Revenue", ascending=True),
    x="Revenue",
    y="Country",
    orientation="h",
    title="Top Countries by Revenue"
)
st.plotly_chart(fig_country_revenue, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_country_customers = px.bar(
        top_countries.sort_values("Customers", ascending=True),
        x="Customers",
        y="Country",
        orientation="h",
        title="Top Countries by Number of Customers"
    )
    st.plotly_chart(fig_country_customers, use_container_width=True)

with col2:
    fig_country_aov = px.bar(
        top_countries.sort_values("Average_Order_Value", ascending=True),
        x="Average_Order_Value",
        y="Country",
        orientation="h",
        title="Top Countries by Average Order Value"
    )
    st.plotly_chart(fig_country_aov, use_container_width=True)

fig_country_treemap = px.treemap(
    top_countries,
    path=["Country"],
    values="Revenue",
    color="Revenue",
    title="Country Revenue Treemap: Top Markets"
)
st.plotly_chart(fig_country_treemap, use_container_width=True)

top_country = country_summary.iloc[0]
top_country_name = top_country["Country"]
top_country_revenue = top_country["Revenue"]
top_country_share = top_country["Revenue_Share"] * 100

non_uk = country_summary[country_summary["Country"] != "United Kingdom"]

if not non_uk.empty:
    top_non_uk = non_uk.iloc[0]
    top_non_uk_name = top_non_uk["Country"]
    top_non_uk_revenue = top_non_uk["Revenue"]
else:
    top_non_uk_name = "N/A"
    top_non_uk_revenue = 0

premium_country = country_summary[country_summary["Customers"] >= 3].sort_values(
    "Average_Order_Value",
    ascending=False
).iloc[0]

premium_country_name = premium_country["Country"]
premium_country_aov = premium_country["Average_Order_Value"]

st.markdown(f"""
### Interpretation

The strongest country by revenue is **{top_country_name}**, generating **${top_country_revenue:,.0f}**,
or approximately **{top_country_share:.1f}%** of total revenue.

This result should be read carefully. Since the retailer is UK-based, the dominance of **{top_country_name}**
mainly reflects the company's home-market concentration.

Outside the UK, the strongest revenue market is **{top_non_uk_name}**, with revenue of **${top_non_uk_revenue:,.0f}**.
This country is more useful for evaluating international demand.

The country with the highest average order value among markets with at least three customers is **{premium_country_name}**,
with an average order value of **${premium_country_aov:,.2f}**.
This may indicate a smaller but higher-value market, possibly driven by wholesale or larger basket purchases.
""")


# --------------------------------------------------
# Question 5
# --------------------------------------------------

st.markdown("---")
st.header("5. Who are the most valuable customers?")

st.markdown("""
### Research Question
Which customers generate the highest revenue?

### Objective
The objective is to identify high-value customers and understand their purchasing behavior.

### Method
We aggregate transactions by customer and calculate total revenue, number of orders,
number of products purchased, total quantity, and average order value.
""")

customer_summary = df_filtered.groupby("CustomerID").agg(
    Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Quantity=("Quantity", "sum"),
    Products=("StockCode", "nunique"),
    Country=("Country", "first")
).reset_index()

customer_summary["Average_Order_Value"] = customer_summary["Revenue"] / customer_summary["Orders"]
customer_summary["Revenue_Share"] = customer_summary["Revenue"] / customer_summary["Revenue"].sum()

top_customers = customer_summary.sort_values("Revenue", ascending=False).head(15)

st.markdown("### Results: Top Customers by Revenue")
st.dataframe(top_customers)

customer_hist = customer_summary[
    (customer_summary["Revenue"] <= customer_summary["Revenue"].quantile(0.99))
    & (customer_summary["Orders"] <= customer_summary["Orders"].quantile(0.99))
].copy()

col1, col2 = st.columns(2)

with col1:
    fig_customer_revenue_hist = px.histogram(
        customer_hist,
        x="Revenue",
        nbins=40,
        title="Distribution of Customer Revenue",
        labels={"Revenue": "Customer Revenue"}
    )
    fig_customer_revenue_hist.update_layout(
        yaxis_title="Number of Customers",
        xaxis_title="Customer Revenue"
    )
    st.plotly_chart(fig_customer_revenue_hist, use_container_width=True)

with col2:
    fig_customer_orders_hist = px.histogram(
        customer_hist,
        x="Orders",
        nbins=40,
        title="Distribution of Orders per Customer",
        labels={"Orders": "Orders per Customer"}
    )
    fig_customer_orders_hist.update_layout(
        yaxis_title="Number of Customers",
        xaxis_title="Orders per Customer"
    )
    st.plotly_chart(fig_customer_orders_hist, use_container_width=True)

fig_customer_scatter = px.scatter(
    customer_hist,
    x="Orders",
    y="Revenue",
    size="Average_Order_Value",
    size_max=60,
    color="Country",
    hover_data=["CustomerID", "Products", "Average_Order_Value"],
    title="Customer Value: Orders vs Revenue"
)
st.plotly_chart(fig_customer_scatter, use_container_width=True)

customer_top20_share = customer_summary.sort_values("Revenue", ascending=False).head(20)["Revenue"].sum() / customer_summary["Revenue"].sum() * 100

top_customer = top_customers.iloc[0]
top_customer_id = top_customer["CustomerID"]
top_customer_revenue = top_customer["Revenue"]
top_customer_orders = top_customer["Orders"]
top_customer_country = top_customer["Country"]
top_customer_share = top_customer["Revenue_Share"] * 100

st.markdown(f"""
### Interpretation

The highest-value customer is **{top_customer_id}** from **{top_customer_country}**.
This customer generated **${top_customer_revenue:,.0f}** across **{top_customer_orders:,} orders**,
representing approximately **{top_customer_share:.1f}%** of total revenue.

The customer revenue distribution shows that most customers generate relatively modest revenue,
while a smaller group of customers generates much higher value. This is typical in e-commerce and wholesale-oriented businesses.

The top 20 customers generate approximately **{customer_top20_share:.1f}%** of total revenue.
This indicates whether the business is highly dependent on a small group of customers.

In the scatter plot, customers with many orders and high revenue are loyal high-value customers.
Customers with fewer orders but high revenue may be wholesale or large-basket buyers.
These customers should be treated differently: frequent buyers need loyalty management, while large-basket buyers may need
personalized account management or volume-based offers.
""")


# --------------------------------------------------
# Question 6: RFM Analysis
# --------------------------------------------------

st.markdown("---")
st.header("6. Can customers be segmented using RFM analysis?")

st.markdown("""
### Research Question
Can customers be segmented based on recency, frequency, and monetary value?

### Objective
The objective is to classify customers based on their purchasing behavior and identify meaningful customer segments.

### Method
RFM analysis uses three dimensions:

- **Recency:** how recently the customer purchased. Lower recency means the customer bought more recently.
- **Frequency:** how often the customer purchased. Higher frequency means stronger repeat-purchase behavior.
- **Monetary:** how much revenue the customer generated. Higher monetary value means stronger customer value.

Each customer receives a score from 1 to 5 for each dimension.
""")

snapshot_date = df_filtered["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df_filtered.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

rfm["R_Score"] = pd.qcut(
    rfm["Recency"].rank(method="first"),
    5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["M_Score"] = pd.qcut(
    rfm["Monetary"].rank(method="first"),
    5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]


def assign_rfm_segment(row):
    if row["R_Score"] >= 4 and row["F_Score"] >= 4 and row["M_Score"] >= 4:
        return "Champions"
    elif row["F_Score"] >= 4 and row["M_Score"] >= 4:
        return "Loyal High-Value"
    elif row["R_Score"] >= 4 and row["F_Score"] <= 2:
        return "New or Recent"
    elif row["R_Score"] <= 2 and row["F_Score"] >= 4:
        return "At Risk Loyal"
    elif row["M_Score"] >= 4:
        return "Big Spenders"
    elif row["R_Score"] <= 2 and row["F_Score"] <= 2:
        return "Inactive"
    else:
        return "Regular Customers"


rfm["RFM_Segment"] = rfm.apply(assign_rfm_segment, axis=1)

st.markdown("### Results: RFM Customer Table")
st.dataframe(rfm.head(100))

rfm_segment_summary = rfm.groupby("RFM_Segment").agg(
    Customers=("CustomerID", "count"),
    Average_Recency=("Recency", "mean"),
    Average_Frequency=("Frequency", "mean"),
    Average_Monetary=("Monetary", "mean"),
    Total_Revenue=("Monetary", "sum")
).reset_index().sort_values("Total_Revenue", ascending=False)

st.markdown("### Results: RFM Segment Summary")
st.dataframe(rfm_segment_summary)

fig_rfm_segments = px.bar(
    rfm_segment_summary.sort_values("Customers", ascending=True),
    x="Customers",
    y="RFM_Segment",
    orientation="h",
    title="Number of Customers by RFM Segment"
)
st.plotly_chart(fig_rfm_segments, use_container_width=True)

fig_rfm_revenue = px.bar(
    rfm_segment_summary.sort_values("Total_Revenue", ascending=True),
    x="Total_Revenue",
    y="RFM_Segment",
    orientation="h",
    title="Total Revenue by RFM Segment"
)
st.plotly_chart(fig_rfm_revenue, use_container_width=True)

rfm_visual = rfm[
    (rfm["Monetary"] <= rfm["Monetary"].quantile(0.99))
    & (rfm["Frequency"] <= rfm["Frequency"].quantile(0.99))
].copy()

fig_rfm_scatter = px.scatter(
    rfm_visual,
    x="Frequency",
    y="Monetary",
    color="RFM_Segment",
    size="Monetary",
    size_max=60,
    hover_data=["CustomerID", "Recency", "RFM_Score"],
    title="RFM Segments: Frequency vs Monetary Value"
)
st.plotly_chart(fig_rfm_scatter, use_container_width=True)

top_rfm_segment = rfm_segment_summary.iloc[0]
top_rfm_segment_name = top_rfm_segment["RFM_Segment"]
top_rfm_revenue = top_rfm_segment["Total_Revenue"]
top_rfm_customers = top_rfm_segment["Customers"]

largest_segment = rfm_segment_summary.sort_values("Customers", ascending=False).iloc[0]
largest_segment_name = largest_segment["RFM_Segment"]
largest_segment_customers = largest_segment["Customers"]

inactive_segment = rfm_segment_summary[rfm_segment_summary["RFM_Segment"] == "Inactive"]

if not inactive_segment.empty:
    inactive_customers = int(inactive_segment.iloc[0]["Customers"])
    inactive_revenue = inactive_segment.iloc[0]["Total_Revenue"]
else:
    inactive_customers = 0
    inactive_revenue = 0

st.markdown(f"""
### Interpretation

The segment generating the highest revenue is **{top_rfm_segment_name}**, with **{top_rfm_customers:,} customers**
and total revenue of **${top_rfm_revenue:,.0f}**.

The largest segment by number of customers is **{largest_segment_name}**, with **{largest_segment_customers:,} customers**.
This distinction matters because the largest segment is not always the most profitable segment.

The inactive segment contains **{inactive_customers:,} customers** and generated **${inactive_revenue:,.0f}**.
If this group is large, the retailer may have a reactivation opportunity.

RFM translates sales history into marketing action:

- **Champions** should receive loyalty rewards and early access to new products.
- **Loyal High-Value** customers should receive personalized offers and retention attention.
- **Big Spenders** may respond well to premium bundles or volume incentives.
- **At Risk Loyal** customers should receive reactivation messages before they become inactive.
- **New or Recent** customers should receive second-purchase offers.
- **Inactive** customers should be targeted only if the expected reactivation value exceeds the campaign cost.
""")


# --------------------------------------------------
# Question 7: K-means segmentation
# --------------------------------------------------

st.markdown("---")
st.header("7. Can customers be segmented using K-means clustering?")

st.markdown("""
### Research Question
Can customers be grouped into data-driven segments based on their purchasing behavior?

### Objective
The objective is to identify customer groups automatically using unsupervised machine learning.

### Method
K-means clustering groups customers based on similarity across three behavioral indicators:

- **Recency:** how many days have passed since the customer’s last purchase.
- **Frequency:** how many times the customer purchased.
- **Monetary:** how much revenue the customer generated.

Before applying K-means, the variables are standardized because they are measured on different scales:
Monetary is measured in currency, Frequency is measured in number of orders, and Recency is measured in days.

The algorithm creates four clusters. Customers inside the same cluster have similar purchasing profiles.
Unlike RFM segmentation, K-means does not use predefined business labels. The clusters are interpreted after
looking at their average Recency, Frequency, and Monetary values.
""")

cluster_df = rfm[["Recency", "Frequency", "Monetary"]].copy()

if len(cluster_df) < 10:
    st.warning("Not enough customers to perform K-means clustering with the current filters.")
else:
    try:
        scaler = StandardScaler()
        cluster_scaled = scaler.fit_transform(cluster_df)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        rfm["Cluster"] = kmeans.fit_predict(cluster_scaled)

        cluster_summary = rfm.groupby("Cluster").agg(
            Customers=("CustomerID", "count"),
            Average_Recency=("Recency", "mean"),
            Average_Frequency=("Frequency", "mean"),
            Average_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum")
        ).reset_index()

        st.markdown("### Results: K-means Cluster Summary")
        st.dataframe(cluster_summary)

        cluster_visual = rfm[
            (rfm["Monetary"] <= rfm["Monetary"].quantile(0.99))
            & (rfm["Frequency"] <= rfm["Frequency"].quantile(0.99))
        ].copy()

        fig_cluster = px.scatter(
            cluster_visual,
            x="Frequency",
            y="Monetary",
            color="Cluster",
            size="Monetary",
            size_max=60,
            hover_data=["CustomerID", "Recency", "RFM_Segment"],
            title="Customer Clusters: Frequency vs Monetary Value"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        fig_cluster_recency = px.scatter(
            cluster_visual,
            x="Recency",
            y="Monetary",
            color="Cluster",
            size="Frequency",
            size_max=60,
            hover_data=["CustomerID", "RFM_Segment"],
            title="Customer Clusters: Recency vs Monetary Value"
        )
        st.plotly_chart(fig_cluster_recency, use_container_width=True)

        cluster_long = cluster_summary.melt(
            id_vars="Cluster",
            value_vars=[
                "Average_Recency",
                "Average_Frequency",
                "Average_Monetary"
            ],
            var_name="Indicator",
            value_name="Value"
        )

        fig_cluster_profile = px.bar(
            cluster_long,
            x="Indicator",
            y="Value",
            color="Cluster",
            barmode="group",
            title="Cluster Profiles Across RFM Indicators"
        )
        st.plotly_chart(fig_cluster_profile, use_container_width=True)

        best_cluster = cluster_summary.sort_values("Average_Monetary", ascending=False).iloc[0]
        inactive_cluster = cluster_summary.sort_values("Average_Recency", ascending=False).iloc[0]
        frequent_cluster = cluster_summary.sort_values("Average_Frequency", ascending=False).iloc[0]

        st.markdown(f"""
        ### Interpretation

        K-means creates customer groups directly from the data. Each cluster is interpreted by comparing its average
        **Recency**, **Frequency**, and **Monetary** values.

        - **Cluster {int(best_cluster["Cluster"])}** has the highest average monetary value:
          **${best_cluster["Average_Monetary"]:,.0f}** per customer.
          This is the most valuable cluster. These customers generate the highest revenue and should be prioritized
          for loyalty programs, premium offers, and personalized retention actions.

        - **Cluster {int(frequent_cluster["Cluster"])}** has the highest average purchase frequency:
          **{frequent_cluster["Average_Frequency"]:.1f} orders per customer**.
          These customers purchase repeatedly. They may not always be the biggest spenders, but they are commercially important
          because they show strong engagement and repeat-buying behavior.

        - **Cluster {int(inactive_cluster["Cluster"])}** has the highest average recency:
          **{inactive_cluster["Average_Recency"]:.1f} days since last purchase**.
          This means customers in this group have not purchased recently. They may represent an at-risk or inactive segment
          and could be targeted with reactivation campaigns.

        The advantage of K-means is that it detects patterns automatically instead of assigning customers to predefined categories.
        The limitation is that the clusters do not have business meaning by themselves. They need to be interpreted using the
        cluster averages.

        In this dashboard, **RFM** provides clear marketing labels, while **K-means** validates whether customers naturally form
        different behavioral groups. Used together, they provide a stronger segmentation strategy.
        """)

    except Exception as e:
        st.warning("K-means clustering could not be performed with the current filters.")
        st.write(e)


# --------------------------------------------------
# Managerial recommendations
# --------------------------------------------------

st.markdown("---")
st.header("Managerial Recommendations")

st.markdown("""
Based on the analysis, the retailer could act on four priorities:

1. **Protect the revenue engine.**  
   High-value customers and top revenue products deserve loyalty offers, priority service, and inventory protection.

2. **Turn product concentration into strategy.**  
   Products that dominate revenue should be promoted carefully, but the retailer should also monitor dependence on a narrow product portfolio.

3. **Treat the UK as the home market, not as a global benchmark.**  
   The UK dominates because the retailer is UK-based. International markets should be evaluated separately, especially countries with high order value.

4. **Use segmentation for action.**  
   Champions need loyalty rewards, at-risk loyal customers need reactivation, new customers need second-purchase incentives,
   and inactive customers need low-cost win-back campaigns.
""")


# --------------------------------------------------
# Dashboard-style conclusion
# --------------------------------------------------

st.markdown("---")
st.header("Executive Dashboard Conclusion")

top_product_final = top_revenue_products.iloc[0]["Description"]
top_product_final_revenue = top_revenue_products.iloc[0]["Revenue"]

top_country_final = country_summary.iloc[0]["Country"]
top_country_final_revenue = country_summary.iloc[0]["Revenue"]

top_customer_final = top_customers.iloc[0]["CustomerID"]
top_customer_final_revenue = top_customers.iloc[0]["Revenue"]

top_rfm_final = rfm_segment_summary.iloc[0]["RFM_Segment"]
top_rfm_final_revenue = rfm_segment_summary.iloc[0]["Total_Revenue"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Top Product", str(top_product_final)[:25], f"${top_product_final_revenue:,.0f}")
col2.metric("Top Market", str(top_country_final), f"${top_country_final_revenue:,.0f}")
col3.metric("Top Customer", str(top_customer_final), f"${top_customer_final_revenue:,.0f}")
col4.metric("Top RFM Segment", str(top_rfm_final), f"${top_rfm_final_revenue:,.0f}")

st.markdown("### Strategic Reading of the Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    #### 1. Revenue Engine

    The business is mainly driven by a combination of:

    - the top product: **{top_product_final}**;
    - the top market: **{top_country_final}**;
    - the top customer: **{top_customer_final}**;
    - the strongest behavioral segment: **{top_rfm_final}**.

    This means that performance is not evenly distributed. A limited number of products, customers,
    and markets carry an important part of total revenue.
    """)

with col2:
    st.markdown(f"""
    #### 2. Market Context

    The strongest market is **{top_country_final}**, generating **${top_country_final_revenue:,.0f}**.

    Since the retailer is based in the United Kingdom, UK dominance should be interpreted as a home-market effect.
    For international strategy, non-UK markets should be analyzed separately to identify expansion opportunities.
    """)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    #### 3. Customer Strategy

    The most important customer segment is **{top_rfm_final}**.

    This segment should receive priority attention because it generates the highest revenue.
    Depending on the segment profile, the business can design loyalty actions, premium offers,
    personalized communication, or reactivation campaigns.
    """)

with col2:
    st.markdown("""
    #### 4. Operational Strategy

    Monthly, daily, and hourly sales patterns help the company plan:

    - inventory;
    - campaign timing;
    - customer service workload;
    - order processing capacity;
    - seasonal promotions.

    The dashboard therefore supports both marketing and operational decisions.
    """)

st.markdown("### Final Managerial Message")

st.success("""
This project transforms raw transaction data into a business decision tool. 
It identifies where revenue comes from, which products matter most, which customers deserve priority,
which markets require attention, and how customer segmentation can guide marketing actions.
""")

st.markdown("### Portfolio Value")

st.markdown("""
This dashboard demonstrates a complete business analytics workflow:

- data cleaning;
- KPI design;
- sales trend analysis;
- product performance analysis;
- country-level market analysis;
- customer value measurement;
- RFM segmentation;
- K-means clustering;
- managerial recommendations.

It complements the previous marketing analytics project by shifting the focus from customer profiles and campaign response
to transaction-level sales intelligence, customer value management, and e-commerce segmentation.
""")
