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

# Basic cleaning
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

# Visual capped versions only for readability of boxplots and scatter plots
df_visual = df[df["Revenue"] <= df["Revenue"].quantile(0.99)].copy()
order_visual = order_df[
    (order_df["Order_Revenue"] <= order_df["Order_Revenue"].quantile(0.99)) &
    (order_df["Order_Quantity"] <= order_df["Order_Quantity"].quantile(0.99))
].copy()


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

df_visual_filtered = df_visual[
    (df_visual["Country"].isin(selected_countries))
    & (df_visual["InvoiceDate"].dt.date >= start_date)
    & (df_visual["InvoiceDate"].dt.date <= end_date)
].copy()

order_filtered = order_df[
    (order_df["Country"].isin(selected_countries))
    & (order_df["InvoiceDate"].dt.date >= start_date)
    & (order_df["InvoiceDate"].dt.date <= end_date)
].copy()

order_visual_filtered = order_visual[
    (order_visual["Country"].isin(selected_countries))
    & (order_visual["InvoiceDate"].dt.date >= start_date)
    & (order_visual["InvoiceDate"].dt.date <= end_date)
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
This project analyzes the **Online Retail** dataset obtained from the **UCI Machine Learning Repository**.

The dataset contains transaction-level sales data for a **UK-based non-store online retail company**
between December 2010 and December 2011. The company mainly sells unique all-occasion gift products,
and many of its customers are wholesalers.

Because the company is based in the United Kingdom, the United Kingdom naturally appears as the dominant market
in the data. Therefore, if the UK generates the highest revenue, this should not be interpreted as a global market
comparison. It mainly reflects the retailer's home-market concentration and its likely domestic customer base.

This dashboard analyzes sales performance, product revenue, customer value, country-level markets,
RFM segmentation, and customer clusters.
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
The dataset contains transaction-level information, including invoice number, product code,
product description, quantity, invoice date, unit price, customer ID, and country.

Rows with missing customer IDs, negative quantities, cancellations, and non-positive unit prices are removed
to focus on completed purchase transactions.
""")

with st.expander("View first rows of the cleaned dataset"):
    st.dataframe(df_filtered.head(100))

with st.expander("View variable types"):
    st.write(df_filtered.dtypes)

dataset_summary = pd.DataFrame({
    "Indicator": [
        "Raw transaction lines",
        "Cleaned transaction lines",
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
How are overall sales performing in terms of revenue, orders, customers, products, and average order value?

### Objective
The objective is to summarize the overall commercial performance of the e-commerce business.

### Method
We calculate key performance indicators including total revenue, number of orders, number of customers,
number of products sold, total quantity, and average order value. We also examine order size, hourly performance,
and weekday performance.
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

st.markdown("### Results: Order Value and Order Quantity Distribution")

col1, col2 = st.columns(2)

with col1:
    fig_order_revenue_box = px.box(
        order_visual_filtered,
        y="Order_Revenue",
        points="outliers",
        title="Order Revenue Distribution",
        labels={"Order_Revenue": "Order Revenue"}
    )
    st.plotly_chart(fig_order_revenue_box, use_container_width=True)

with col2:
    fig_order_quantity_box = px.box(
        order_visual_filtered,
        y="Order_Quantity",
        points="outliers",
        title="Order Quantity Distribution",
        labels={"Order_Quantity": "Order Quantity"}
    )
    st.plotly_chart(fig_order_quantity_box, use_container_width=True)

st.markdown("""
The boxplots show the distribution of order revenue and order quantity without using a `count` axis.
They help identify the median order, the spread of orders, and unusually large transactions.
The charts are visually capped at the 99th percentile to make the central distribution easier to read.
""")

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
        marker=dict(size=14)
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
        marker=dict(size=14)
    )

    st.plotly_chart(fig_day, use_container_width=True)

st.markdown(f"""
### Interpretation

The filtered data show total revenue of **${total_revenue:,.0f}** generated from **{total_orders:,} orders**
and **{total_customers:,} customers**.

The average order value is **${average_order_value:,.2f}**, which represents the average revenue generated per invoice.

The strongest revenue hour is around **{best_hour}:00**, generating **${best_hour_revenue:,.0f}**.
This may reflect the time when customers or wholesale buyers are most active, possibly during business hours
or during the period when orders are processed by the retailer.

The strongest revenue day is **{best_day}**, generating **${best_day_revenue:,.0f}**.
This may be linked to weekly purchasing routines, business-to-business ordering behavior, or operational cycles.
For a retailer with many wholesale customers, orders may concentrate during specific working days rather than weekends.
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
    marker=dict(size=14)
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
        marker=dict(size=14)
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
        marker=dict(size=14)
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

A monthly peak may be explained by several business factors. For a gift-products retailer, stronger sales can occur
before end-of-year holidays, Christmas preparation, seasonal promotions, or wholesale restocking periods.
If revenue rises together with orders and active customers, the increase likely reflects broader demand.
If revenue rises while the number of orders remains stable, the increase may instead come from larger order values.

The comparison between revenue, orders, customers, and average order value is important because sales growth can come from:
- more customers;
- more frequent orders;
- larger orders;
- higher-value products;
- seasonal or promotional effects.
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
    size_max=45,
    hover_name="Description",
    title="Top Products: Quantity Sold vs Revenue"
)
st.plotly_chart(fig_product_quantity_revenue, use_container_width=True)

top_product_name = top_revenue_products.iloc[0]["Description"]
top_product_revenue = top_revenue_products.iloc[0]["Revenue"]

st.markdown(f"""
### Interpretation

The product generating the highest revenue is **{top_product_name}**, with total revenue of **${top_product_revenue:,.0f}**.

Products with high revenue are not always the same as products with the highest quantities sold.
A product may generate high revenue because it has a higher unit price, while another product may generate high quantity
because it is inexpensive and frequently purchased.

The treemap helps detect product concentration. If a small number of products occupy most of the treemap,
the business may depend heavily on a limited set of products. This can be positive for focus, but risky if demand
for those products declines.
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

Important clarification: because the retailer is UK-based, the United Kingdom is expected to dominate the data.
Therefore, the UK result should be interpreted as the company's home-market concentration rather than as evidence
that the UK is necessarily the most attractive global e-commerce market.
""")

country_summary = df_filtered.groupby("Country").agg(
    Revenue=("Revenue", "sum"),
    Orders=("InvoiceNo", "nunique"),
    Customers=("CustomerID", "nunique"),
    Quantity=("Quantity", "sum")
).reset_index()

country_summary["Average_Revenue_Per_Customer"] = (
    country_summary["Revenue"] / country_summary["Customers"]
)

country_summary["Average_Order_Value"] = (
    country_summary["Revenue"] / country_summary["Orders"]
)

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

top_country = country_summary.iloc[0]["Country"]
top_country_revenue = country_summary.iloc[0]["Revenue"]

st.markdown(f"""
### Interpretation

The strongest market by revenue is **{top_country}**, generating **${top_country_revenue:,.0f}**.

If the United Kingdom appears first, this is expected because the dataset comes from a UK-based online retailer.
Therefore, the result mainly reflects home-market concentration.

Countries with high revenue and many customers can be considered core markets.
Countries with fewer customers but high average order value may represent premium or wholesale-oriented markets.

This analysis can support decisions about geographic targeting, logistics, customer service, and international expansion.
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

customer_summary["Average_Order_Value"] = (
    customer_summary["Revenue"] / customer_summary["Orders"]
)

top_customers = customer_summary.sort_values("Revenue", ascending=False).head(15)

st.markdown("### Results: Top Customers by Revenue")
st.dataframe(top_customers)

fig_top_customers = px.bar(
    top_customers.sort_values("Revenue", ascending=True),
    x="Revenue",
    y="CustomerID",
    orientation="h",
    title="Top 15 Customers by Revenue"
)
st.plotly_chart(fig_top_customers, use_container_width=True)

customer_visual = customer_summary[
    (customer_summary["Revenue"] <= customer_summary["Revenue"].quantile(0.99)) &
    (customer_summary["Orders"] <= customer_summary["Orders"].quantile(0.99))
].copy()

col1, col2 = st.columns(2)

with col1:
    fig_customer_revenue_box = px.box(
        customer_visual,
        y="Revenue",
        points="outliers",
        title="Customer Revenue Distribution",
        labels={"Revenue": "Customer Revenue"}
    )
    st.plotly_chart(fig_customer_revenue_box, use_container_width=True)

with col2:
    fig_customer_orders_box = px.box(
        customer_visual,
        y="Orders",
        points="outliers",
        title="Orders per Customer Distribution",
        labels={"Orders": "Orders per Customer"}
    )
    st.plotly_chart(fig_customer_orders_box, use_container_width=True)

fig_customer_scatter = px.scatter(
    customer_visual,
    x="Orders",
    y="Revenue",
    size="Average_Order_Value",
    size_max=45,
    color="Country",
    hover_data=["CustomerID", "Products", "Average_Order_Value"],
    title="Customer Value: Orders vs Revenue"
)
st.plotly_chart(fig_customer_scatter, use_container_width=True)

top_customer = top_customers.iloc[0]["CustomerID"]
top_customer_revenue = top_customers.iloc[0]["Revenue"]

st.markdown(f"""
### Interpretation

The highest-value customer in the selected data is customer **{top_customer}**, generating **${top_customer_revenue:,.0f}** in revenue.

The customer revenue boxplot shows whether revenue is concentrated among a small number of customers.
In many e-commerce businesses, a small number of customers, especially repeat buyers or wholesalers, generate a large share of sales.

The scatter plot compares the number of orders with total revenue.
Large points represent customers with higher average order values. This helps distinguish:
- frequent customers with many orders;
- high-ticket customers with fewer but larger orders;
- low-value customers with limited purchasing activity.
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
We use RFM analysis:

- **Recency:** how recently the customer purchased. Lower recency means the customer bought more recently.
- **Frequency:** how often the customer purchased. Higher frequency means stronger repeat-purchase behavior.
- **Monetary:** how much revenue the customer generated. Higher monetary value means higher customer value.

Customers receive scores from 1 to 5 for each dimension.
A high RFM score generally indicates a customer who is recent, frequent, and valuable.
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
    (rfm["Monetary"] <= rfm["Monetary"].quantile(0.99)) &
    (rfm["Frequency"] <= rfm["Frequency"].quantile(0.99))
].copy()

fig_rfm_scatter = px.scatter(
    rfm_visual,
    x="Frequency",
    y="Monetary",
    color="RFM_Segment",
    size="Monetary",
    size_max=45,
    hover_data=["CustomerID", "Recency", "RFM_Score"],
    title="RFM Segments: Frequency vs Monetary Value"
)
st.plotly_chart(fig_rfm_scatter, use_container_width=True)

fig_rfm_recency = px.box(
    rfm,
    x="RFM_Segment",
    y="Recency",
    title="Recency Distribution by RFM Segment"
)
st.plotly_chart(fig_rfm_recency, use_container_width=True)

top_rfm_segment = rfm_segment_summary.iloc[0]["RFM_Segment"]
top_rfm_revenue = rfm_segment_summary.iloc[0]["Total_Revenue"]

st.markdown(f"""
### Interpretation

The RFM analysis separates customers according to how recently they purchased, how frequently they purchase,
and how much revenue they generate.

The segment generating the highest revenue is **{top_rfm_segment}**, with total revenue of **${top_rfm_revenue:,.0f}**.

The main segment meanings are:

- **Champions:** recent, frequent, and high-spending customers. These are the most valuable customers and should be protected.
- **Loyal High-Value:** customers who buy frequently and generate high revenue, even if they are not always the most recent.
- **Big Spenders:** customers with high monetary value, but not necessarily frequent or recent.
- **At Risk Loyal:** customers who used to buy frequently but have not purchased recently. These are important for reactivation.
- **New or Recent:** customers who purchased recently but may not yet have high frequency or monetary value.
- **Inactive:** customers with weak recent activity and low purchase frequency.

RFM is useful because it translates transaction data into marketing actions.
For example, champions can receive loyalty rewards, inactive customers can receive reactivation campaigns,
and recent customers can receive onboarding or second-purchase offers.
""")


# --------------------------------------------------
# Question 7: K-means segmentation
# --------------------------------------------------

st.markdown("---")
st.header("7. Can customers be segmented using K-means clustering?")

st.markdown("""
### Research Question
Can customers be grouped into data-driven segments based on their RFM behavior?

### Objective
The objective is to create customer groups using unsupervised machine learning.

### Method
We apply K-means clustering to Recency, Frequency, and Monetary values after standardization.
Unlike RFM rules, K-means does not use predefined labels. It groups customers based on similarity in the data.
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
            (rfm["Monetary"] <= rfm["Monetary"].quantile(0.99)) &
            (rfm["Frequency"] <= rfm["Frequency"].quantile(0.99))
        ].copy()

        fig_cluster = px.scatter(
            cluster_visual,
            x="Frequency",
            y="Monetary",
            color="Cluster",
            size="Monetary",
            size_max=45,
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
            size_max=45,
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

        best_cluster = cluster_summary.sort_values("Average_Monetary", ascending=False).iloc[0]["Cluster"]
        inactive_cluster = cluster_summary.sort_values("Average_Recency", ascending=False).iloc[0]["Cluster"]
        frequent_cluster = cluster_summary.sort_values("Average_Frequency", ascending=False).iloc[0]["Cluster"]

        st.markdown(f"""
        ### Interpretation

        K-means clustering identifies four customer groups based on similarity in recency, frequency, and monetary value.

        - Cluster **{best_cluster}** has the highest average monetary value and can be interpreted as a high-value customer group.
        - Cluster **{frequent_cluster}** has the highest average frequency and represents the most frequent buyers.
        - Cluster **{inactive_cluster}** has the highest average recency, meaning customers in this group have not purchased recently.

        The difference between RFM and K-means is important:

        - **RFM segmentation** is rule-based and easy to explain to managers.
        - **K-means segmentation** is data-driven and can detect natural groups in the data.

        In practice, both approaches are complementary. RFM provides clear marketing labels, while K-means confirms
        whether customers naturally form distinct behavioral groups.
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
Based on the analysis, the business could consider the following actions:

1. **Protect high-value customers** through loyalty programs, personalized offers, and priority service.
2. **Reactivate inactive customers** with targeted email campaigns, discounts, or product recommendations.
3. **Use RFM segments for campaign design**: champions should receive loyalty rewards, while at-risk customers should receive reactivation offers.
4. **Prioritize top revenue products** in inventory planning and promotional campaigns.
5. **Interpret UK dominance carefully**, because the company is UK-based and the dataset reflects its home-market activity.
6. **Identify premium markets** where revenue per customer or average order value is high even if the number of customers is smaller.
7. **Monitor monthly, daily, and hourly sales patterns** to anticipate demand and plan campaign timing.
8. **Combine managerial rules and machine learning** by using RFM for explainability and K-means for data-driven segmentation.
""")


# --------------------------------------------------
# Conclusion
# --------------------------------------------------

st.markdown("---")
st.header("Conclusion")

st.markdown("""
This project shows how raw transaction data can be transformed into a practical e-commerce decision-support system.

The dashboard does more than report sales. It connects sales performance, product concentration, market structure,
customer value, and behavioral segmentation into one analytical workflow.

From a business perspective, the main value of this analysis is that it helps answer four managerial questions:

- **Where is revenue coming from?**
- **Which products and markets deserve attention?**
- **Which customers should be protected, developed, or reactivated?**
- **How can transaction data be converted into targeted commercial actions?**

The analysis also shows why context matters. The United Kingdom dominates the dataset largely because the retailer is UK-based.
This means the dashboard should be used as a company-level business intelligence tool rather than as a global market ranking.

Overall, the project demonstrates a complete business analytics workflow: data cleaning, KPI design, descriptive analysis,
sales trend analysis, product and country analysis, customer value measurement, RFM segmentation, machine learning clustering,
and managerial recommendations.

This dashboard complements the previous marketing analytics project by shifting the focus from customer profiles and campaign response
to transaction-level sales behavior, customer value, and e-commerce segmentation.
""")
