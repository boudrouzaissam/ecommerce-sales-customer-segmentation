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

# Remove extreme line-level revenue outliers for clearer product visuals
df_visual = df[df["Revenue"] <= df["Revenue"].quantile(0.99)].copy()


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

The dataset contains transaction-level sales data for a **UK-based and registered non-store online retail company**
between December 2010 and December 2011. The company mainly sells unique all-occasion gift products,
and many of its customers are wholesalers.

Because the company is based in the United Kingdom, the United Kingdom naturally appears as the dominant market
in the data. Therefore, if the UK generates the highest revenue, this should not be interpreted as a global market
comparison. It mainly reflects the fact that the retailer is UK-based and likely serves a large domestic customer base.

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
number of products sold, total quantity, and average order value.
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

col1, col2 = st.columns(2)

with col1:
    fig_order_revenue = px.histogram(
        order_filtered,
        x="Order_Revenue",
        nbins=50,
        title="Distribution of Order Revenue"
    )
    st.plotly_chart(fig_order_revenue, use_container_width=True)

with col2:
    fig_order_quantity = px.histogram(
        order_filtered,
        x="Order_Quantity",
        nbins=50,
        title="Distribution of Order Quantity"
    )
    st.plotly_chart(fig_order_quantity, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    revenue_by_hour = df_filtered.groupby("InvoiceHour").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique")
    ).reset_index()

    fig_hour = px.bar(
        revenue_by_hour,
        x="InvoiceHour",
        y="Revenue",
        title="Revenue by Hour of the Day"
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

    fig_day = px.bar(
        revenue_by_day,
        x="InvoiceDay",
        y="Revenue",
        title="Revenue by Day of the Week"
    )
    st.plotly_chart(fig_day, use_container_width=True)

st.markdown(f"""
### Interpretation

The filtered data show total revenue of **${total_revenue:,.0f}** generated from **{total_orders:,} orders**
and **{total_customers:,} customers**.

The average order value is **${average_order_value:,.2f}**, which represents the average revenue generated per invoice.

The order revenue distribution helps identify whether most orders are small or whether the business depends heavily
on a small number of large transactions. The hourly and weekday graphs provide operational insights that can support
staff planning, campaign timing, and order-processing decisions.
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

fig_monthly_revenue = px.line(
    monthly_sales,
    x="InvoiceMonth",
    y="Revenue",
    markers=True,
    title="Monthly Revenue Trend"
)
st.plotly_chart(fig_monthly_revenue, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig_monthly_orders = px.line(
        monthly_sales,
        x="InvoiceMonth",
        y="Orders",
        markers=True,
        title="Monthly Orders Trend"
    )
    st.plotly_chart(fig_monthly_orders, use_container_width=True)

with col2:
    fig_monthly_customers = px.line(
        monthly_sales,
        x="InvoiceMonth",
        y="Customers",
        markers=True,
        title="Monthly Active Customers"
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

best_month = monthly_sales.sort_values("Revenue", ascending=False).iloc[0]["InvoiceMonth"]
best_month_revenue = monthly_sales.sort_values("Revenue", ascending=False).iloc[0]["Revenue"]

st.markdown(f"""
### Interpretation

The monthly revenue trend helps identify whether sales are stable, increasing, decreasing, or seasonal.

The highest revenue month in the selected data is **{best_month}**, with total revenue of **${best_month_revenue:,.0f}**.

Comparing revenue, orders, customers, and average order value helps distinguish between different business patterns.
For example, revenue may increase because the business has more customers, more orders, or larger order values.
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
We aggregate transactions by product description and calculate total revenue, quantity sold, and number of orders.
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
A product may generate high revenue because it is expensive, while another may generate high quantity because it is frequently purchased.

The treemap shows product concentration. If a few products dominate the treemap, the business may depend heavily on a limited number of items.
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
We aggregate revenue, orders, and customers by country.

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
Countries with fewer customers but high average order value may represent premium or high-value markets.

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

col1, col2 = st.columns(2)

with col1:
    fig_customer_revenue_dist = px.histogram(
        customer_summary,
        x="Revenue",
        nbins=50,
        title="Distribution of Customer Revenue"
    )
    st.plotly_chart(fig_customer_revenue_dist, use_container_width=True)

with col2:
    fig_customer_orders_dist = px.histogram(
        customer_summary,
        x="Orders",
        nbins=50,
        title="Distribution of Number of Orders per Customer"
    )
    st.plotly_chart(fig_customer_orders_dist, use_container_width=True)

fig_customer_scatter = px.scatter(
    customer_summary,
    x="Orders",
    y="Revenue",
    size="Average_Order_Value",
    color="Country",
    hover_data=["CustomerID", "Products"],
    title="Customer Value: Orders vs Revenue"
)
st.plotly_chart(fig_customer_scatter, use_container_width=True)

top_customer = top_customers.iloc[0]["CustomerID"]
top_customer_revenue = top_customers.iloc[0]["Revenue"]

st.markdown(f"""
### Interpretation

The highest-value customer in the selected data is customer **{top_customer}**, generating **${top_customer_revenue:,.0f}** in revenue.

The customer revenue distribution usually shows that a small number of customers generate a large share of total sales.
This is important for retention strategy, especially if many top customers are wholesalers or repeat buyers.

The scatter plot helps distinguish frequent customers from customers who place fewer but larger orders.
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

- **Recency:** number of days since the customer's last purchase.
- **Frequency:** number of unique orders.
- **Monetary:** total revenue generated by the customer.

Customers are scored from 1 to 5 for each dimension and then grouped into segments.
""")

snapshot_date = df_filtered["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df_filtered.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum")
).reset_index()

# RFM scores
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

fig_rfm_scatter = px.scatter(
    rfm,
    x="Frequency",
    y="Monetary",
    color="RFM_Segment",
    size="Monetary",
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

The RFM analysis shows how customer value differs across behavioral segments.

The segment generating the highest revenue is **{top_rfm_segment}**, with total revenue of **${top_rfm_revenue:,.0f}**.

- **Champions** are recent, frequent, and high-spending customers.
- **Loyal High-Value** customers buy frequently and generate high revenue.
- **At Risk Loyal** customers purchased frequently in the past but have not purchased recently.
- **Inactive** customers have low frequency and have not purchased recently.

This segmentation is useful for designing targeted actions such as loyalty rewards, reactivation campaigns,
premium offers, and retention strategies.
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

        fig_cluster = px.scatter(
            rfm,
            x="Frequency",
            y="Monetary",
            color="Cluster",
            size="Monetary",
            hover_data=["CustomerID", "Recency", "RFM_Segment"],
            title="Customer Clusters: Frequency vs Monetary Value"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

        fig_cluster_recency = px.scatter(
            rfm,
            x="Recency",
            y="Monetary",
            color="Cluster",
            size="Frequency",
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

        st.markdown(f"""
        ### Interpretation

        K-means clustering identifies four customer groups based on recency, frequency, and monetary value.

        - Cluster **{best_cluster}** has the highest average monetary value and can be interpreted as a high-value segment.
        - Cluster **{inactive_cluster}** has the highest average recency, meaning customers in this group have not purchased recently.

        Compared with rule-based RFM segmentation, K-means provides a data-driven grouping.
        Both approaches are useful: RFM is easier to explain to managers, while K-means can reveal patterns automatically.
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
3. **Prioritize top revenue products** in inventory planning and promotional campaigns.
4. **Interpret the UK dominance carefully**, because the company is UK-based and the dataset reflects its home-market activity.
5. **Strengthen core markets** where revenue and customer base are already high.
6. **Identify premium markets** where revenue per customer or average order value is high even if the number of customers is smaller.
7. **Use RFM segmentation** to personalize marketing actions based on customer value and recent activity.
8. **Monitor monthly and weekly sales trends** to anticipate seasonal demand and plan marketing campaigns.
""")


# --------------------------------------------------
# Conclusion
# --------------------------------------------------

st.markdown("---")
st.header("Conclusion")

st.markdown("""
This project demonstrates how transaction-level e-commerce data can support sales and customer analytics.

The dashboard combines:

- sales performance indicators;
- descriptive sales graphs;
- monthly revenue, orders, and customer trends;
- product-level revenue analysis;
- country-level market analysis;
- customer value analysis;
- RFM segmentation;
- K-means clustering;
- managerial recommendations.

This project complements the previous marketing analytics dashboard by focusing on transaction-level sales behavior,
customer value, and e-commerce segmentation.
""")
