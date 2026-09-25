import pandas as pd
import numpy as np
from pathlib import Path

#cleaned datasets

customers = pd.read_csv("cleaned_saas_customers.csv")
subscriptions = pd.read_csv("cleaned_saas_subscriptions.csv")
usage = pd.read_csv("cleaned_saas_usage.csv")
tickets = pd.read_csv("cleaned_saas_tickets.csv")

# MODULE 3 — NUMPY ANALYSIS
# 1. Convert metrics to NumPy arrays

mrr_array = subscriptions["MRR"].to_numpy()
seats_array = subscriptions["Seats"].to_numpy()
logins_array = usage["Logins"].to_numpy()
active_users_array = usage["ActiveUsers"].to_numpy()
api_calls_array = usage["APICalls"].to_numpy()
session_minutes_array = usage["SessionMinutes"].to_numpy()
# 2. Reusable NumPy statistics function
def numpy_statistics(array, metric_name):

    print(f"\n{metric_name} Statistics")
    print("-" * 40)
    print("Mean:", np.mean(array))
    print("Std:", np.std(array))
    print("Min:", np.min(array))
    print("Max:", np.max(array))

# 3. Calculate statistics

numpy_statistics(mrr_array, "MRR")
numpy_statistics(seats_array, "Seats")
numpy_statistics(logins_array, "Logins")
numpy_statistics(active_users_array, "Active Users")
numpy_statistics(api_calls_array, "API Calls")
numpy_statistics(session_minutes_array, "Session Minutes")

# 4. Normalize MRR

mrr_min = np.min(mrr_array)
mrr_max = np.max(mrr_array)
mrr_normalized = (
    (mrr_array - mrr_min)
    /
    (mrr_max - mrr_min)
)
print("\nNormalized MRR")
print(mrr_normalized)
print(
    "Normalized MRR Min:",
    np.min(mrr_normalized)
)
print(
    "Normalized MRR Max:",
    np.max(mrr_normalized)
)
# 5. Customer-level MRR
customer_mrr = (
    subscriptions
    .groupby("CustomerID")["MRR"]
    .sum()
    .reset_index()
)

# 6. High-value accounts

high_value_threshold = np.percentile(
    customer_mrr["MRR"].to_numpy(),
    75
)
print(
    "\nHigh-value MRR threshold:",
    high_value_threshold
)
customer_mrr["HighValueFlag"] = np.where(
    customer_mrr["MRR"] >= high_value_threshold,
    "High Value",
    "Standard"
)
# 7. Customer-level usage
customer_usage = (
    usage
    .groupby("CustomerID")
    .agg({
        "Logins": "sum",
        "ActiveUsers": "mean",
        "APICalls": "sum",
        "SessionMinutes": "sum"
    })
    .reset_index()
)
# 8. At-risk subscription customers
at_risk_customers = (
    subscriptions[
        subscriptions["Status"].isin(
            ["Paused", "Churned"]
        )
    ]["CustomerID"]
    .drop_duplicates()
)
# 9. Merge customer-level analysis
customer_analysis = customer_mrr.merge(
    customer_usage,
    on="CustomerID",
    how="left"
)
# 10. np.where() — At-risk accounts
customer_analysis["AtRiskFlag"] = np.where(
    customer_analysis["CustomerID"].isin(
        at_risk_customers
    ),
    "At Risk",
    "Not At Risk"
)
# 11. Add low-usage risk
login_median = customer_analysis["Logins"].median()
customer_analysis["AtRiskFlag"] = np.where(
    (
        customer_analysis["AtRiskFlag"] == "At Risk"
    )
    |
    (
        customer_analysis["Logins"] < login_median
    ),
    "At Risk",
    "Not At Risk"
)
# 12. Display results
print("\nCustomer Analysis")
print("-" * 60)
print(
    customer_analysis[
        [
            "CustomerID",
            "MRR",
            "HighValueFlag",
            "Logins",
            "ActiveUsers",
            "APICalls",
            "SessionMinutes",
            "AtRiskFlag"
        ]
    ].head(20)
)
# 13. Counts

print("\nHigh-value account count:")
print(
    (
        customer_analysis["HighValueFlag"]
        == "High Value"
    ).sum()
)
print("\nAt-risk account count:")
print(
    (
        customer_analysis["AtRiskFlag"]
        == "At Risk"
    ).sum()
)
# 14. High-value + At-risk accounts

high_value_at_risk = customer_analysis[
    (
        customer_analysis["HighValueFlag"]
        == "High Value"
    )
    &
    (
        customer_analysis["AtRiskFlag"]
        == "At Risk"
    )
]
print(
    "\nHigh-value + At-risk accounts:"
)
print(high_value_at_risk)

# 15. Save result

customer_analysis.to_csv(
    "saas_numpy_customer_analysis.csv",
    index=False
)

print(
    "\nNumPy analysis saved successfully."
)

#MODULE 4 — Pandas Wrangling & EDA

# CONVERT DATES

customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"]
)
subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"]
)
subscriptions["EndDate"] = pd.to_datetime(
    subscriptions["EndDate"]
)
usage["Month"] = pd.to_datetime(
    usage["Month"]
)
tickets["OpenedDate"] = pd.to_datetime(
    tickets["OpenedDate"]
)

# Aggregate subscriptions to customer level

subscription_customer = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        TotalMRR=("MRR", "sum"),
        TotalSeats=("Seats", "sum"),
        SubscriptionCount=("SubscriptionID", "nunique")
    )
    .reset_index()
)
print(subscription_customer.head())

#SUBSCRIPTION CUSTOMER SUMMARY
subscriptions_sorted = subscriptions.sort_values(
    ["CustomerID", "StartDate"]
)
latest_subscription = (
    subscriptions_sorted
    .groupby("CustomerID")
    .tail(1)
    [
        [
            "CustomerID",
            "PlanName",
            "BillingTerm",
            "Status"
        ]
    ]
)
latest_subscription = latest_subscription.rename(
    columns={
        "PlanName": "CurrentPlan",
        "BillingTerm": "CurrentBillingTerm",
        "Status": "CurrentStatus"
    }
)

#Merge this into subscription summary:
subscription_customer = subscription_customer.merge(
    latest_subscription,
    on="CustomerID",
    how="left"
)

#Aggregate usage to customer level
usage_customer = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgActiveUsers=("ActiveUsers", "mean"),
        TotalAPICalls=("APICalls", "sum"),
        TotalSessionMinutes=("SessionMinutes", "sum"),
        UsageMonths=("Month", "nunique")
    )
    .reset_index()
)

print(usage_customer.head())

#Calculate usage trend

def calculate_usage_trend(group):
    group = group.sort_values("Month")

    if len(group) < 2:
        return np.nan

    x = np.arange(len(group))
    y = group["Logins"].to_numpy()

    slope = np.polyfit(x, y, 1)[0]

    return slope

usage_trend = (
    usage
    .groupby("CustomerID")
    .apply(
        calculate_usage_trend,
        include_groups=False
    )
    .reset_index(name="UsageTrend")
)

usage_trend = (
    usage
    .groupby("CustomerID")
    .apply(calculate_usage_trend)
    .reset_index(name="UsageTrend")
)

usage_trend = (
    usage
    .groupby("CustomerID")
    .apply(calculate_usage_trend)
    .reset_index(name="UsageTrend")
)

#Aggregate tickets to customer level

ticket_customer = (
    tickets
    .groupby("CustomerID")
    .agg(
        TicketCount=("TicketID", "nunique"),
        AvgResolutionHours=("ResolutionHours", "mean"),
        AvgSatisfaction=("SatisfactionScore", "mean")
    )
    .reset_index()
)

print(ticket_customer.head())

#Calculate Tickets per Month

ticket_months = (
    tickets
    .groupby("CustomerID")["OpenedDate"]
    .agg(
        FirstTicketDate="min",
        LastTicketDate="max"
    )
    .reset_index()
)

ticket_months["TicketObservationMonths"] = (
    (
        ticket_months["LastTicketDate"]
        - ticket_months["FirstTicketDate"]
    ).dt.days / 30.44
).clip(lower=1)

#Merge ticket months

ticket_customer = ticket_customer.merge(
    ticket_months[
        [
            "CustomerID",
            "TicketObservationMonths"
        ]
    ],
    on="CustomerID",
    how="left"
)

ticket_customer["TicketsPerMonth"] = (
    ticket_customer["TicketCount"]
    /
    ticket_customer["TicketObservationMonths"]
)

#Create the final customer-level view

customer_view = customers.copy()

customer_view = customer_view.merge(
    subscription_customer,
    on="CustomerID",
    how="left"
)

customer_view = customer_view.merge(
    usage_customer,
    on="CustomerID",
    how="left"
)

customer_view = customer_view.merge(
    ticket_customer,
    on="CustomerID",
    how="left"
)

#final customer-level shape

print("Customer-level view shape:")
print(customer_view.shape)

print(
    "Duplicate CustomerIDs:",
    customer_view["CustomerID"].duplicated().sum()
)

#Handle customers with no child records

customer_view["TicketCount"] = (
    customer_view["TicketCount"].fillna(0)
)
customer_view["TicketsPerMonth"] = (
    customer_view["TicketsPerMonth"].fillna(0)
)
customer_view["TotalLogins"] = (
    customer_view["TotalLogins"].fillna(0)
)
customer_view["TotalAPICalls"] = (
    customer_view["TotalAPICalls"].fillna(0)
)
customer_view["TotalSessionMinutes"] = (
    customer_view["TotalSessionMinutes"].fillna(0)
)

#Calculated column: Tenure

analysis_date = max(
    customers["SignupDate"].max(),
    subscriptions["StartDate"].max(),
    usage["Month"].max(),
    tickets["OpenedDate"].max()
)
print("Analysis date:", analysis_date)
customer_view["TenureYears"] = (
    (
        analysis_date
        - customer_view["SignupDate"]
    ).dt.days / 365.25
)
customer_view["TenureMonths"] = (
    customer_view["TenureYears"] * 12
)

# Calculated column: Revenue per Seat

customer_view["RevenuePerSeat"] = np.where(
    customer_view["TotalSeats"] > 0,
    customer_view["TotalMRR"]
    / customer_view["TotalSeats"],
    np.nan
)
# Revenue per Sea
print(
    customer_view[
        [
            "CustomerID",
            "TotalMRR",
            "TotalSeats",
            "RevenuePerSeat"
        ]
    ].head(10)
)

# GroupBy analysis
def group_analysis(df, group_column):

    result = (
        df
        .groupby(group_column)
        .agg(
            Customers=("CustomerID", "nunique"),
            TotalMRR=("TotalMRR", "sum"),
            AvgMRR=("TotalMRR", "mean"),
            TotalSeats=("TotalSeats", "sum"),
            AvgSeats=("TotalSeats", "mean"),
            AvgUsage=("TotalLogins", "mean"),
            AvgTicketsPerMonth=("TicketsPerMonth", "mean"),
            AvgSatisfaction=("AvgSatisfaction", "mean")
        )
        .sort_values("TotalMRR", ascending=False)
    )

    return result

# GroupBy Plan

plan_analysis = group_analysis(
    customer_view,
    "CurrentPlan"
)
print("\nPLAN ANALYSIS")
print(plan_analysis)

#GroupBy Industry
industry_analysis = group_analysis(
    customer_view,
    "Industry"
)
print("\nINDUSTRY ANALYSIS")
print(industry_analysis)

#Region does not exist(Country using )

country_analysis = group_analysis(
    customer_view,
    "Country"
)

print("\nCOUNTRY ANALYSIS")
print(country_analysis)

#GroupBy Acquisition Channel

channel_analysis = group_analysis(
    customer_view,
    "AcquisitionChannel"
)

print("\nACQUISITION CHANNEL ANALYSIS")
print(channel_analysis)

# Build Pivot Table 1

pivot_plan_industry = pd.pivot_table(
    customer_view,
    index="CurrentPlan",
    columns="Industry",
    values="TotalMRR",
    aggfunc="sum",
    fill_value=0
)
print("\nPIVOT — PLAN × INDUSTRY")
print(pivot_plan_industry)

# Pivot Table 2
pivot_country_plan = pd.pivot_table(
    customer_view,
    index="Country",
    columns="CurrentPlan",
    values="TotalMRR",
    aggfunc="sum",
    fill_value=0
)
print("\nPIVOT — COUNTRY × PLAN")
print(pivot_country_plan)

# Pivot Table 3

pivot_channel_plan = pd.pivot_table(
    customer_view,
    index="AcquisitionChannel",
    columns="CurrentPlan",
    values="TotalMRR",
    aggfunc="sum",
    fill_value=0
)

print("\nPIVOT — CHANNEL × PLAN")
print(pivot_channel_plan)

#IQR Outlier Detection
def detect_iqr_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_mask = (
        (df[column] < lower_bound)
        |
        (df[column] > upper_bound)
    )
    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "LowerBound": lower_bound,
        "UpperBound": upper_bound,
        "OutlierCount": outlier_mask.sum(),
        "OutlierRows": df.loc[outlier_mask]
    }

#Detect MRR outliers
mrr_outliers = detect_iqr_outliers(
    customer_view,
    "TotalMRR"
)
print("\nMRR OUTLIER RESULTS")
print("Q1:", mrr_outliers["Q1"])
print("Q3:", mrr_outliers["Q3"])
print("IQR:", mrr_outliers["IQR"])
print("Lower Bound:", mrr_outliers["LowerBound"])
print("Upper Bound:", mrr_outliers["UpperBound"])
print("Outlier Count:", mrr_outliers["OutlierCount"])

print(
    mrr_outliers["OutlierRows"][
        [
            "CustomerID",
            "CompanyName",
            "CurrentPlan",
            "TotalMRR"
        ]
    ]
)

#Detect usage outliers

login_outliers = detect_iqr_outliers(
    customer_view,
    "TotalLogins"
)
print("\nLOGIN OUTLIERS")
print("Outlier Count:", login_outliers["OutlierCount"])
api_outliers = detect_iqr_outliers(
    customer_view,
    "TotalAPICalls"
)
print("\nAPI CALL OUTLIERS")
print("Outlier Count:", api_outliers["OutlierCount"])

#Create correlation matrix
print("\nCustomer View Columns:")
print(customer_view.columns.tolist())

usage_trend = (
    usage
    .groupby("CustomerID")
    .apply(calculate_usage_trend)
    .reset_index(name="UsageTrend")
)

usage_customer = usage_customer.merge(
    usage_trend,
    on="CustomerID",
    how="left"
)
# USAGE TREND

def calculate_usage_trend(group):
    group = group.sort_values("Month")
    # Need at least 2 months to calculate a trend
    if len(group) < 2:
        return np.nan
    x = np.arange(len(group))
    y = group["Logins"].to_numpy()

    slope = np.polyfit(x, y, 1)[0]

    return slope
usage_trend = (
    usage
    .groupby("CustomerID")
    .apply(calculate_usage_trend)
    .reset_index(name="UsageTrend")
)
print("\nUsage Trend:")
print(usage_trend.head())

#First check your columns
print("\n" + "=" * 60)
print("STEP 29 - CHECK CUSTOMER VIEW COLUMNS")
print("=" * 60)
print(customer_view.columns.tolist())

#  - CORRELATION MATRIX
print("\n" + "=" * 60)
print("STEP 29 - CORRELATION MATRIX")
print("=" * 60)

# Select only columns that actually exist
correlation_columns = [
    "TotalMRR",
    "TotalSeats",
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "TotalSessionMinutes",
    "TicketCount",
    "TicketsPerMonth",
    "AvgResolutionHours",
    "AvgSatisfaction",
    "TenureYears",
    "RevenuePerSeat"
]

# Check which columns are available
available_columns = [
    column
    for column in correlation_columns
    if column in customer_view.columns
]
missing_columns = [
    column
    for column in correlation_columns
    if column not in customer_view.columns
]
print("\nAvailable correlation columns:")
print(available_columns)
print("\nMissing correlation columns:")
print(missing_columns)

# Convert selected columns to numeric
correlation_data = customer_view[available_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Create correlation matrix
correlation_matrix = correlation_data.corr()

print("\nCORRELATION MATRIX")
print(correlation_matrix.round(2))

#Save the correlation matrix

correlation_matrix.to_csv(
    "saas_correlation_matrix.csv"
)
print("\nCorrelation matrix saved successfully:")
print("saas_correlation_matrix.csv")

#Print meaningful correlations
print("\n" + "=" * 60)
print("MEANINGFUL CORRELATIONS")
print("=" * 60)
threshold = 0.30
for column1 in correlation_matrix.columns:
    for column2 in correlation_matrix.columns:
        if column1 >= column2:
            continue
        value = correlation_matrix.loc[column1, column2]
        if pd.notna(value) and abs(value) >= threshold:
            if value > 0:
                direction = "positive"
            else:
                direction = "negative"
            print(
                f"{column1} vs {column2}: "
                f"{value:.2f} ({direction})"
            )

