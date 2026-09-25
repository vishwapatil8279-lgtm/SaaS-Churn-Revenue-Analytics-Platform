
#MODULE 9 — Churn Risk Scoring

import pandas as pd
import numpy as np

customers = pd.read_csv("cleaned_saas_customers.csv")
subscriptions = pd.read_csv("cleaned_saas_subscriptions.csv")
usage = pd.read_csv("cleaned_saas_usage.csv")
tickets = pd.read_csv("cleaned_saas_tickets.csv")

print("Customers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)

#Convert dates
customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"],
    errors="coerce"
)
subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"],
    errors="coerce"
)
usage["Month"] = pd.to_datetime(
    usage["Month"],
    errors="coerce"
)

#Create customer-level MRR
subscription_summary = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        TotalMRR=("MRR", "sum"),
        TotalSeats=("Seats", "sum"),
        SubscriptionCount=("CustomerID", "count")
    )
    .reset_index()
)
print("\nSubscription summary:")
print(subscription_summary.head())

#Find latest subscription status
latest_subscription = (
    subscriptions
    .sort_values(["CustomerID", "StartDate"])
    .groupby("CustomerID")
    .tail(1)
    .copy()
)
print("\nLatest subscription:")
print(
    latest_subscription[
        ["CustomerID", "Status"]
    ].head(10)
)

#Create usage features
usage_summary = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgLogins=("Logins", "mean"),
        AvgActiveUsers=("ActiveUsers", "mean"),
        TotalAPICalls=("APICalls", "sum"),
        TotalSessionMinutes=("SessionMinutes", "sum")
    )
    .reset_index()
)
print("\nUsage summary:")
print(usage_summary.head())

#Create ticket features
ticket_summary = (
    tickets
    .groupby("CustomerID")
    .agg(
        TicketCount=("CustomerID", "count"),
        AvgResolutionHours=("ResolutionHours", "mean")
    )
    .reset_index()
)
print("\nTicket summary:")
print(ticket_summary.head())

#Create customer-level dataset
risk_data = customers[
    [
        "CustomerID",
        "EmployeeCount",
        "SignupDate"
    ]
].copy()

#Merge MRR:
risk_data = risk_data.merge(
    subscription_summary,
    on="CustomerID",
    how="left"
)

#Merge usage:
risk_data = risk_data.merge(
    usage_summary,
    on="CustomerID",
    how="left"
)

#Merge tickets:
risk_data = risk_data.merge(
    ticket_summary,
    on="CustomerID",
    how="left"
)

#Merge latest status:
risk_data = risk_data.merge(
    latest_subscription[
        ["CustomerID", "Status"]
    ],
    on="CustomerID",
    how="left"
)
print("\nRisk data:")
print(risk_data.head())
print("\nRisk data shape:")
print(risk_data.shape)
print("\nDuplicate CustomerIDs:")
print(
    risk_data["CustomerID"].duplicated().sum()
)

#Handle missing values correctly
risk_data["TotalMRR"] = (
    risk_data["TotalMRR"].fillna(0)
)
risk_data["TotalSeats"] = (
    risk_data["TotalSeats"].fillna(0)
)
risk_data["SubscriptionCount"] = (
    risk_data["SubscriptionCount"].fillna(0)
)
risk_data["TotalLogins"] = (
    risk_data["TotalLogins"].fillna(0)
)
risk_data["AvgLogins"] = (
    risk_data["AvgLogins"].fillna(0)
)
risk_data["AvgActiveUsers"] = (
    risk_data["AvgActiveUsers"].fillna(0)
)
risk_data["TotalAPICalls"] = (
    risk_data["TotalAPICalls"].fillna(0)
)
risk_data["TotalSessionMinutes"] = (
    risk_data["TotalSessionMinutes"].fillna(0)
)
risk_data["TicketCount"] = (
    risk_data["TicketCount"].fillna(0)
)
resolution_median = (
    risk_data["AvgResolutionHours"].median()
)
risk_data["AvgResolutionHours"] = (
    risk_data["AvgResolutionHours"]
    .fillna(resolution_median)
)

#Churned/paused status
risk_data["StatusRisk"] = np.where(
    risk_data["Status"].isin(["Churned", "Paused"]),
    1,
    0
)

#Low login activity
login_threshold = risk_data["AvgLogins"].quantile(0.25)
print("\nLow login threshold:")
print(login_threshold)
risk_data["LowLoginRisk"] = np.where(
    risk_data["AvgLogins"] <= login_threshold,
    1,
    0
)

#High support burden
ticket_threshold = risk_data["TicketCount"].quantile(0.75)
print("\nHigh ticket threshold:")
print(ticket_threshold)

risk_data["HighTicketRisk"] = np.where(
    risk_data["TicketCount"] >= ticket_threshold,
    1,
    0
)

#Slow ticket resolution
resolution_threshold = (
    risk_data["AvgResolutionHours"].quantile(0.75)
)

print("\nHigh resolution-time threshold:")
print(resolution_threshold)

risk_data["SlowResolutionRisk"] = np.where(
    risk_data["AvgResolutionHours"] >= resolution_threshold,
    1,
    0
)

#Create the risk score
risk_data["RiskScore"] = (
    risk_data["StatusRisk"]
    + risk_data["LowLoginRisk"]
    + risk_data["HighTicketRisk"]
    + risk_data["SlowResolutionRisk"]
)

# Create Risk Level
risk_data["RiskLevel"] = np.select(
    [
        risk_data["RiskScore"] >= 3,
        risk_data["RiskScore"] == 2,
        risk_data["RiskScore"] <= 1
    ],
    [
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ],
    default="Low Risk"
)

print("\nRisk level counts:")
print(
    risk_data["RiskLevel"].value_counts()
)

#Rank customers by risk
risk_data = risk_data.sort_values(
    ["RiskScore", "TotalMRR"],
    ascending=[False, False]
).reset_index(drop=True)

risk_data["RiskRank"] = (
    risk_data.index + 1
)

print("\nTop 20 highest-risk customers:")

print(
    risk_data[
        [
            "RiskRank",
            "CustomerID",
            "Status",
            "TotalMRR",
            "AvgLogins",
            "TicketCount",
            "AvgResolutionHours",
            "RiskScore",
            "RiskLevel"
        ]
    ].head(20)
)

#Calculate MRR in highest-risk group
total_mrr = risk_data["TotalMRR"].sum()

print("\nTotal MRR:")
print(total_mrr)

#calculate high-risk MRR:
high_risk_mrr = (
    risk_data.loc[
        risk_data["RiskLevel"] == "High Risk",
        "TotalMRR"
    ].sum()
)

print("\nMRR in highest-risk group:")
print(high_risk_mrr)

#Calculate percentage of MRR at risk
high_risk_mrr_percentage = (
    high_risk_mrr / total_mrr * 100
)

print("\nPercentage of total MRR in High Risk group:")
print(
    round(high_risk_mrr_percentage, 2),
    "%"
)

#Create risk summary
risk_summary = (
    risk_data
    .groupby("RiskLevel")
    .agg(
        Customers=("CustomerID", "nunique"),
        TotalMRR=("TotalMRR", "sum"),
        AverageRiskScore=("RiskScore", "mean")
    )
    .round(2)
    .reset_index()
)

print("\nRisk Summary:")
print(risk_summary)

#Save the risk summary
risk_summary.to_csv(
    "saas_risk_summary.csv",
    index=False
)

#Save customer risk scores
risk_output = risk_data[
    [
        "RiskRank",
        "CustomerID",
        "Status",
        "TotalMRR",
        "AvgLogins",
        "TicketCount",
        "AvgResolutionHours",
        "StatusRisk",
        "LowLoginRisk",
        "HighTicketRisk",
        "SlowResolutionRisk",
        "RiskScore",
        "RiskLevel"
    ]
]
risk_output.to_csv(
    "saas_churn_risk_scores.csv",
    index=False
)

print("\nRisk scores saved successfully.")

#Save the high-risk MRR result
high_risk_summary = pd.DataFrame({
    "TotalMRR": [total_mrr],
    "HighRiskMRR": [high_risk_mrr],
    "HighRiskMRRPercentage": [
        round(high_risk_mrr_percentage, 2)
    ]
})

high_risk_summary.to_csv(
    "saas_highest_risk_mrr.csv",
    index=False
)

print("\nHigh-risk MRR summary saved successfully.")

