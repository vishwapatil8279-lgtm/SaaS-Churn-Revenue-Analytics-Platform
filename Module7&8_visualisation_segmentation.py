import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#MODULE 7 — VISUALISATION

customers = pd.read_csv("cleaned_saas_customers.csv")
subscriptions = pd.read_csv("cleaned_saas_subscriptions.csv")
usage = pd.read_csv("cleaned_saas_usage.csv")
tickets = pd.read_csv("cleaned_saas_tickets.csv")

print("Customers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)


#Convert date columns

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
tickets["OpenedDate"] = pd.to_datetime(
    tickets["OpenedDate"],
    errors="coerce"
)

#Create latest subscription status

subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"],
    errors="coerce"
)

latest_subscription = (
    subscriptions
    .sort_values(["CustomerID", "StartDate"])
    .groupby("CustomerID")
    .tail(1)
    .copy()
)

print("\nActual subscription columns:")
print(subscriptions.columns.tolist())

print("\nLatest subscription status:")
print(
    latest_subscription[
        ["CustomerID", "Status"]
    ].head(10)
)


# CHART 1 — Churn Trend Over Time

churned = latest_subscription[
    latest_subscription["Status"] == "Churned"
].copy()

churned["ChurnMonth"] = (
    churned["StartDate"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

churn_trend = (
    churned
    .groupby("ChurnMonth")["CustomerID"]
    .nunique()
    .reset_index(name="ChurnedCustomers")
)

print(churn_trend)

plt.figure(figsize=(10, 5))

plt.plot(
    churn_trend["ChurnMonth"],
    churn_trend["ChurnedCustomers"],
    marker="o"
)

plt.title("Churn Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Churned Customers")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("01_churn_trend.png", dpi=300)
plt.show()

#CHART 2 — Retention Curve

cohort_activity = pd.read_csv(
    "saas_cohort_activity.csv"
)
print(cohort_activity.head())
retention_curve = (
    cohort_activity
    .groupby("MonthsSinceSignup")["RetentionRate"]
    .mean()
    .reset_index()
)
print(retention_curve)

plt.figure(figsize=(10, 5))
plt.plot(
    retention_curve["MonthsSinceSignup"],
    retention_curve["RetentionRate"],
    marker="o"
)
plt.title("Customer Retention Curve")
plt.xlabel("Months Since Signup")
plt.ylabel("Average Retention Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("02_retention_curve.png", dpi=300)
plt.show()

#CHART 3 — Churn by Segment

customer_status = customers[
    ["CustomerID", "Industry", "Country", "AcquisitionChannel"]
].merge(
    latest_subscription[
        ["CustomerID", "Status"]
    ],
    on="CustomerID",
    how="left"
)
customer_status["Churned"] = np.where(
    customer_status["Status"] == "Churned",
    1,
    0
)
churn_by_industry = (
    customer_status
    .groupby("Industry")
    .agg(
        Customers=("CustomerID", "nunique"),
        ChurnedCustomers=("Churned", "sum")
    )
    .reset_index()
)

churn_by_industry["ChurnRate"] = (
    churn_by_industry["ChurnedCustomers"]
    / churn_by_industry["Customers"]
    * 100
)
print(churn_by_industry)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=churn_by_industry,
    x="Industry",
    y="ChurnRate"
)
plt.title("Churn Rate by Industry")
plt.xlabel("Industry")
plt.ylabel("Churn Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("03_churn_by_industry.png", dpi=300)
plt.show()

# CHART 4 — Usage Distribution
usage_distribution = (
    usage
    .groupby("CustomerID")["Logins"]
    .sum()
    .reset_index()
)
print(usage_distribution.head())
plt.figure(figsize=(10, 5))

plt.hist(
    usage_distribution["Logins"],
    bins=20
)

plt.title("Distribution of Customer Logins")
plt.xlabel("Total Logins")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("04_usage_distribution.png", dpi=300)
plt.show()

#CHART 5 — Usage vs Churn Relationship
customer_usage = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgActiveUsers=("ActiveUsers", "mean")
    )
    .reset_index()
)
usage_churn = customer_usage.merge(
    latest_subscription[
        ["CustomerID", "Status"]
    ],
    on="CustomerID",
    how="left"
)
usage_churn["Churned"] = np.where(
    usage_churn["Status"] == "Churned",
    1,
    0
)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=usage_churn,
    x="TotalLogins",
    y="Churned"
)
plt.title("Usage vs Churn Relationship")
plt.xlabel("Total Logins")
plt.ylabel("Churn Status (0 = Retained, 1 = Churned)")
plt.tight_layout()
plt.savefig("05_usage_vs_churn.png", dpi=300)
plt.show()


# CHART 6 - TICKET SATISFACTION IMPACT

print("\nTicket columns:")
print(tickets.columns.tolist())

# Check possible satisfaction column names
possible_satisfaction_columns = [
    "Satisfaction",
    "SatisfactionScore",
    "CSAT",
    "CustomerSatisfaction",
    "SatisfactionRating"
]

satisfaction_column = None

for column in possible_satisfaction_columns:
    if column in tickets.columns:
        satisfaction_column = column
        break

if satisfaction_column is None:
    print("\nERROR: No satisfaction column was found.")
    print("Available ticket columns:")
    print(tickets.columns.tolist())
else:

    print("\nUsing satisfaction column:", satisfaction_column)

    # Convert satisfaction column to numeric
    tickets[satisfaction_column] = pd.to_numeric(
        tickets[satisfaction_column],
        errors="coerce"
    )

    # Ticket summary by customer
    ticket_summary = (
        tickets
        .groupby("CustomerID")
        .agg(
            TicketCount=("CustomerID", "count"),
            AvgSatisfaction=(satisfaction_column, "mean"),
            AvgResolutionHours=("ResolutionHours", "mean")
        )
        .reset_index()
    )

    print("\nTicket summary:")
    print(ticket_summary.head())

    # Merge with latest subscription status
    ticket_churn = ticket_summary.merge(
        latest_subscription[
            ["CustomerID", "Status"]
        ],
        on="CustomerID",
        how="left"
    )

    print("\nTicket satisfaction by status:")
    print(
        ticket_churn.groupby("Status")["AvgSatisfaction"].mean()
    )

    # Chart
    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=ticket_churn,
        x="Status",
        y="AvgSatisfaction"
    )

    plt.title("Ticket Satisfaction by Churn Status")
    plt.xlabel("Subscription Status")
    plt.ylabel("Average Ticket Satisfaction")

    plt.tight_layout()

    plt.savefig(
        "06_ticket_satisfaction_impact.png",
        dpi=300
    )

    plt.show()

# CHART 7 - CORRELATION HEATMAP

# Subscription summary
subscription_summary = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        TotalMRR=("MRR", "sum"),
        TotalSeats=("Seats", "sum")
    )
    .reset_index()
)

# Usage summary
usage_summary = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgActiveUsers=("ActiveUsers", "mean"),
        TotalAPICalls=("APICalls", "sum"),
        TotalSessionMinutes=("SessionMinutes", "sum")
    )
    .reset_index()
)

# Ticket summary
ticket_summary = (
    tickets
    .groupby("CustomerID")
    .agg(
        TicketCount=("CustomerID", "count"),
        AvgResolutionHours=("ResolutionHours", "mean")
    )
    .reset_index()
)

# Merge all customer-level summaries
correlation_data = (
    customers[["CustomerID", "EmployeeCount"]]
    .merge(
        subscription_summary,
        on="CustomerID",
        how="left"
    )
    .merge(
        usage_summary,
        on="CustomerID",
        how="left"
    )
    .merge(
        ticket_summary,
        on="CustomerID",
        how="left"
    )
)
print("\nCorrelation data:")
print(correlation_data.head())

#correlation columns
correlation_columns = [
    "EmployeeCount",
    "TotalMRR",
    "TotalSeats",
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "TotalSessionMinutes",
    "TicketCount",
    "AvgResolutionHours"
]
correlation_matrix = correlation_data[
    correlation_columns
].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

#Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)
plt.title("Correlation Heatmap of SaaS Customer Metrics")
plt.xlabel("Metrics")
plt.ylabel("Metrics")
plt.tight_layout()
plt.savefig(
    "07_correlation_heatmap.png",
    dpi=300
)
plt.show()

correlation_matrix.to_csv(
    "saas_module7_correlation_matrix.csv"
)
print("\nCorrelation heatmap completed successfully.")

# MODULE 8 — Customer Segmentation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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
tickets["OpenedDate"] = pd.to_datetime(
    tickets["OpenedDate"],
    errors="coerce"
)

#Build subscription-level customer features
subscription_features = (
    subscriptions
    .groupby("CustomerID")
    .agg(
        TotalMRR=("MRR", "sum"),
        TotalSeats=("Seats", "sum"),
        SubscriptionCount=("CustomerID", "count")
    )
    .reset_index()
)
print("\nSubscription features:")
print(subscription_features.head())

#Build usage-level customer features
usage_features = (
    usage
    .groupby("CustomerID")
    .agg(
        TotalLogins=("Logins", "sum"),
        AvgActiveUsers=("ActiveUsers", "mean"),
        TotalAPICalls=("APICalls", "sum"),
        TotalSessionMinutes=("SessionMinutes", "sum")
    )
    .reset_index()
)
print("\nUsage features:")
print(usage_features.head())

#Build ticket-level customer features
ticket_features = (
    tickets
    .groupby("CustomerID")
    .agg(
        TicketCount=("CustomerID", "count"),
        AvgResolutionHours=("ResolutionHours", "mean")
    )
    .reset_index()
)
print("\nTicket features:")
print(ticket_features.head())

#Calculate customer tenure
today = pd.Timestamp.today()
customers["TenureYears"] = (
    (today - customers["SignupDate"]).dt.days / 365.25
)
print(
    customers[
        ["CustomerID", "SignupDate", "TenureYears"]
    ].head()
)

#Combine everything into one customer-level dataset
customer_features = customers[
    ["CustomerID", "EmployeeCount", "TenureYears"]
].copy()
#merge
customer_features = customer_features.merge(
    subscription_features,
    on="CustomerID",
    how="left"
)
customer_features = customer_features.merge(
    usage_features,
    on="CustomerID",
    how="left"
)
customer_features = customer_features.merge(
    ticket_features,
    on="CustomerID",
    how="left"
)
print("\nCustomer-level dataset:")
print(customer_features.head())
print("\nShape:")
print(customer_features.shape)
print("\nDuplicate CustomerIDs:")
print(
    customer_features["CustomerID"].duplicated().sum()
)

#Select segmentation features
features = [
    "EmployeeCount",
    "TenureYears",
    "TotalMRR",
    "TotalSeats",
    "TotalLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "TotalSessionMinutes",
    "TicketCount",
    "AvgResolutionHours"
]

print("\nSegmentation features:")
print(features)
print("\nMissing values:")
print(customer_features[features].isna().sum())

# HANDLE MISSING VALUES FOR SEGMENTATION

# No subscription record = no subscription revenue/seats
customer_features["TotalMRR"] = (
    customer_features["TotalMRR"].fillna(0)
)
customer_features["TotalSeats"] = (
    customer_features["TotalSeats"].fillna(0)
)
customer_features["SubscriptionCount"] = (
    customer_features["SubscriptionCount"].fillna(0)
)

# No usage record = no observed usage
customer_features["TotalLogins"] = (
    customer_features["TotalLogins"].fillna(0)
)
customer_features["AvgActiveUsers"] = (
    customer_features["AvgActiveUsers"].fillna(0)
)
customer_features["TotalAPICalls"] = (
    customer_features["TotalAPICalls"].fillna(0)
)
customer_features["TotalSessionMinutes"] = (
    customer_features["TotalSessionMinutes"].fillna(0)
)

# No tickets = zero ticket count
customer_features["TicketCount"] = (
    customer_features["TicketCount"].fillna(0)
)
# Calculate median from available resolution times
resolution_median = (
    customer_features["AvgResolutionHours"].median()
)
# Fill missing resolution time
customer_features["AvgResolutionHours"] = (
    customer_features["AvgResolutionHours"]
    .fillna(resolution_median)
)
print("\nMissing values after cleaning:")
print(customer_features[features].isna().sum())

#Now create X 
X = customer_features[features].copy()
print("\nX shape:")
print(X.shape)
print("\nMissing values in X:")
print(X.isna().sum())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled data:")
print(X_scaled[:5])
# ELBOW METHOD
inertia = []
k_values = range(2, 9)
for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
print("\nElbow results:")
elbow_results = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia
})
print(elbow_results)
plt.figure(figsize=(8, 5))
plt.plot(
    k_values,
    inertia,
    marker="o"
)
plt.title("Elbow Method for Customer Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(k_values))
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "08_elbow_method.png",
    dpi=300
)
plt.show()

print(customer_features[features].isna().sum().sum())

#Elbow results code

elbow_results = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertia
})

print("\nElbow results:")
print(elbow_results)

elbow_results.to_csv(
    "saas_elbow_results.csv",
    index=False
)

# RUN K-MEANS

k = 4
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
customer_features["Cluster"] = kmeans.fit_predict(X_scaled)
print("\nK-Means clustering completed.")

#run cluster counts
cluster_counts = (
    customer_features["Cluster"]
    .value_counts()
    .sort_index()
)
print("\nCustomers in each cluster:")
print(cluster_counts)

#Create the cluster profile
cluster_profile = (
    customer_features
    .groupby("Cluster")[features]
    .mean()
    .round(2)
)
print("\nCluster Profile:")
print(cluster_profile)

cluster_profile.to_csv(
    "saas_cluster_profile.csv"
)

#Create the behavioral segment column
cluster_names = {
    0: "High-Value Highly Engaged",
    1: "Low-Engagement New Customers",
    2: "Growing Active Customers",
    3: "Support-Heavy At-Risk Customers"
}

customer_features["BehaviorSegment"] = (
    customer_features["Cluster"]
    .map(cluster_names)
)

print(
    customer_features[
        ["CustomerID", "Cluster", "BehaviorSegment"]
    ].head(20)
)

retention_recommendations = {
    "High-Value Highly Engaged":
        "Use proactive account management, loyalty benefits and expansion opportunities.",

    "Low-Engagement New Customers":
        "Provide onboarding assistance, product education and early engagement campaigns.",

    "Growing Active Customers":
        "Encourage feature adoption and plan expansion while engagement is strong.",

    "Support-Heavy At-Risk Customers":
        "Prioritize support resolution, proactive outreach and issue-resolution follow-up."
}

#Create column:
customer_features["RetentionRecommendation"] = (
    customer_features["BehaviorSegment"]
    .map(retention_recommendations)
)

#Final segmentation table
final_segmentation = customer_features[
    [
        "CustomerID",
        "Cluster",
        "BehaviorSegment",
        "TotalMRR",
        "TotalSeats",
        "TotalLogins",
        "AvgActiveUsers",
        "TotalAPICalls",
        "TotalSessionMinutes",
        "TicketCount",
        "AvgResolutionHours",
        "TenureYears",
        "RetentionRecommendation"
    ]
]
print("\nFinal Customer Segmentation:")
print(final_segmentation.head(20))

final_segmentation.to_csv(
    "saas_customer_segmentation.csv",
    index=False
)
print("\nCustomer segmentation saved successfully.")

# Create a final segment summary
segment_summary = (
    customer_features
    .groupby("BehaviorSegment")
    .agg(
        Customers=("CustomerID", "nunique"),
        AvgMRR=("TotalMRR", "mean"),
        AvgSeats=("TotalSeats", "mean"),
        AvgLogins=("TotalLogins", "mean"),
        AvgActiveUsers=("AvgActiveUsers", "mean"),
        AvgTickets=("TicketCount", "mean"),
        AvgResolutionHours=("AvgResolutionHours", "mean"),
        AvgTenureYears=("TenureYears", "mean")
    )
    .round(2)
    .reset_index()
)
print("\nFinal Segment Summary:")
print(segment_summary)

segment_summary.to_csv(
    "saas_segment_summary.csv",
    index=False
)

