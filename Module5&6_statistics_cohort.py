import pandas as pd
import numpy as np
from scipy import stats

# MODULE 5 - STATISTICS
# USING CLEANED SAAS DATA

customers = pd.read_csv("cleaned_saas_customers.csv")
subscriptions = pd.read_csv("cleaned_saas_subscriptions.csv")
usage = pd.read_csv("cleaned_saas_usage.csv")
tickets = pd.read_csv("cleaned_saas_tickets.csv")

print("=" * 60)
print("CLEANED SAAS DATA LOADED")
print("=" * 60)
print("Customers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)

#Verify that we are using cleaned data
print("\nCustomers columns:")
print(customers.columns.tolist())
print("\nSubscriptions columns:")
print(subscriptions.columns.tolist())
print("\nUsage columns:")
print(usage.columns.tolist())
print("\nTickets columns:")
print(tickets.columns.tolist())

print("\nMissing values:")
print("\nCustomers:")
print(customers.isna().sum())
print("\nSubscriptions:")
print(subscriptions.isna().sum())
print("\nUsage:")
print(usage.isna().sum())
print("\nTickets:")
print(tickets.isna().sum())

#Prepare subscription data
subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"],
    errors="coerce"
)
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

#Prepare usage data
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

#Get latest subscription status
latest_subscription = (
    subscriptions
    .sort_values(["CustomerID", "StartDate"])
    .groupby("CustomerID")
    .tail(1)
)
latest_subscription = latest_subscription[
    ["CustomerID", "Status"]
].copy()
print("\nLatest subscription status:")
print(latest_subscription.head())

#Create the customer-level statistics table
stats_data = customers.copy()
stats_data = stats_data.merge(
    subscription_summary,
    on="CustomerID",
    how="left"
)
stats_data = stats_data.merge(
    usage_summary,
    on="CustomerID",
    how="left"
)
stats_data = stats_data.merge(
    latest_subscription,
    on="CustomerID",
    how="left"
)

print("\nCustomer-level statistics data:")
print(stats_data.head())
print("\nShape:")
print(stats_data.shape)
print("\nDuplicate CustomerIDs:")
print(stats_data["CustomerID"].duplicated().sum())

#Check subscription status
print("\n" + "=" * 60)
print("SUBSCRIPTION STATUS")
print("=" * 60)
print(
    stats_data["Status"].value_counts(dropna=False)
)

#Descriptive Statistics
key_metrics = [
    "TotalMRR",
    "TotalSeats",
    "TotalLogins",
    "AvgLogins",
    "AvgActiveUsers",
    "TotalAPICalls",
    "TotalSessionMinutes"
]
print("\nChecking metrics:")
for column in key_metrics:
    if column in stats_data.columns:
        print("FOUND:", column)
    else:
        print("MISSING:", column)

#Calculate descriptive statistics
descriptive_stats = (
    stats_data[key_metrics]
    .describe()
    .T
)
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print(descriptive_stats.round(2))

#Save descriptive statistics
descriptive_stats.to_csv(
    "saas_cleaned_descriptive_statistics.csv"
)
print(
    "\nSaved: saas_cleaned_descriptive_statistics.csv"
)

#Random Sample 1
login_population = stats_data[
    "AvgLogins"
].dropna()
population_mean = login_population.mean()
print("\nPopulation size:", len(login_population))
print(
    "Population mean:",
    round(population_mean, 2)
)

#sample 10 customers:
sample1 = login_population.sample(
    n=min(10, len(login_population)),
    random_state=42
)
sample1_mean = sample1.mean()
print("\n" + "=" * 60)
print("RANDOM SAMPLE 1")
print("=" * 60)
print(sample1.to_list())
print(
    "\nSample 1 mean:",
    round(sample1_mean, 2)
)
print(
    "Population mean:",
    round(population_mean, 2)
)

#Random Sample 2
sample2 = login_population.sample(
    n=min(10, len(login_population)),
    random_state=100
)
sample2_mean = sample2.mean()
print("\n" + "=" * 60)
print("RANDOM SAMPLE 2")
print("=" * 60)
print(sample2.to_list())
print(
    "\nSample 2 mean:",
    round(sample2_mean, 2)
)
print(
    "Population mean:",
    round(population_mean, 2)
)

#Compare the samples with population
difference1 = sample1_mean - population_mean
difference2 = sample2_mean - population_mean
print("\n" + "=" * 60)
print("SAMPLE MEANS VS POPULATION")
print("=" * 60)
print(
    f"Population mean: {population_mean:.2f}"
)
print(
    f"Sample 1 mean: {sample1_mean:.2f}"
)
print(
    f"Sample 1 difference: {difference1:.2f}"
)
print(
    f"Sample 2 mean: {sample2_mean:.2f}"
)
print(
    f"Sample 2 difference: {difference2:.2f}"
)

#Create churned and retained groups
print(
    stats_data["Status"].unique()
)

churned = stats_data[
    stats_data["Status"] == "Churned"
]["AvgLogins"].dropna()
retained = stats_data[
    stats_data["Status"] == "Active"
]["AvgLogins"].dropna()

#Check the two groups
print("\n" + "=" * 60)
print("CHURNED VS RETAINED")
print("=" * 60)
print(
    "Churned customers:",
    len(churned)
)
print(
    "Retained customers:",
    len(retained)
)
print(
    "Churned mean logins:",
    round(churned.mean(), 2)
)
print(
    "Retained mean logins:",
    round(retained.mean(), 2)
)

#Run Welch's t-test
t_statistic, p_value = stats.ttest_ind(
    churned,
    retained,
    equal_var=False,
    alternative="less"
)
print("\n" + "=" * 60)
print("HYPOTHESIS TEST")
print("=" * 60)
print(
    "Test: Welch's independent two-sample t-test"
)
print(
    f"T-statistic: {t_statistic:.4f}"
)
print(
    f"P-value: {p_value:.4f}"
)

#Interpret the p-value
alpha = 0.05

print(
    f"\nSignificance level: {alpha}"
)
if p_value < alpha:
    print(
        "\nResult: Reject the null hypothesis."
    )
    print(
        "There is statistically significant evidence "
        "that churned customers have lower average "
        "logins than retained customers."
    )
else:
    print(
        "\nResult: Fail to reject the null hypothesis."
    )
    print(
        "There is not enough statistical evidence "
        "to conclude that churned customers have lower "
        "average logins than retained customers."
    )

#Calculate the difference in means
mean_difference = (
    churned.mean() - retained.mean()
)
print(
    f"\nMean difference "
    f"(Churned - Retained): "
    f"{mean_difference:.2f}"
)

#Save the hypothesis result
hypothesis_result = pd.DataFrame({
    "Metric": ["Average Logins"],
    "Churned Mean": [churned.mean()],
    "Retained Mean": [retained.mean()],
    "Mean Difference": [mean_difference],
    "T Statistic": [t_statistic],
    "P Value": [p_value],
    "Alpha": [alpha]
})
print("\n" + "=" * 60)
print("FINAL HYPOTHESIS RESULT")
print("=" * 60)
print(
    hypothesis_result.round(4)
)
hypothesis_result.to_csv(
    "saas_cleaned_churn_hypothesis_test.csv",
    index=False
)
print(
    "\nSaved: saas_cleaned_churn_hypothesis_test.csv"
)

#MODULE 6 — Cohort & Retention Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

customers = pd.read_csv("cleaned_saas_customers.csv")
subscriptions = pd.read_csv("cleaned_saas_subscriptions.csv")
usage = pd.read_csv("cleaned_saas_usage.csv")
tickets = pd.read_csv("cleaned_saas_tickets.csv")

print("=" * 60)
print("MODULE 6 - CLEANED DATA LOADED")
print("=" * 60)
print("Customers:", customers.shape)
print("Subscriptions:", subscriptions.shape)
print("Usage:", usage.shape)
print("Tickets:", tickets.shape)

#Check the required columns
print("\nCustomers columns:")
print(customers.columns.tolist())
print("\nUsage columns:")
print(usage.columns.tolist())

#Convert dates correctly

customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"],
    errors="coerce"
)
usage["Month"] = pd.to_datetime(
    usage["Month"],
    errors="coerce"
)
print("\nDate types:")
print(customers["SignupDate"].dtype)
print(usage["Month"].dtype)

#Create Signup Cohort Month

customers["CohortMonth"] = (
    customers["SignupDate"]
    .dt.to_period("M")
    .dt.to_timestamp()
)
print("\nCustomer cohorts:")
print(
    customers[
        ["CustomerID", "SignupDate", "CohortMonth"]
    ].head(10)
)

#Merge cohort information with usage

cohort_usage = usage.merge(
    customers[
        ["CustomerID", "CohortMonth"]
    ],
    on="CustomerID",
    how="inner"
)
print("\nCohort usage:")
print(cohort_usage.head())

#Remove invalid dates
cohort_usage = cohort_usage.dropna(
    subset=["CustomerID", "Month", "CohortMonth"]
)
print(
    "\nRows after date validation:",
    len(cohort_usage)
)

#Convert Usage Month to month level
cohort_usage["ActivityMonth"] = (
    cohort_usage["Month"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

#Calculate months since signup
cohort_usage["CohortPeriod"] = (
    cohort_usage["CohortMonth"].dt.year * 12
    + cohort_usage["CohortMonth"].dt.month
)
cohort_usage["ActivityPeriod"] = (
    cohort_usage["ActivityMonth"].dt.year * 12
    + cohort_usage["ActivityMonth"].dt.month
)
cohort_usage["MonthsSinceSignup"] = (
    cohort_usage["ActivityPeriod"]
    - cohort_usage["CohortPeriod"]
)
print("\nMonths since signup:")
print(
    cohort_usage[
        [
            "CustomerID",
            "CohortMonth",
            "ActivityMonth",
            "MonthsSinceSignup"
        ]
    ].head(20)
)

#Remove activity before signup

cohort_usage = cohort_usage[
    cohort_usage["MonthsSinceSignup"] >= 0
].copy()

cohort_sizes = (
    customers
    .groupby("CohortMonth")["CustomerID"]
    .nunique()
)
print("\n" + "=" * 60)
print("COHORT SIZES")
print("=" * 60)
print(cohort_sizes)

#Count active customers by cohort and month
cohort_activity = (
    cohort_usage
    .groupby(
        ["CohortMonth", "MonthsSinceSignup"]
    )["CustomerID"]
    .nunique()
    .reset_index(name="ActiveCustomers")
)

print("\nCohort activity:")
print(cohort_activity.head(20))

#Calculate retention %
cohort_activity["CohortSize"] = (
    cohort_activity["CohortMonth"]
    .map(cohort_sizes)
)
cohort_activity["RetentionRate"] = (
    cohort_activity["ActiveCustomers"]
    / cohort_activity["CohortSize"]
    * 100
)
print("\nRetention calculation:")
print(
    cohort_activity.head(20).round(2)
)

#Create the Cohort Retention Table
retention_table = cohort_activity.pivot(
    index="CohortMonth",
    columns="MonthsSinceSignup",
    values="RetentionRate"
)
print("\n" + "=" * 60)
print("COHORT RETENTION TABLE")
print("=" * 60)
print(
    retention_table.round(2)
)

# Format cohort names
retention_table.index = (
    retention_table.index
    .strftime("%Y-%m")
)
print("\nFormatted retention table:")
print(
    retention_table.round(2)
)

#Save the retention table

retention_table.to_csv(
    "saas_cohort_retention_table.csv"
)
print(
    "\nSaved: saas_cohort_retention_table.csv"
)

#Create the Retention Curve

retention_curve = (
    cohort_activity
    .groupby("MonthsSinceSignup")["RetentionRate"]
    .mean()
)
print("\n" + "=" * 60)
print("RETENTION CURVE DATA")
print("=" * 60)
print(
    retention_curve.round(2)
)

#Plot the Retention Curve

plt.figure(figsize=(10, 6))
plt.plot(
    retention_curve.index,
    retention_curve.values,
    marker="o"
)
plt.title("SaaS Customer Retention Curve")
plt.xlabel("Months Since Signup")
plt.ylabel("Average Retention (%)")
plt.xticks(
    retention_curve.index
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "saas_retention_curve.png",
    dpi=300
)
plt.show()

#Find the best cohort
month_to_compare = 1
month1_cohorts = cohort_activity[
    cohort_activity["MonthsSinceSignup"] == month_to_compare
].copy()

print("\nRetention at Month 1:")
print(
    month1_cohorts[
        ["CohortMonth", "RetentionRate"]
    ]
    .sort_values("RetentionRate", ascending=False)
    .round(2)
)

#Find the worst cohort

best_cohort = month1_cohorts.loc[
    month1_cohorts["RetentionRate"].idxmax()
]
worst_cohort = month1_cohorts.loc[
    month1_cohorts["RetentionRate"].idxmin()
]

print("\n" + "=" * 60)
print("COHORT COMPARISON")
print("=" * 60)
print(
    f"Best cohort at Month 1: "
    f"{best_cohort['CohortMonth'].strftime('%Y-%m')}"
)
print(
    f"Retention: "
    f"{best_cohort['RetentionRate']:.2f}%"
)
print(
    f"\nLowest cohort at Month 1: "
    f"{worst_cohort['CohortMonth'].strftime('%Y-%m')}"
)
print(
    f"Retention: "
    f"{worst_cohort['RetentionRate']:.2f}%"
)

#Calculate retention change
month0 = cohort_activity[
    cohort_activity["MonthsSinceSignup"] == 0
]
month1 = cohort_activity[
    cohort_activity["MonthsSinceSignup"] == 1
]
month0_avg = month0["RetentionRate"].mean()
month1_avg = month1["RetentionRate"].mean()
retention_change = month1_avg - month0_avg

print("\n" + "=" * 60)
print("RETENTION CHANGE")
print("=" * 60)
print(
    f"Month 0 average retention: "
    f"{month0_avg:.2f}%"
)
print(
    f"Month 1 average retention: "
    f"{month1_avg:.2f}%"
)
print(
    f"Change: {retention_change:.2f} percentage points"
)

#Compare Month 1 vs Month 3

month3 = cohort_activity[
    cohort_activity["MonthsSinceSignup"] == 3
]

if len(month3) > 0:

    month3_avg = month3["RetentionRate"].mean()

    print(
        f"\nMonth 3 average retention: "
        f"{month3_avg:.2f}%"
    )

    print(
        f"Month 0 to Month 3 change: "
        f"{month3_avg - month0_avg:.2f} "
        f"percentage points"
    )

# Compare Month 1 vs Month 3
month3 = cohort_activity[
    cohort_activity["MonthsSinceSignup"] == 3
]
if len(month3) > 0:
    month3_avg = month3["RetentionRate"].mean()
    print(
        f"\nMonth 3 average retention: "
        f"{month3_avg:.2f}%"
    )
    print(
        f"Month 0 to Month 3 change: "
        f"{month3_avg - month0_avg:.2f} "
        f"percentage points"
    )

#Save cohort analysis data
cohort_activity.to_csv(
    "saas_cohort_activity.csv",
    index=False
)
print(
    "\nSaved: saas_cohort_activity.csv"
)

