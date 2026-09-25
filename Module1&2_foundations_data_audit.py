import pandas as pd
import numpy as np
from pathlib import Path
# Create a reusable CSV loader
def load_csv(file_path):
    """
    Load a CSV file safely with exception handling.
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            raise ValueError(f"{file_path} is empty.")

        print(f"Successfully loaded: {file_path}")
        print(f"Shape: {df.shape}")

        return df
    except FileNotFoundError:
        print(f"ERROR: File not found -> {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"ERROR: File is empty -> {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"ERROR: CSV file is malformed -> {file_path}")
        return None
    except UnicodeDecodeError:
        print(f"ERROR: File encoding problem -> {file_path}")
        return None
    except Exception as e:
        print(f"ERROR while loading {file_path}: {e}")
        return None

# Utility Function 1: Validate columns

def validate_columns(df, expected_columns, table_name):
    """
    Check whether expected columns exist.
    """
    actual_columns = list(df.columns)
    missing_columns = [
        col for col in expected_columns
        if col not in actual_columns
    ]
    extra_columns = [
        col for col in actual_columns
        if col not in expected_columns
    ]
    print(f"\nColumn validation: {table_name}")
    if missing_columns:
        print("Missing columns:", missing_columns)
    else:
        print("All expected columns are present.")
    if extra_columns:
        print("Extra columns:", extra_columns)

# Utility Function 2: Basic audit
def audit_dataframe(df, table_name):
    """
    Perform a basic data quality audit.
    """
    print("\n" + "=" * 60)
    print(f"AUDIT: {table_name}")
    print("=" * 60)
    print("\nShape:")
    print(df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isna().sum())
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())
    print("\nUnique values for text columns:")
    text_columns = df.select_dtypes(include="object").columns
    for column in text_columns:
        print(f"\n{column}:")
        print(df[column].dropna().unique())

# Utility Function 3: Cleaning log

cleaning_log = []
def add_cleaning_log(table, issue, rows_affected, action, reason):
    cleaning_log.append({
        "Table": table,
        "Issue": issue,
        "Rows Affected": rows_affected,
        "Action": action,
        "Why": reason
    })

# MODULE 1 — Load the four tables
customers = load_csv("saas_customers.csv")
subscriptions = load_csv("saas_subscriptions.csv")
usage = load_csv("saas_usage.csv")
tickets = load_csv("saas_tickets.csv")

#loading worked
if any(df is None for df in [
    customers,
    subscriptions,
    usage,
    tickets
]):
    raise SystemExit("One or more files could not be loaded.")

#Validate columns BEFORE cleaning
#Customers
validate_columns(
    customers,
    [
        "CustomerID",
        "CompanyName",
        "Industry",
        "Country",
        "City",
        "EmployeeCount",
        "SignupDate",
        "AcquisitionChannel"
    ],
    "Customers"
)
# Subscriptions
validate_columns(
    subscriptions,
    [
        "SubscriptionID",
        "CustomerID",
        "PlanName",
        "BillingTerm",
        "Seats",
        "MRR",
        "StartDate",
        "EndDate",
        "Status"
    ],
    "Subscriptions"
)
# Usage
validate_columns(
    usage,
    [
        "CustomerID",
        "SubscriptionID",
        "Month",
        "Logins",
        "ActiveUsers",
        "FeatureUsed",
        "APICalls",
        "SessionMinutes"
    ],
    "Usage"
)
# Tickets
validate_columns(
    tickets,
    [
        "TicketID",
        "CustomerID",
        "OpenedDate",
        "Category",
        "Priority",
        "ResolutionHours",
        "SatisfactionScore"
    ],
    "Tickets"
)
#Run the FULL initial audit
audit_dataframe(customers, "Customers")
audit_dataframe(subscriptions, "Subscriptions")
audit_dataframe(usage, "Usage")
audit_dataframe(tickets, "Tickets")

# MODULE 2 — Data Audit & Cleaning
#Remove exact duplicate rows
def remove_duplicates(df, table_name):
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates().copy()

        add_cleaning_log(
            table_name,
            "Exact duplicate rows",
            duplicate_count,
            "Removed exact duplicate rows",
            "Duplicate records would cause double counting and incorrect analysis."
        )

    return df
customers = remove_duplicates(customers, "Customers")
subscriptions = remove_duplicates(subscriptions, "Subscriptions")
usage = remove_duplicates(usage, "Usage")
tickets = remove_duplicates(tickets, "Tickets")

#Standardize text columns
#Customers
# Remove unnecessary spaces
for column in [
    "CustomerID",
    "CompanyName",
    "Industry",
    "Country",
    "City",
    "AcquisitionChannel"
]:
    customers[column] = customers[column].str.strip()
#Standardize IDs:
customers["CustomerID"] = customers["CustomerID"].str.upper()
#Standardize categories:
for column in [
    "Industry",
    "Country",
    "City",
    "AcquisitionChannel"
]:
    customers[column] = customers[column].str.title()

#Standardize Subscriptions
for column in [
    "SubscriptionID",
    "CustomerID",
    "PlanName",
    "BillingTerm",
    "Status"
]:
    subscriptions[column] = subscriptions[column].str.strip()
#IDs
subscriptions["SubscriptionID"] = (
    subscriptions["SubscriptionID"].str.upper()
)

subscriptions["CustomerID"] = (
    subscriptions["CustomerID"].str.upper()
)
#Categories:
for column in [
    "PlanName",
    "BillingTerm",
    "Status"
]:
    subscriptions[column] = subscriptions[column].str.title()

#Standardize Usage
for column in [
    "CustomerID",
    "SubscriptionID",
    "FeatureUsed"
]:
    usage[column] = usage[column].str.strip()
#IDs
usage["CustomerID"] = usage["CustomerID"].str.upper()
usage["SubscriptionID"] = usage["SubscriptionID"].str.upper()
#Feature:
usage["FeatureUsed"] = usage["FeatureUsed"].str.title()
usage["FeatureUsed"] = usage["FeatureUsed"].replace({
    "Api": "API"
})

#Standardize Tickets
for column in [
    "TicketID",
    "CustomerID",
    "Category",
    "Priority"
]:
    tickets[column] = tickets[column].str.strip()
#IDs
tickets["TicketID"] = tickets["TicketID"].str.upper()
tickets["CustomerID"] = tickets["CustomerID"].str.upper()
#Categories:
tickets["Category"] = tickets["Category"].str.title()
tickets["Priority"] = tickets["Priority"].str.title()

#Handle Customers missing Industry
count = customers["Industry"].isna().sum()
customers["Industry"] = customers["Industry"].fillna("Unknown")
add_cleaning_log(
    "Customers",
    "Missing Industry",
    count,
    "Replace missing values with 'Unknown'",
    "Industry cannot be reliably inferred from the available fields, so 'Unknown' avoids inventing information."
)
#Handle missing AcquisitionChannel
count = customers["AcquisitionChannel"].isna().sum()
customers["AcquisitionChannel"] = (
    customers["AcquisitionChannel"].fillna("Unknown")
)
add_cleaning_log(
    "Customers",
    "Missing AcquisitionChannel",
    count,
    "Replace missing values with 'Unknown'",
    "The original acquisition source cannot be reliably inferred."
)
#Handle missing EmployeeCount
count = customers["EmployeeCount"].isna().sum()
industry_median = (
    customers.groupby("Industry")["EmployeeCount"]
    .transform("median")
)
customers["EmployeeCount"] = (
    customers["EmployeeCount"]
    .fillna(industry_median)
)
add_cleaning_log(
    "Customers",
    "Missing EmployeeCount",
    count,
    "Fill using median EmployeeCount within Industry",
    "Company size can vary by industry, so an industry-level median is more appropriate than zero or a global median."
)
#Handle missing Seats
count = subscriptions["Seats"].isna().sum()
plan_median = (
    subscriptions.groupby("PlanName")["Seats"]
    .transform("median")
)
subscriptions["Seats"] = (
    subscriptions["Seats"]
    .fillna(plan_median)
)
add_cleaning_log(
    "Subscriptions",
    "Missing Seats",
    count,
    "Fill using median Seats within PlanName",
    "Seat requirements differ by subscription plan."
)
#Handle missing MRR
count = subscriptions["MRR"].isna().sum()
mrr_median = (
    subscriptions
    .groupby(["PlanName", "BillingTerm"])["MRR"]
    .transform("median")
)
subscriptions["MRR"] = (
    subscriptions["MRR"]
    .fillna(mrr_median)
)
add_cleaning_log(
    "Subscriptions",
    "Missing MRR",
    count,
    "Fill using median MRR within PlanName and BillingTerm",
    "MRR depends on both the plan and billing term, so this preserves the structure of the data."
)
#Understand EndDate correctly
print(
    pd.crosstab(
        subscriptions["Status"],
        subscriptions["EndDate"].isna()
    )
)
add_cleaning_log(
    "Subscriptions",
    "Missing EndDate",
    318,
    "No replacement; retain as missing",
    "Missing EndDate is expected for Active and Paused subscriptions; filling it would create false churn dates."
)
#Handle Usage missing ActiveUsers
count = usage["ActiveUsers"].isna().sum()
active_users_median = (
    usage.groupby("FeatureUsed")["ActiveUsers"]
    .transform("median")
)
usage["ActiveUsers"] = (
    usage["ActiveUsers"]
    .fillna(active_users_median)
)
add_cleaning_log(
    "Usage",
    "Missing ActiveUsers",
    count,
    "Fill using median ActiveUsers within FeatureUsed",
    "Usage patterns differ by feature, so a feature-level median is more appropriate than zero."
)
#Handle SessionMinutes
count = usage["SessionMinutes"].isna().sum()
session_median = (
    usage.groupby("FeatureUsed")["SessionMinutes"]
    .transform("median")
)
usage["SessionMinutes"] = (
    usage["SessionMinutes"]
    .fillna(session_median)
)
add_cleaning_log(
    "Usage",
    "Missing SessionMinutes",
    count,
    "Fill using median SessionMinutes within FeatureUsed",
    "Session duration varies by feature."
)
#Handle Ticket missing ResolutionHours
count = tickets["ResolutionHours"].isna().sum()
resolution_median = (
    tickets
    .groupby(["Category", "Priority"])["ResolutionHours"]
    .transform("median")
)
tickets["ResolutionHours"] = (
    tickets["ResolutionHours"]
    .fillna(resolution_median)
)
add_cleaning_log(
    "Tickets",
    "Missing ResolutionHours",
    count,
    "Fill using median within Category and Priority",
    "Resolution time depends on ticket type and priority."
)
#Handle SatisfactionScore
count = tickets["SatisfactionScore"].isna().sum()
satisfaction_median = (
    tickets
    .groupby(["Category", "Priority"])["SatisfactionScore"]
    .transform("median")
)
tickets["SatisfactionScore"] = (
    tickets["SatisfactionScore"]
    .fillna(satisfaction_median)
)
add_cleaning_log(
    "Tickets",
    "Missing SatisfactionScore",
    count,
    "Fill using median within Category and Priority",
    "Customer satisfaction can vary by ticket type and priority."
)
#Convert dates to proper datetime
customers["SignupDate"] = pd.to_datetime(
    customers["SignupDate"],
    errors="coerce"
)
subscriptions["StartDate"] = pd.to_datetime(
    subscriptions["StartDate"],
    errors="coerce"
)
subscriptions["EndDate"] = pd.to_datetime(
    subscriptions["EndDate"],
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
#Find orphan customers
customer_ids = set(customers["CustomerID"])
subscription_customer_ids = set(
    subscriptions["CustomerID"]
)
orphan_subscription_customers = (
    subscription_customer_ids - customer_ids
)
print(orphan_subscription_customers)
#Remove orphan subscriptions
mask = ~subscriptions["CustomerID"].isin(customer_ids)
count = mask.sum()
subscriptions = subscriptions.loc[~mask].copy()
add_cleaning_log(
    "Subscriptions",
    "Orphan CustomerID",
    count,
    "Remove subscription rows with CustomerID not present in Customers",
    "A subscription cannot be reliably analyzed without a valid customer."
)
#Remove orphan Usage rows
valid_customer_ids = set(customers["CustomerID"])
valid_subscription_ids = set(subscriptions["SubscriptionID"])
mask = (
    ~usage["CustomerID"].isin(valid_customer_ids)
    |
    ~usage["SubscriptionID"].isin(valid_subscription_ids)
)
count = mask.sum()
usage = usage.loc[~mask].copy()
add_cleaning_log(
    "Usage",
    "Orphan CustomerID/SubscriptionID",
    count,
    "Remove usage rows without a valid CustomerID and SubscriptionID",
    "Usage must belong to a valid customer and subscription."
)
#Remove orphan Tickets
mask = ~tickets["CustomerID"].isin(valid_customer_ids)
count = mask.sum()
tickets = tickets.loc[~mask].copy()
add_cleaning_log(
    "Tickets",
    "Orphan CustomerID",
    count,
    "Remove ticket rows with CustomerID not present in Customers",
    "A ticket must belong to a valid customer."
)
#Re-check referential integrity
print(
    "Subscription orphan customers:",
    len(
        set(subscriptions["CustomerID"])
        - set(customers["CustomerID"])
    )
)
print(
    "Usage orphan customers:",
    len(
        set(usage["CustomerID"])
        - set(customers["CustomerID"])
    )
)
print(
    "Ticket orphan customers:",
    len(
        set(tickets["CustomerID"])
        - set(customers["CustomerID"])
    )
)
print(
    "Usage orphan subscriptions:",
    len(
        set(usage["SubscriptionID"])
        - set(subscriptions["SubscriptionID"])
    )
)
#Re-run the audit
audit_dataframe(customers, "Customers - CLEANED")
audit_dataframe(subscriptions, "Subscriptions - CLEANED")
audit_dataframe(usage, "Usage - CLEANED")
audit_dataframe(tickets, "Tickets - CLEANED")

#Check remaining missing values
print("\nCustomers missing:")
print(customers.isna().sum())
print("\nSubscriptions missing:")
print(subscriptions.isna().sum())
print("\nUsage missing:")
print(usage.isna().sum())
print("\nTickets missing:")
print(tickets.isna().sum())

#duplicates are gone
print("Customers duplicates:", customers.duplicated().sum())
print("Subscriptions duplicates:", subscriptions.duplicated().sum())
print("Usage duplicates:", usage.duplicated().sum())
print("Tickets duplicates:", tickets.duplicated().sum())

#text standardization
print("Customer Industries:")
print(sorted(customers["Industry"].dropna().unique()))
print("\nCountries:")
print(sorted(customers["Country"].dropna().unique()))
print("\nPlan Names:")
print(sorted(subscriptions["PlanName"].dropna().unique()))
print("\nBilling Terms:")
print(sorted(subscriptions["BillingTerm"].dropna().unique()))
print("\nFeatures:")
print(sorted(usage["FeatureUsed"].dropna().unique()))
print("\nPriorities:")
print(sorted(tickets["Priority"].dropna().unique()))

#Save the Data Cleaning Log
cleaning_log_df = pd.DataFrame(cleaning_log)
cleaning_log_df.to_csv(
    "saas_data_cleaning_log.csv",
    index=False
)
print("\nCleaning log saved successfully.")

#Save the cleaned datasets
customers.to_csv(
    "cleaned_saas_customers.csv",
    index=False
)
subscriptions.to_csv(
    "cleaned_saas_subscriptions.csv",
    index=False
)
usage.to_csv(
    "cleaned_saas_usage.csv",
    index=False
)
tickets.to_csv(
    "cleaned_saas_tickets.csv",
    index=False
)
print("All cleaned files saved successfully.")

