import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of leads
N = 5000

# ---------------------------------------------------
# 1. Basic categories
# ---------------------------------------------------

industries = [
    "Technology",
    "Healthcare",
    "Finance",
    "Retail",
    "Education",
    "Manufacturing",
    "Real Estate",
    "Marketing"
]

company_sizes = ["Small", "Medium", "Large"]

website_quality = [
    "Poor",
    "Average",
    "Good",
    "Excellent"
]

linkedin_activity = [
    "Low",
    "Medium",
    "High"
]

locations = [
    "Lahore",
    "Karachi",
    "Islamabad",
    "Rawalpindi",
    "Faisalabad",
    "Multan",
    "Peshawar",
    "Quetta"
]

previous_response = [
    "No Response",
    "Negative",
    "Neutral",
    "Positive"
]

meeting_status = [
    "Not Contacted",
    "Not Scheduled",
    "Scheduled",
    "Completed"
]

# ---------------------------------------------------
# 2. Generate features
# ---------------------------------------------------

df = pd.DataFrame({

    "Industry": np.random.choice(
        industries,
        N,
        p=[0.20, 0.12, 0.15, 0.12, 0.10, 0.12, 0.09, 0.10]
    ),

    "Company_Size": np.random.choice(
        company_sizes,
        N,
        p=[0.45, 0.35, 0.20]
    ),

    "Website_Quality": np.random.choice(
        website_quality,
        N,
        p=[0.12, 0.28, 0.38, 0.22]
    ),

    "LinkedIn_Activity": np.random.choice(
        linkedin_activity,
        N,
        p=[0.30, 0.45, 0.25]
    ),

    "Email_Available": np.random.choice(
        ["Yes", "No"],
        N,
        p=[0.82, 0.18]
    ),

    "Location": np.random.choice(
        locations,
        N
    ),

    "Estimated_Revenue": np.random.lognormal(
        mean=np.log(1500000),
        sigma=1.0,
        size=N
    ).round(-3),

    "Number_of_Employees": np.random.lognormal(
        mean=np.log(100),
        sigma=0.9,
        size=N
    ).astype(int).clip(5, 5000),

    "Previous_Response": np.random.choice(
        previous_response,
        N,
        p=[0.30, 0.12, 0.23, 0.35]
    ),

    "Meeting_Status": np.random.choice(
        meeting_status,
        N,
        p=[0.20, 0.30, 0.30, 0.20]
    )
})


# ---------------------------------------------------
# 3. Create meaningful lead-quality score
# ---------------------------------------------------

score = np.zeros(N)


# Website quality
score += df["Website_Quality"].map({
    "Poor": 0,
    "Average": 1,
    "Good": 2,
    "Excellent": 3
})


# LinkedIn activity
score += df["LinkedIn_Activity"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})


# Email availability
score += df["Email_Available"].map({
    "No": 0,
    "Yes": 2
})


# Company size
score += df["Company_Size"].map({
    "Small": 0,
    "Medium": 1,
    "Large": 2
})


# Previous response
score += df["Previous_Response"].map({
    "No Response": 0,
    "Negative": 0,
    "Neutral": 1,
    "Positive": 3
})


# Meeting status
score += df["Meeting_Status"].map({
    "Not Contacted": 0,
    "Not Scheduled": 0.5,
    "Scheduled": 2,
    "Completed": 3
})


# Revenue contribution
score += np.log1p(df["Estimated_Revenue"]) / 5


# Employee contribution
score += np.log1p(df["Number_of_Employees"]) / 3


# ---------------------------------------------------
# 4. Add realistic variation/noise
# ---------------------------------------------------

# Keep moderate variation while making the synthetic labels learnable from the
# available business features.
noise = np.random.normal(0, 0.6, N)

score = score + noise


# ---------------------------------------------------
# 5. Convert score into High / Medium / Low
# ---------------------------------------------------

low_threshold = np.percentile(score, 35)
high_threshold = np.percentile(score, 70)

df["Lead_Quality"] = np.where(
    score >= high_threshold,
    "High",
    np.where(
        score >= low_threshold,
        "Medium",
        "Low"
    )
)


# ---------------------------------------------------
# 6. Add missing values for preprocessing practice
# ---------------------------------------------------

missing_columns = [
    "Website_Quality",
    "LinkedIn_Activity",
    "Estimated_Revenue",
    "Previous_Response"
]

for column in missing_columns:

    missing_indices = np.random.choice(
        df.index,
        size=int(N * 0.02),
        replace=False
    )

    df.loc[missing_indices, column] = np.nan


# ---------------------------------------------------
# 7. Add a few duplicate rows
# ---------------------------------------------------

duplicates = df.sample(
    25,
    random_state=42
)

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)


# ---------------------------------------------------
# 8. Shuffle dataset
# ---------------------------------------------------

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ---------------------------------------------------
# 9. Save CSV
# ---------------------------------------------------

file_name = "lead_quality_dataset.csv"

df.to_csv(
    file_name,
    index=False
)


# ---------------------------------------------------
# 10. Display information
# ---------------------------------------------------

print("\n======================================")
print("LEAD QUALITY DATASET CREATED")
print("======================================")

print(f"\nFile: {file_name}")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())

print("\nLead Quality Distribution:")
print(df["Lead_Quality"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset saved successfully!")