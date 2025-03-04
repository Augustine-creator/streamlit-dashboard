import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import openai
import gspread
from google.oauth2.service_account import Credentials
from datetime import date, datetime
import smtplib
from email.mime.text import MIMEText
import urllib.parse

# ----------------- CONFIGURATION -----------------
st.set_page_config(page_title="Advanced Analytics for Vehicle Sales Conversion", layout="wide")
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key

# (Optional) Insert your authentication mechanism here (Firebase/Google OAuth)

# ----------------- DATA FETCHING FROM GOOGLE SHEETS -----------------
@st.cache_data(ttl=3600)  # Refreshes data every hour
def load_data():
    # Google Sheets credentials and connection
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    SERVICE_ACCOUNT_FILE = "/content/tmdk-ai-d41a4e33d83f.json"  # Upload your service account JSON file to Colab
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(creds)
    
    # Define your Google Sheet and worksheet details
    SHEET_ID = "1XGsXLYDL1UoBvcVG4pPMPBTHICt9ZeCAHz63qxo-JmQ"
    WORKSHEET_NAME = "DATASET_DB"
    sheet = gc.open_by_key(SHEET_ID).worksheet(WORKSHEET_NAME)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Convert monetary columns to numeric
    for col in ["RETAIL_PRICE", "FINANCED_AMOUNT", "DEPOSIT PAYABLE", "DEPOSIT_PAID", "BALANCE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Convert dates from PAYMENT_DATE
    df["PAYMENT_DATE"] = pd.to_datetime(df["PAYMENT_DATE"], errors="coerce")
    
    # Standardize month names if available
    if "MONTH" in df.columns:
        df["MONTH"] = df["MONTH"].str.capitalize()
    
    # If "NO" column exists, use it as a unique key; otherwise, use all data.
    if "NO" in df.columns:
        df.drop_duplicates(subset=["NO"], inplace=True)
        df.set_index("NO", inplace=True)
    else:
        st.warning("No 'NO' column found; using all available data.")
    return df

df = load_data()

# ----------------- DATA PREPROCESSING -----------------
# Flag converted deals (those with PAYMENT_DATE) and allocated vehicles (non-empty CHASSIS)
df['is_converted'] = df["PAYMENT_DATE"].notnull()
df['is_allocated'] = df["CHASSIS"].astype(str).str.strip().astype(bool)
# Calculate variance (for example: Retail Price minus Financed Amount)
df['variance'] = df["RETAIL_PRICE"] - df["FINANCED_AMOUNT"]

# ----------------- CUSTOM CSS FOR WRAPPING DATA TABLE TEXT -----------------
st.markdown(
    """
    <style>
    .dataframe td {
        white-space: normal;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------- INTERACTIVE SIDEBAR FILTERS -----------------
st.sidebar.header("🔍 Filter Data")

# Slicers for different dimensions
model_filter = st.sidebar.multiselect("Vehicle Model", options=df["MODEL"].unique(), default=list(df["MODEL"].unique()))
financing_filter = st.sidebar.multiselect("Type of Financing", options=df["TYPE OF FINANCING"].unique(), default=list(df["TYPE OF FINANCING"].unique()))
bank_filter = st.sidebar.multiselect("Bank Name", options=df["BANK_NAME"].unique(), default=list(df["BANK_NAME"].unique()))
status_filter = st.sidebar.multiselect("Status", options=df["STATUS"].unique(), default=list(df["STATUS"].unique()))
year_filter = st.sidebar.multiselect("Year", options=df["YEAR"].unique(), default=list(df["YEAR"].unique()))
# Added Month slicer using the MONTH column
month_filter = st.sidebar.multiselect("Month", options=df["MONTH"].unique(), default=list(df["MONTH"].unique()))
ceo_filter = st.sidebar.multiselect("CEO", options=df["CEO"].unique(), default=list(df["CEO"].unique()))
ceo_branch_filter = st.sidebar.multiselect("CEO Branch", options=df["CEO_BRANCH"].unique(), default=list(df["CEO_BRANCH"].unique()))
bank_branch_filter = st.sidebar.multiselect("Bank Branch", options=df["BANK_BRANCH"].unique(), default=list(df["BANK_BRANCH"].unique()))

# Date slicer based on PAYMENT_DATE
if df["PAYMENT_DATE"].notnull().any():
    start_date = st.sidebar.date_input("Start Date", value=df["PAYMENT_DATE"].min().date())
    end_date = st.sidebar.date_input("End Date", value=df["PAYMENT_DATE"].max().date())
else:
    start_date, end_date = date.today(), date.today()

# Build filter mask using all slicers
mask = (
    df["MODEL"].isin(model_filter) &
    df["TYPE OF FINANCING"].isin(financing_filter) &
    df["BANK_NAME"].isin(bank_filter) &
    df["STATUS"].isin(status_filter) &
    df["YEAR"].isin(year_filter) &
    df["MONTH"].isin(month_filter) &
    df["CEO"].isin(ceo_filter) &
    df["CEO_BRANCH"].isin(ceo_branch_filter) &
    df["BANK_BRANCH"].isin(bank_branch_filter)
)
filtered_df = df[mask].copy()

# Apply date filtering (only to converted deals, based on PAYMENT_DATE)
if filtered_df["PAYMENT_DATE"].notnull().any():
    filtered_df = filtered_df[
        (filtered_df["PAYMENT_DATE"] >= pd.Timestamp(start_date)) &
        (filtered_df["PAYMENT_DATE"] <= pd.Timestamp(end_date))
    ]

# ----------------- DASHBOARD HEADER & KPIs -----------------
st.title("🚀 Advanced Analytics for Vehicle Sales Conversion")
st.markdown("## Key Performance Indicators")

# Compute KPIs from filtered data
total_converted = filtered_df[filtered_df["is_converted"]].shape[0]
total_allocated = filtered_df[filtered_df["is_allocated"]].shape[0]
total_retail = filtered_df["RETAIL_PRICE"].sum()
total_financed = filtered_df["FINANCED_AMOUNT"].sum()
total_deposit = filtered_df["DEPOSIT_PAID"].sum()
avg_variance = filtered_df["variance"].mean()

# Display KPIs in columns
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Converted Vehicles", total_converted)
col2.metric("Allocated Vehicles", total_allocated)
col3.metric("Total Retail Price", f"${total_retail:,.2f}")
col4.metric("Total Financed", f"${total_financed:,.2f}")
col5.metric("Total Deposit", f"${total_deposit:,.2f}")
col6.metric("Avg. Variance", f"${avg_variance:,.2f}")

# ----------------- GROUPING ANALYSES -----------------
st.markdown("## Grouping Analyses")

# Group by CEO Branch (wrap text for clarity)
group_ceo_branch = filtered_df.groupby("CEO_BRANCH").agg({"PRIMARY KEY": "count"}).rename(columns={"PRIMARY KEY": "Vehicle Count"}).sort_values("Vehicle Count", ascending=False)
st.subheader("Vehicles by CEO Branch")
st.dataframe(group_ceo_branch)

# Group by CEO
group_ceo = filtered_df.groupby("CEO").agg({"PRIMARY KEY": "count"}).rename(columns={"PRIMARY KEY": "Vehicle Count"}).sort_values("Vehicle Count", ascending=False)
st.subheader("Vehicles by CEO")
st.dataframe(group_ceo)

# Group by Vehicle Model with aggregated metrics
group_model = filtered_df.groupby("MODEL").agg({
    "PRIMARY KEY": "count",
    "RETAIL_PRICE": "sum",
    "FINANCED_AMOUNT": "sum",
    "DEPOSIT_PAID": "sum",
    "variance": "mean"
}).rename(columns={"PRIMARY KEY": "Count", "variance": "Avg Variance"}).sort_values("Count", ascending=False)
st.subheader("Vehicle Model Analysis")
st.dataframe(group_model)

# Group by Bank Name with aggregated metrics
group_bank = filtered_df.groupby("BANK_NAME").agg({
    "PRIMARY KEY": "count",
    "RETAIL_PRICE": "sum",
    "FINANCED_AMOUNT": "sum",
    "DEPOSIT_PAID": "sum",
    "variance": "mean"
}).rename(columns={"PRIMARY KEY": "Count", "variance": "Avg Variance"}).sort_values("Count", ascending=False)
st.subheader("Bank Analysis")
st.dataframe(group_bank)

# Group by Bank Branch (nested under Bank Name)
group_bank_branch = filtered_df.groupby(["BANK_NAME", "BANK_BRANCH"]).agg({
    "PRIMARY KEY": "count",
    "RETAIL_PRICE": "sum",
    "FINANCED_AMOUNT": "sum",
    "DEPOSIT_PAID": "sum",
    "variance": "mean"
}).rename(columns={"PRIMARY KEY": "Count", "variance": "Avg Variance"}).sort_values("Count", ascending=False)
st.subheader("Bank Branch Analysis")
st.dataframe(group_bank_branch)

# Group by Type of Financing
group_financing = filtered_df.groupby("TYPE OF FINANCING").agg({
    "PRIMARY KEY": "count",
    "RETAIL_PRICE": "sum",
    "FINANCED_AMOUNT": "sum",
    "DEPOSIT_PAID": "sum",
    "variance": "mean"
}).rename(columns={"PRIMARY KEY": "Count", "variance": "Avg Variance"}).sort_values("Count", ascending=False)
st.subheader("Type of Financing Analysis")
st.dataframe(group_financing)

# Customer frequency based on CUSTOMER_NAME
group_customer = filtered_df.groupby("CUSTOMER_NAME").agg({"PRIMARY KEY": "count"}).rename(columns={"PRIMARY KEY": "Purchase Frequency"}).sort_values("Purchase Frequency", ascending=False)
st.subheader("Customer Purchase Frequency")
st.dataframe(group_customer)

# ----------------- VISUALIZATIONS -----------------
st.markdown("## Visualizations")
fig1 = px.bar(filtered_df, x="MODEL", y="RETAIL_PRICE", color="TYPE OF FINANCING",
              title="Retail Price by Model", text_auto=True)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.bar(filtered_df, x="BANK_NAME", y="FINANCED_AMOUNT", color="BANK_BRANCH",
              title="Financed Amount by Bank and Branch", text_auto=True)
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.bar(filtered_df, x="CEO", y="FINANCED_AMOUNT", color="CEO_BRANCH",
              title="Financed Amount by CEO (Grouped by CEO Branch)", text_auto=True)
st.plotly_chart(fig3, use_container_width=True)

# ----------------- LLM INSIGHTS -----------------
st.markdown("## LLM Insights")
llm_question = st.text_input("Ask a question for data insights:")
if llm_question:
    prompt = f"Given the following summary:\n" \
             f"- Converted Vehicles: {total_converted}\n" \
             f"- Allocated Vehicles: {total_allocated}\n" \
             f"- Total Retail Price: ${total_retail:,.2f}\n" \
             f"- Total Financed: ${total_financed:,.2f}\n" \
             f"- Total Deposit: ${total_deposit:,.2f}\n" \
             f"- Avg. Variance: ${avg_variance:,.2f}\n" \
             f"and the aggregated analyses provided, please provide an insightful analysis regarding: {llm_question}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    st.markdown("**LLM Analysis:**")
    st.write(response["choices"][0]["message"]["content"])

# ----------------- EMAIL REPORT FUNCTIONALITY -----------------
st.markdown("## Email Report")
with st.form("email_form"):
    recipient_email = st.text_input("Enter recipient email address:")
    submit_email = st.form_submit_button("Send Report")
    if submit_email and recipient_email:
        report_content = f"""
        Advanced Vehicle Sales Conversion Report

        Key Metrics:
        - Converted Vehicles: {total_converted}
        - Allocated Vehicles: {total_allocated}
        - Total Retail Price: ${total_retail:,.2f}
        - Total Financed: ${total_financed:,.2f}
        - Total Deposit: ${total_deposit:,.2f}
        - Average Variance: ${avg_variance:,.2f}

        Grouping Analyses:
        Vehicles by CEO Branch:
        {group_ceo_branch.to_string()}

        Vehicles by CEO:
        {group_ceo.to_string()}

        Vehicle Model Analysis:
        {group_model.to_string()}

        Bank Analysis:
        {group_bank.to_string()}

        Bank Branch Analysis:
        {group_bank_branch.to_string()}

        Type of Financing Analysis:
        {group_financing.to_string()}

        Customer Purchase Frequency:
        {group_customer.to_string()}

        LLM Insights:
        {response["choices"][0]["message"]["content"] if llm_question else "N/A"}
        """
        try:
            msg = MIMEText(report_content)
            msg["Subject"] = "Advanced Vehicle Sales Conversion Report"
            msg["From"] = "your_email@gmail.com"  # Replace with sender email
            msg["To"] = recipient_email
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login("your_email@gmail.com", "your_email_password")  # Replace with your credentials
                server.sendmail("your_email@gmail.com", recipient_email, msg.as_string())
            st.success("Report sent successfully!")
        except Exception as e:
            st.error(f"Failed to send email: {e}")

st.markdown("---")
st.caption("Advanced Analytics for Vehicle Sales Conversion | Powered by Streamlit, OpenAI, and Google Sheets")
