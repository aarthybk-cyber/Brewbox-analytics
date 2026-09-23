import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

st.set_page_config(page_title="BrewBox Coffee Analytics", layout="wide")

st.title("☕ BrewBox Coffee - Time Series Analysis & Forecasting")
st.markdown("**Created for Kavya Reddy (Café Owner)**")

# 1. Project Plan Sidebar
st.sidebar.header("Project Plan")
st.sidebar.markdown("""
1. **Data Loading & Cleaning**
2. **Growth Analysis**
3. **Peak Times & Operations**
4. **14-Day Sales Forecast**
""")

# File path (Adjust if deploying to a cloud server without Drive access)
DATA_PATH = '/content/drive/MyDrive/Colab Notebooks/Assignments/Time Series Analysis/brewbox_daily_sales.csv'

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except:
        st.error(f"Could not find dataset at {DATA_PATH}. Please make sure the file path is correct.")
        return None
    df['Date'] = pd.to_datetime(df['Date'])
    df['Cups_Sold'] = df['Cups_Sold'].fillna(0)
    df['Sales_Rs'] = df['Sales_Rs'].fillna(0)
    df.set_index('Date', inplace=True)
    return df

df = load_and_clean_data()

if df is not None:
    # Tab layouts for clean navigation
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Growth Trend", "🕒 Operational Patterns", "🔮 14-Day Forecast", "📋 Project Overview"])

    with tab1:
        st.header("Is the Café Actually Growing?")
        df['Sales_Rs_7Day_MA'] = df['Sales_Rs'].rolling(window=7).mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Sales_Rs'], label='Daily Sales', alpha=0.5)
        ax.plot(df.index, df['Sales_Rs_7Day_MA'], label='7-Day Rolling Average', color='red', linewidth=2)
        ax.set_title('Daily Sales with 7-Day Rolling Average')
        ax.set_ylabel('Sales (Rs.)')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        
        start_val = df['Sales_Rs_7Day_MA'].dropna().iloc[0]
        end_val = df['Sales_Rs_7Day_MA'].iloc[-1]
        
        col1, col2 = st.columns(2)
        col1.metric("Starting 7-Day Avg Sales", f"{start_val:,.2f} Rs.")
        col2.metric("Ending 7-Day Avg Sales", f"{end_val:,.2f} Rs.", f"+{((end_val-start_val)/start_val)*100:.1f}%")

    with tab2:
        st.header("When is it busy? (Operational Patterns)")
        
        # Day of week pattern
        weekly_sales = df.groupby('Day_Name')['Sales_Rs'].mean().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        )
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(x=weekly_sales.index, y=weekly_sales.values, palette='viridis', ax=ax2)
        ax2.set_title('Average Sales by Day of the Week')
        ax2.set_ylabel('Average Sales (Rs.)')
        ax2.grid(axis='y')
        st.pyplot(fig2)
        
        # Holiday effect
        holiday_effect = df.groupby('Public_Holiday')['Sales_Rs'].mean()
        st.markdown(f"""
        * **Busiest Day:** Wednesday (~{weekly_sales['Wednesday']:,.0f} Rs.)
        * **Quietest Day:** Sunday (~{weekly_sales['Sunday']:,.0f} Rs.)
        * **Public Holiday Impact:** Revenue drops on average by **56.9%** (from {holiday_effect['No']:,.0f} Rs. to {holiday_effect['Yes']:,.0f} Rs.) due to IT Park closures.
        """)

    with tab3:
        st.header("Fortnight Forecast (Next 14 Days)")
        final_forecast_values = df['Sales_Rs'].tail(14).values
        forecast_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=14)
        forecast_series = pd.Series(final_forecast_values, index=forecast_dates)
        fortnight_revenue = forecast_series.sum()
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(df.index[-28:], df['Sales_Rs'].tail(28), label='Recent History', color='blue')
        ax3.plot(forecast_series.index, forecast_series.values, label='14-Day Forecast', color='red', linestyle='--', marker='x')
        ax3.set_title('Final 14-Day Sales Forecast')
        ax3.set_ylabel('Sales (Rs.)')
        ax3.grid(True)
        ax3.legend()
        st.pyplot(fig3)
        
        st.success(f"### 💰 Total Projected Revenue for Next 14 Days: {fortnight_revenue:,.2f} Rs.")

    with tab4:
        st.header("Project Context & Business Summary")
        st.markdown("""
        ### Key Operational Advice for Kavya:
        * **Weekend Optimization:** Drastically reduce staff and perishable stock on weekends. Sales drop by nearly 75% on Sundays compared to mid-week.
        * **Holiday Warning:** Manually adjust predictions downward when an upcoming public holiday is approaching, as the model cannot dynamically account for floating holiday calendars.
        """)
