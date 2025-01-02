import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Forecasting App", layout="wide")

# Custom CSS for modern buttons
st.markdown("""
    <style>
        .stButton button {
            background-color: #2c3e50;
            color: white;
            font-size: 16px;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #3498db;
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.2);
        }
        .stButton button:active {
            background-color: #2980b9;
        }
    </style>
""", unsafe_allow_html=True)

# App Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Home", "Forecasting"])

# Home Page
if page == "Home":
    st.title("Welcome to the Forecasting App")
    st.write("""
    This application uses the Prophet model to forecast time series data.
    Upload your dataset, select the forecast type (Calls or Chat), choose a forecast range, 
    and visualize the results. Use the **Forecasting** section to get started.
    """, unsafe_allow_html=True)

    # Data structure guide
    st.subheader("Data Structure Guide", anchor="data-structure-guide")
    st.write("""
    The dataset should contain the following columns:
    - **Date**: A column with date values (in any recognizable format).
    - **Calls Offered**: A column with numeric values for calls data.
    - **Chat Offered**: A column with numeric values for chat data.
    
    Example of the expected structure:
    """)

    # Sample data to show the expected structure
    example_data = {
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "Calls Offered": [150, 160, 140, 180, 170],
        "Chat Offered": [90, 95, 85, 110, 100]
    }
    example_df = pd.DataFrame(example_data)
    st.write(example_df)

# Forecasting Page
elif page == "Forecasting":
    st.title("Forecasting with Prophet")

    # File Uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load the uploaded dataset
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview", anchor="data-preview")
        st.write(df.head())

        # Check if required columns are present
        if 'Date' in df.columns and 'Calls Offered' in df.columns and 'Chat Offered' in df.columns:
            # Display data types
            #st.subheader("Data Types", anchor="data-types")
            #st.write(df.dtypes)

            # Select Forecast Type as Dropdown
            forecast_type = st.selectbox("Select Forecast Type", ["Calls Offered", "Chat Offered"])

            # Prepare the data based on selected type
            if forecast_type == "Calls Offered":
                df = df[['Date', 'Calls Offered']].rename(columns={'Date': 'ds', 'Calls Offered': 'y'})
            else:
                df = df[['Date', 'Chat Offered']].rename(columns={'Date': 'ds', 'Chat Offered': 'y'})

            df['ds'] = pd.to_datetime(df['ds'])

            # Prophet Model Training
            model = Prophet()
            model.fit(df)

            # Select Forecast Range
            st.subheader("Select Date Range for Forecasting", anchor="forecast-range")
            min_date = df['ds'].max().date()  # Latest date in the data
            max_date = min_date + timedelta(days=365)  # Default max range is 1 year

            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

            if start_date and end_date:
                # Calculate periods for forecast
                periods = (end_date - min_date).days + 1

                # Create future dataframe and forecast
                future = model.make_future_dataframe(periods=periods)
                df_forecast = model.predict(future)

                # Filter forecast for selected range
                df_forecast = df_forecast[(
                    df_forecast['ds'] >= pd.Timestamp(start_date)) & (df_forecast['ds'] <= pd.Timestamp(end_date))]

                # Display forecasted data
                st.subheader("Forecasted Data", anchor="forecasted-data")
                st.write(df_forecast[['ds', 'yhat']])

                # Plot the forecast
                st.subheader("Forecast Plot", anchor="forecast-plot")
                fig = model.plot(df_forecast)

                # Adjust figure size to medium
                fig.set_size_inches(10, 6)  # Set the width and height of the figure to medium size

                # Adjust labels to improve readability
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Forecast", fontsize=12)

                # Coloring for the plot
                plt.title(f"{forecast_type} Forecast", fontsize=14, color="green")
                st.pyplot(fig)

                # Calculate Accuracy Metrics (using the last portion of data as test set)
                if len(df) > 1:
                    # Using the last 20% of the data for evaluation
                    train_size = int(len(df) * 0.8)
                    df_train = df[:train_size]
                    df_test = df[train_size:]

                    # Create a new Prophet model instance for the training data
                    model = Prophet()
                    model.fit(df_train)

                    # Forecast on the test data
                    future_test = model.make_future_dataframe(periods=len(df_test))
                    df_test_forecast = model.predict(future_test)

                    # Merge the forecast with the test data
                    df_test = df_test.set_index('ds').join(df_test_forecast.set_index('ds')[['yhat']])

                    # Calculate metrics
                    y_true = df_test['y']
                    y_pred = df_test['yhat']

                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)

                    # Display Accuracy Metrics
                    st.subheader("Accuracy Metrics", anchor="accuracy-metrics")
                    st.markdown(f"**Mean Absolute Error (MAE):** {mae:.2f}", unsafe_allow_html=True)
                    st.markdown(f"**Mean Squared Error (MSE):** {mse:.2f}", unsafe_allow_html=True)
                    st.markdown(f"**R-squared (R²):** {r2:.2f}", unsafe_allow_html=True)

                # Prepare CSV for download
                csv_buffer = BytesIO()
                df_forecast[['ds', 'yhat']].to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                # Digital Download Button with modern style
                st.markdown("""
                    <style>
                        .stDownloadButton>button {
                            background-color: #27ae60;
                            color: white;
                            padding: 10px 20px;
                            border-radius: 5px;
                            font-size: 16px;
                            font-weight: bold;
                            border: none;
                            cursor: pointer;
                            transition: all 0.3s ease;
                        }
                        .stDownloadButton>button:hover {
                            background-color: #2ecc71;
                            transform: scale(1.05);
                        }
                    </style>
                """, unsafe_allow_html=True)

                st.download_button(
                    label=f"Download {forecast_type} Forecast as CSV",
                    data=csv_buffer,
                    file_name=f"{forecast_type}_forecast.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Click to download the forecasted data."
                )
        else:
            st.error("The dataset must contain 'Date', 'Calls Offered', and 'Chat Offered' columns.", icon="🚨")
