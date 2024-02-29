import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns  # Import seaborn for heatmap
from scipy.optimize import minimize

# Disable warning about PyplotGlobalUse
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to fetch stock data
def get_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    return stock_data

# Function to fetch index data
def get_index_data(indices, start_date, end_date):
    index_data = yf.download(indices, start=start_date, end=end_date)
    return index_data

# Function to normalize data to the starting value
def normalize_data(data):
    return data / data.iloc[0]

# Streamlit app title
st.title('Financial Data Analytics with Python - Project')

# Sidebar for user input
st.sidebar.header('User Input Parameters')
available_stocks = ['0005.HK', '0011.HK', '2388.HK', 'NVDA', 'SMCI', 'AAPL']
selected_stocks = st.sidebar.multiselect('Select stocks for analysis', available_stocks, default=available_stocks[:1])
number_of_neighbors = st.sidebar.slider('Number of neighbors for kNN', 1, 20, 5)
start_date = st.sidebar.date_input('Start date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End date', pd.to_datetime('2025-01-01'))

# Get stock data
if len(selected_stocks) > 1:
    stock_data = get_data(selected_stocks, start_date, end_date)

    if 'Date' not in stock_data.columns:
        stock_data.reset_index(inplace=True)

    simple_returns = stock_data['Close'].pct_change().dropna()

    st.header('Correlation Matrix for Selected Stocks')

    # Calculate correlation matrix
    correlation_matrix = simple_returns.corr()

    # Create heatmap for correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    st.pyplot()

    st.header('Closing Price')
    closing_prices = stock_data['Close']
    st.line_chart(closing_prices)

    st.header('Simple Returns')
    st.line_chart(simple_returns)

    st.header('Logarithmic Returns')
    log_returns = np.log(closing_prices / closing_prices.shift(1)).dropna()
    st.line_chart(log_returns)

    annual_returns = simple_returns.mean() * 252
    st.header('Annualized Returns')
    st.write(annual_returns)

    st.header('kNN Stock Movement Prediction')
    returns = closing_prices.pct_change().dropna()
    predictions = {}
    for stock in selected_stocks:
        X = returns[:-1]
        y = (returns[stock][1:] > 0).astype(int)

        knn = KNeighborsClassifier(n_neighbors=number_of_neighbors)
        knn.fit(X, y)

        predicted_movement = knn.predict(X.iloc[-1].values.reshape(1, -1))
        predictions[stock] = 'Up' if predicted_movement == 1 else 'Down'

    st.write(predictions)

    # Calculate mean returns and covariance matrix
    mean_returns = simple_returns.mean()
    cov_matrix = simple_returns.cov()

    # Calculate mean, standard deviation, and variance for selected stocks
    mean_returns = simple_returns.mean()
    std_returns = simple_returns.std()
    var_returns = simple_returns.var()

# Check if lengths match
    if len(mean_returns) == len(std_returns) == len(var_returns) == len(selected_stocks):
        stats_df = pd.DataFrame({
            'Mean': mean_returns,
            'Standard Deviation': std_returns,
            'Variance': var_returns
            })
    
    # Display the dataframe
        st.header('Mean, Standard Deviation, and Variance for Selected Stocks')
        st.write(stats_df)
    else:
            st.write('Error: Lengths of calculated statistics do not match the number of selected stocks.')


    st.header('Covariance Matrix for Selected Stocks')
    st.write(cov_matrix)

else:
    st.write('Please select at least two stocks for analysis.')

# Download and analyze index data
indices = ['^GSPC', '^HSI']
index_data = get_index_data(indices, start_date, end_date)

if not index_data.empty:
    st.header('Index Data')

    # Normalize index data to the starting value
    normalized_index_data = normalize_data(index_data['Close'])
    st.line_chart(normalized_index_data)

    # Calculate daily returns of the index data
    index_returns = index_data['Close'].pct_change().dropna()
    st.header('Index Daily Returns')
    st.line_chart(index_returns)

    # Calculate annualized returns of the index data
    index_annual_returns = index_returns.mean()
