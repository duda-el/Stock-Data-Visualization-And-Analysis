import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Scrape NVIDIA Stock Data
def scrape_stock_data(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {'class': 'table yf-1jecxey noDl'})
    data = []

    if table:
        rows = table.find_all('tr')
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) == 7:
                data.append([
                    cols[0].text.strip(),
                    cols[1].text.strip(),
                    cols[2].text.strip(),
                    cols[3].text.strip(),
                    cols[4].text.strip(),
                    cols[5].text.strip(),
                    cols[6].text.strip()
                ])
    else:
        raise ValueError("Table not found on the page. Please check the URL.")

    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    return pd.DataFrame(data, columns=columns)



def clean_data(df):
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
    return df.dropna()


# Step 2: Visualizations
def plot_pie_chart(df):
    df['Month-Year'] = df['Date'].dt.to_period('M')
    monthly_avg_volume = df.groupby('Month-Year')['Volume'].mean()
    labels = monthly_avg_volume.index.astype(str)
    plt.figure(figsize=(10, 8))
    plt.pie(monthly_avg_volume, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Average Monthly Volume Distribution")
    plt.show()

def plot_scatter(df):
    dec_2024 = df[(df['Date'] >= "2024-12-01") & (df['Date'] <= "2024-12-31")]
    plt.figure(figsize=(8, 6))
    plt.scatter(dec_2024['High'], dec_2024['Low'], alpha=0.7, color='blue')
    plt.title("Scatter Plot of High vs Low Prices (December 2024)")
    plt.xlabel("High Prices")
    plt.ylabel("Low Prices")
    plt.grid(True)
    for _, row in dec_2024.iterrows():
        plt.text(row['High'], row['Low'], row['Date'].strftime('%d'), fontsize=8, color='red')
    plt.show()

def plot_bar_chart(df):
    today = datetime.now()
    last_30_days = df[df['Date'] >= today - timedelta(days=30)]
    plt.figure(figsize=(12, 8))
    plt.bar(last_30_days['Date'].dt.strftime('%Y-%m-%d'), last_30_days['Close'], color='orange')
    plt.title("Closing Prices Over the Last 30 Days")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def train_model_on_all_data(df):
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']

    # Use the full dataset for both training and evaluation
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    # Evaluate the model
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y, y_pred):.2f}")
    print(f"R² Score: {r2_score(y, y_pred):.2f}")

    # Add dates for console debugging
    dates = df['Date']

    # Debugging: Print alignment of actual, predicted, and dates in the console
    for actual, predicted, date in zip(y, y_pred, dates):
        print(f"Date: {date.strftime('%Y-%m-%d')}, Actual: {actual}, Predicted: {predicted}")

    # Plot Actual vs Predicted Closing Prices
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.7, color='blue', label="Predicted vs Actual")
    plt.title("Actual vs Predicted Closing Prices (Using 100% Data)")
    plt.xlabel("Actual Closing Prices")
    plt.ylabel("Predicted Closing Prices")
    plt.grid(True)

    # Remove date annotations from the chart (skip plt.text())
    plt.legend()
    plt.show()

    return model



# Main Execution
if __name__ == "__main__":
    url = "https://finance.yahoo.com/quote/NVDA/history?p=NVDA"
    df = scrape_stock_data(url)
    df = clean_data(df)
    plot_pie_chart(df)
    plot_scatter(df)
    plot_bar_chart(df)
    model = train_model_on_all_data(df)

