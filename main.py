import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# URL for the NVIDIA stock history page
url = "https://finance.yahoo.com/quote/NVDA/history?p=NVDA"

# Send an HTTP GET request to the page
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
response.raise_for_status()

# Parse the HTML content with BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing the historical stock data
table = soup.find('table', {'class': 'table yf-j5d1ld noDl'})

if table:
    # Extract table rows
    rows = table.find_all('tr')

    # Prepare lists for storing the extracted data
    data = []

    # Loop through rows and extract data
    for row in rows[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) == 7:  # Ensure the row has the correct number of columns
            date = cols[0].text.strip()
            open_price = cols[1].text.strip()
            high = cols[2].text.strip()
            low = cols[3].text.strip()
            close = cols[4].text.strip()
            adj_close = cols[5].text.strip()
            volume = cols[6].text.strip()
            data.append([date, open_price, high, low, close, adj_close, volume])

    # Create a DataFrame for the extracted data
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df = pd.DataFrame(data, columns=columns)

    # Save the DataFrame to a CSV file
    df.to_csv("nvidia_stock_data.csv", index=False)
    print("Data successfully scraped and saved to 'nvidia_stock_data.csv'.")
    print(df.head())
else:
    print("Table not found on the page. Please check the page structure or URL.")


file_path = "nvidia_stock_data.csv"
df = pd.read_csv(file_path)

# Convert
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'].str.replace(',', ''), errors='coerce')
df = df.dropna()


# Add a 'Month-Year' column for grouping
df['Month-Year'] = df['Date'].dt.to_period('M')

# Group by 'Month-Year' and calculate average volume
monthly_avg_volume = df.groupby('Month-Year')['Volume'].mean().dropna()


labels = monthly_avg_volume.index.astype(str)
avg_volumes = monthly_avg_volume.values


plt.figure(figsize=(10, 8))
plt.pie(avg_volumes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title("Average Monthly Volume Distribution")
plt.show()


# Scatter Plot chart starts here #
january_2024_data = df[(df['Date'] >= "2024-12-01") & (df['Date'] <= "2024-12-31")].dropna(subset=['High', 'Low'])

plt.figure(figsize=(8, 6))
plt.scatter(january_2024_data['High'], january_2024_data['Low'], alpha=0.7, color='blue')
plt.title("Scatter Plot of High vs Low Prices (December 2024)")
plt.xlabel("High Prices")
plt.ylabel("Low Prices")
plt.grid(True)

for i, row in january_2024_data.iterrows():
    plt.text(row['High'], row['Low'], row['Date'].strftime('%d'), fontsize=8, color='red')

plt.show()


# Bar chart starts from here #
today = datetime.now()
last_30_days = df[df['Date'] >= today - timedelta(days=30)].copy()  # Create a copy to avoid warnings

plt.figure(figsize=(12, 8))
plt.bar(last_30_days['Date'].dt.strftime('%Y-%m-%d'), last_30_days['Close'], color='orange')
plt.title("Closing Prices Over the Last 30 Days", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Closing Price", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()