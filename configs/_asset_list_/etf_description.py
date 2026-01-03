import yfinance as yf
import pandas as pd

# List of ETF tickers
etf_tickers = [
    'USO', 'UNG', 'GLD', 'SLV', 'SPY', 'QQQ', 'IWM',
    'VIXY', 'VIXM', 'UUP', 'FXE', 'DBC', 'PDBC', 'CORN',
    'WEAT', 'SOYB', 'COW', 'BNO', 'UGA', 'KOLD', 'BOIL',
    'PPLT', 'PALL', 'DBB', 'CANE', 'JO', 'NIB', 'FXY',
    'FXB', 'FXC', 'FXA', 'CYB'
]

# Create an empty list to store ETF metadata
etf_metadata = []

# Fetch metadata for each ETF ticker
for ticker in etf_tickers:
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        etf_metadata.append({
            "Ticker": ticker,
            "ETF Name": info.get("longName", "N/A"),
            "Description": info.get("longBusinessSummary", "N/A"),
            "Category": info.get("category", "N/A")
        })
        print(f"Retrieved data for {ticker}")
    except Exception as e:
        print(f"Failed to retrieve data for {ticker}: {e}")
        etf_metadata.append({
            "Ticker": ticker,
            "ETF Name": "N/A",
            "Description": "N/A",
            "Category": "N/A"
        })

# Convert the data to a DataFrame
etf_df = pd.DataFrame(etf_metadata)

# Save to Excel
output_file = "./ETF_Metadata.csv"
etf_df.to_csv(output_file, index=False)


# for each category, get the ETFs
# store the ETFs in a dictionary with the category as the key
# save the dictionary to a txt file
etf_dict = {}
categories = etf_df["Category"].unique()
for category in categories:
    etfs = etf_df[etf_df["Category"] == category]["Ticker"].tolist()
    etf_dict[category] = etfs

output_file = "./ETF_Categories.txt"
with open(output_file, "w") as f:
    for category, etfs in etf_dict.items():
        f.write(f"{category}: {etfs}\n")

print(f"Metadata saved to {output_file}")
