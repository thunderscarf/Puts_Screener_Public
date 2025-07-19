# Puts Screener

This Python script automates the screening of U.S. stocks in the Nasdaq 100 Index for **cash-secured put selling** opportunities based on trend and momentum signals. It pulls option chain data, calculates yields, and exports filtered results to an Excel file.
My main data provider is **Polygon**, so you would need to have Polygon API Key to utilise this screener.

---

## Features

- Screens options based on:
  - Price trend regimes
  - Relative Strength Index (RSI)
  - Z-score of price movement
  - % from 400-day highs
- Selects **put options** that are ~25% out-of-the-money
- Supports multiple expiries (`2025-09-19`, `2025-12-19`, `2026-03-20`) - can edit as you see fit
- Calculates:
  - Option yield
  - Bid-ask spread
  - Greeks (delta, gamma, theta, vega)
  - Implied volatility
- Exports results to an Excel workbook with sheets by expiry date

---

## To run the volatility screener: 
1. Double click the execute.bat file
2. When the program is done running, the output file can be retrieved from the excel_output folder, under the name Puts_Screener_{date}.xlsx, 
   where is the last traded date of the option. (i.e. if you run on 23-Apr 10am SGT, the file name will be Puts_Screener_2025-04-22)



