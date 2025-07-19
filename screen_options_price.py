import pandas as pd
from vol_arb_utils import *
import json  
from rich import print
from rich.progress import track
import pandas_market_calendars as mcal

API_KEY = "peepeepoopoo"

#=========================================================================================
#================================== VARIABLES ============================================ 
#=========================================================================================
# in YYYY-MM-DD Format
#
EXPIRY_DATES = ['2025-09-19', '2025-12-19','2026-03-20']
OPTION_PERC = 0.25

#=========================================================================================
#========================= SCREENER CODES ================================================
#=========================================================================================

DATE = get_last_traded_date()
print(f'[bold]Initialising Volatility Screener:[/bold]')
print(f'Puts Expiry Dates: [bold cyan]{EXPIRY_DATES}[/bold cyan]')
print(f'Puts Moneyness from ATM: [bold cyan]{OPTION_PERC*100:.0f}%[/bold cyan]')

# Load the JSON file back into Python
with open("polygon_tickers_ls.json", 'r') as json_file:
    polygon_tickers_ls = json.load(json_file)

all_options_df = pd.DataFrame()
for polygon_ticker in track(polygon_tickers_ls):
    try:
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.valid_days(start_date= pd.to_datetime(DATE) +pd.Timedelta(days=-400), end_date=DATE)
        start_date = trading_days[0]

        ohlcv = get_ohlcv_df(polygon_ticker, start_date= start_date.strftime('%Y-%m-%d'), end_date = DATE, timeframe='1/day', adjusted= 'false')
        df_w_trend_features = get_trend_regime_wo_beta(ohlcv, sma_period = 63, rsi_period = 14)
        
        close_price = df_w_trend_features['Close'].iloc[-1]
        perc_from_high_score = df_w_trend_features['perc_from_high_score'].iloc[-1]
        RSI_score = df_w_trend_features['RSI_score'].iloc[-1]
        z_score = df_w_trend_features['z_score'].iloc[-1]
        last_combined_score = df_w_trend_features['combined_score'].iloc[-1]

        condition1 = (perc_from_high_score == 1)
        condition2 = ((RSI_score == 1) or (z_score == 1))
        
        if condition1 and condition2: 
            for expiry in EXPIRY_DATES:

                sell_put_strike_price = close_price * (1-OPTION_PERC)
                contract = get_puts_contract_by_exp(polygon_ticker, DATE, sell_put_strike_price, expiry)

                option_ticker = contract['ticker'].iloc[0]
                option_strike = contract['strike_price'].iloc[0]
                option_expiry = contract['expiration_date'].iloc[0]

                quotes_df = get_option_quotes(option_ticker, DATE)
                quotes_df = quotes_df[(quotes_df['ask_size'] >= 1) & (quotes_df['bid_size'] >= 1)]
                last_quote = quotes_df.iloc[0]

                bid_ask_spread = last_quote['ask_price'] - last_quote['bid_price']
                midprice = (last_quote['ask_price'] + last_quote['bid_price'])/2
                total_quote_size = last_quote['ask_size'] + last_quote['bid_size']
                quotes_timestamp = last_quote['sip_timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                option_yield = (midprice/option_strike)*100
                spread_against_midprice = bid_ask_spread / midprice
                delta, gamma, theta, vega, iv = get_options_greeks_and_iv(polygon_ticker, option_ticker)

                option_dict = {
                    "stock_ticker" : polygon_ticker,
                    "option_ticker" : option_ticker,
                    "strike_price" : option_strike,
                    "expiration_date" : option_expiry, 
                    "quotes_timestamp" : quotes_timestamp,
                    "ask_size" : last_quote['ask_size'],
                    "bid_size" : last_quote['bid_size'],
                    "latest_bid" : last_quote['bid_price'],
                    "latest_ask"  : last_quote['ask_price'], 
                    "bid_ask_spread" : bid_ask_spread,
                    "mid_price" : midprice,
                    "spread_against_midprice" : spread_against_midprice,
                    "RSI_score" : RSI_score,
                    "perc_from_high_score" : perc_from_high_score,
                    "z_score" : z_score,
                    "combined_score" : last_combined_score,
                    "option_yield" : option_yield,
                    "implied_volatility": iv,
                    "delta" : delta,
                    "gamma" : gamma,
                    "theta" : theta,
                    "vega" : vega

                } 
                to_append_option_df = pd.DataFrame(option_dict, index=[0])
                all_options_df = pd.concat([all_options_df,to_append_option_df], axis=0)

    except:
        pass

if len(all_options_df) == 0:
    print(f"[bold red]No suitable stocks found to sell options on. The programme will now end.[/bold red]")

else:
    df_dict = {}
    for exp_date in all_options_df['expiration_date'].unique().tolist():
        temp_df = all_options_df[all_options_df['expiration_date'] == exp_date].copy()
        temp_df.reset_index(inplace=True, drop=True)
        temp_df.sort_values(inplace=True, by="option_yield", ascending=False)
        temp_df['option_yield'] = temp_df['option_yield'].round(1)
        df_dict[exp_date] = temp_df 

    excel_file = f'./excel_output/Puts_Screener_{DATE}.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:

        for key,val in df_dict.items():
            val.to_excel(writer, sheet_name=f'{key}', index=False)

    print(f"[bold green]Excel file saved successfully at: {excel_file}[/bold green]")

        