import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
import math
from tqdm import tqdm
from scipy.stats import norm
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import ta 
from sklearn.linear_model import LinearRegression
from datetime import datetime

API_KEY = "peepeepoopoo"

def get_last_traded_date():
    nyse = mcal.get_calendar('NYSE')
    today = datetime.today()
    valid_days = nyse.valid_days(start_date=today + pd.Timedelta(days=-10), end_date=today)
    last_traded_date = valid_days[-2]
    last_traded_date_str = last_traded_date.strftime('%Y-%m-%d')
    return last_traded_date_str

def get_ohlcv_df(ticker, start_date, end_date, timeframe='1/day', adjusted= 'false'):
    # print(requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{timeframe}/{start_date}/{end_date}?adjusted={adjusted}&sort=asc&limit=500000&apiKey={API_KEY}").json())
    polygon_data = pd.json_normalize(
        requests.get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/{timeframe}/{start_date}/{end_date}?adjusted={adjusted}&sort=asc&limit=500000&apiKey={API_KEY}").json()["results"])
    polygon_data['t'] = pd.to_datetime(polygon_data['t'], unit = 'ms')
    polygon_data.rename(columns={'o': 'Open', 'h':'High', 'l': 'Low', 'c':'Close', 't': 'Date'}, inplace=True)
    polygon_data = polygon_data[['Date', 'Open', 'High', 'Low', 'Close']]
    return polygon_data

def get_puts_contract_by_exp(ticker, query_date_str, sell_put_strike, desired_exp_date):
    # month_to_dte_dict = {6: '2025-09-19', 9: '2025-12-19', 12: '2026-03-20'}

    lower_bound_strike = 0.3*sell_put_strike
    upper_bound_strike = 1.8*sell_put_strike
    url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&strike_price.gte={lower_bound_strike}&strike_price.lte={upper_bound_strike}&contract_type=put&expiration_date={desired_exp_date}&as_of={query_date_str}&expired=false&order=asc&limit=1000&sort=expiration_date&apiKey={API_KEY}"
    contracts_data = pd.json_normalize(requests.get(url).json()["results"])
    contracts_data['strike_diff'] = abs(sell_put_strike - contracts_data['strike_price'])
    strike_diff_arr = np.unique(contracts_data['strike_diff'])
    smallest_strike = strike_diff_arr[0]
    # sec_smallest_strike = strike_diff_arr[1]
    return contracts_data[contracts_data['strike_diff'] == smallest_strike]

def get_options_contract_by_strike(ticker, query_date_str, sell_put_strike, expiry_upper_bound, desired_dte = 180):
    lower_bound_strike = 0.3*sell_put_strike
    upper_bound_strike = 1.8*sell_put_strike

    url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&expiration_date.lte={expiry_upper_bound}&strike_price.gte={lower_bound_strike}&strike_price.lte={upper_bound_strike}&contract_type=put&as_of={query_date_str}&expired=false&order=asc&limit=1000&sort=expiration_date&apiKey={API_KEY}"
    contracts_data = pd.json_normalize(requests.get(url).json()["results"])
    contracts_data['strike_diff'] = abs(sell_put_strike - contracts_data['strike_price'])
    strike_diff_arr = np.unique(contracts_data['strike_diff'])
    smallest_strike = strike_diff_arr[0]
    sec_smallest_strike = strike_diff_arr[1]

    contracts_data['dte'] = (pd.to_datetime(contracts_data['expiration_date'], format='%Y-%m-%d') - pd.to_datetime(query_date_str, format='%Y-%m-%d')).dt.days
    contracts_data['dte_minus_desired_exp'] = abs(desired_dte - contracts_data['dte'])
    # print(contracts_data)
    smallest_filtered_contract = contracts_data[(contracts_data['strike_diff'] == smallest_strike)]
    sec_smallest_filtered_contract = contracts_data[(contracts_data['strike_diff'] == sec_smallest_strike)]

    smallest_diff_dte = smallest_filtered_contract['dte_minus_desired_exp'].min()
    sec_smallest_diff_dte = sec_smallest_filtered_contract['dte_minus_desired_exp'].min()
    smallest_contracts = smallest_filtered_contract[smallest_filtered_contract['dte_minus_desired_exp'] == smallest_diff_dte]
    sec_smallest_contracts = sec_smallest_filtered_contract[sec_smallest_filtered_contract['dte_minus_desired_exp'] == sec_smallest_diff_dte]
    
    return smallest_contracts,sec_smallest_contracts
    # second_exp_date = contracts_data['expiration_date'].unique()[1]
    # filtered_contracts = contracts_data[contracts_data['expiration_date'] == second_exp_date]

    # filtered_contracts['strike_diff'] = filtered_contracts['strike_price'].diff()

    # # Step 3: Find the smallest positive difference
    # smallest_distance = filtered_contracts['strike_diff'].min()
    # return option_strike_width



def get_vol_score_w_benchmark(df, sector_benchmark = 'QQQ', lookback = 100):
    vol_df = df.copy()
    vol_df.replace(0, np.nan, inplace=True)
    vol_df = vol_df.ffill()
    vol_df['log_returns'] = np.log(vol_df['Close']/vol_df['Close'].shift(1))
    vol_df['3M_realised_vol'] = vol_df['log_returns'].rolling(window=63).std() * np.sqrt(252)
    vol_df['3M_iv_ratio'] = vol_df['combined_iv']/vol_df['3M_realised_vol']

    spy_vol_df = pd.read_csv(f'SPY/SPY_3M_ATM_IV.csv')
    spy_vol_df['Date'] = pd.to_datetime(spy_vol_df['Date'],  format='%Y-%m-%d')
    spy_vol_df.replace(0, np.nan, inplace=True)
    spy_vol_df = spy_vol_df.ffill()
    spy_vol_df['log_returns'] = np.log(spy_vol_df['Close']/spy_vol_df['Close'].shift(1))
    spy_vol_df['3M_realised_vol'] = spy_vol_df['log_returns'].rolling(window=63).std() * np.sqrt(252)
    spy_vol_df['3M_iv_ratio'] = spy_vol_df['combined_iv']/spy_vol_df['3M_realised_vol']

    benchmark_vol_df = pd.read_csv(f'{sector_benchmark}/{sector_benchmark}_3M_ATM_IV.csv')
    benchmark_vol_df['Date'] = pd.to_datetime(benchmark_vol_df['Date'],  format='%Y-%m-%d')
    benchmark_vol_df.replace(0, np.nan, inplace=True)
    benchmark_vol_df = benchmark_vol_df.ffill()
    benchmark_vol_df['log_returns'] = np.log(benchmark_vol_df['Close']/benchmark_vol_df['Close'].shift(1))
    benchmark_vol_df['3M_realised_vol'] = benchmark_vol_df['log_returns'].rolling(window=63).std() * np.sqrt(252)
    benchmark_vol_df['3M_iv_ratio'] = benchmark_vol_df['combined_iv']/benchmark_vol_df['3M_realised_vol']

    merged_vol_df = vol_df.merge(spy_vol_df[['Date', '3M_iv_ratio']], on = 'Date', suffixes = ('', '_SPY')) 
    merged_vol_df = merged_vol_df.merge(benchmark_vol_df[['Date', '3M_iv_ratio']],on = 'Date', suffixes = ('', '_sector')) 

    merged_vol_df['stock_iv_ratio_against_spy'] = merged_vol_df['3M_iv_ratio'] / merged_vol_df['3M_iv_ratio_SPY']
    merged_vol_df['stock_iv_ratio_against_sector'] = merged_vol_df['3M_iv_ratio'] / merged_vol_df['3M_iv_ratio_sector']
    
    merged_vol_df['stock_iv_ratio_against_spy_score'] = np.nan
    merged_vol_df['stock_iv_ratio_against_sector_score'] = np.nan
    merged_vol_df['vol_score'] = np.nan

    # Start loop at std_dev_lookback to ensure lookback_period has enough data
    for idx in range(lookback, len(merged_vol_df)):
        lookback_period = merged_vol_df.iloc[idx - lookback:idx]

        curr_ivratio_against_spy = merged_vol_df['3M_iv_ratio_SPY'].iloc[idx]
        curr_ivratio_against_sector = merged_vol_df['3M_iv_ratio_sector'].iloc[idx]

        perc_ivratio_against_spy = len(lookback_period[lookback_period['3M_iv_ratio_SPY'] <  curr_ivratio_against_spy])/len(lookback_period) * 100
        perc_ivratio_against_sector = len(lookback_period[lookback_period['3M_iv_ratio_sector'] <  curr_ivratio_against_sector])/len(lookback_period) * 100
        avg_perc = (perc_ivratio_against_spy + perc_ivratio_against_sector)/2
        
        merged_vol_df.at[merged_vol_df.index[idx], 'stock_iv_ratio_against_spy_score'] = perc_ivratio_against_spy
        merged_vol_df.at[merged_vol_df.index[idx], 'stock_iv_ratio_against_sector_score'] = perc_ivratio_against_sector
        merged_vol_df.at[merged_vol_df.index[idx], 'vol_score'] = avg_perc

    return merged_vol_df

def get_volatilityscore(df, lookback = 504):
    vol_df = df.copy()
    vol_df.replace(0, np.nan, inplace=True)
    vol_df = vol_df.ffill()
    vol_df['3M_log_returns'] = np.log(vol_df['Close']/vol_df['Close'].shift(1))
    vol_df['3M_realised_vol'] = vol_df['3M_log_returns'].rolling(window=63).std() * np.sqrt(252)
    vol_df['3M_iv_ratio'] = vol_df['combined_iv']/vol_df['3M_realised_vol']

    vol_df['3M_iv_ratio_score'] = np.nan
    vol_df['3M_iv_score'] = np.nan
    vol_df['vol_score'] = np.nan
    # Start loop at std_dev_lookback to ensure lookback_period has enough data
    for idx in range(lookback, len(vol_df)):
        lookback_period = vol_df.iloc[idx - lookback:idx]

        curr_ivratio = vol_df['3M_iv_ratio'].iloc[idx]
        curr_iv = vol_df['combined_iv'].iloc[idx]

        perc_ivratio = len(lookback_period[lookback_period['3M_iv_ratio'] <  curr_ivratio])/len(lookback_period) * 100
        perc_iv = len(lookback_period[lookback_period['combined_iv'] <  curr_iv])/len(lookback_period) * 100
        avg_perc = (perc_ivratio + perc_iv)/2
        
        vol_df.at[vol_df.index[idx], '3M_iv_ratio_score'] = perc_ivratio
        vol_df.at[vol_df.index[idx], '3M_iv_score'] = perc_iv
        vol_df.at[vol_df.index[idx], 'vol_score'] = avg_perc

    return vol_df

def black_scholes(option_type, S, K, t, r, q, sigma):
    """
    Calculate the Black-Scholes option price.

    :param option_type: 'call' for call option, 'put' for put option.
    :param S: Current stock price.
    :param K: Strike price.
    :param t: Time to expiration (in years).
    :param r: Risk-free interest rate (annualized).
    :param q: Dividend yield (annualized).
    :param sigma: Stock price volatility (annualized).

    :return: Option price.
    """
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)

    if option_type == 'call':
        return S * math.exp(-q * t) * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    elif option_type == 'put':
        return K * math.exp(-r * t) * norm.cdf(-d2) - S * math.exp(-q * t) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'.")
        
def call_implied_vol(S, K, t, r, option_price):
    q = 0.01
    option_type = "call"

    def f_call(sigma):
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    try:
        call_newton_vol = optimize.newton(f_call, x0=0.30, tol=0.0001, maxiter=100)
        if call_newton_vol <= 0 or np.isnan(call_newton_vol):
            raise ValueError("Invalid implied volatility")
    except (RuntimeError, ValueError, OverflowError):
        call_newton_vol = np.nan  # Set IV as NaN when it fails

    return call_newton_vol

def put_implied_vol(S, K, t, r, option_price):
    q = 0.01
    option_type = "put"

    def f_put(sigma):
        return black_scholes(option_type, S, K, t, r, q, sigma) - option_price

    try:
        put_newton_vol = optimize.newton(f_put, x0=0.30, tol=0.0001, maxiter=100)
        if put_newton_vol <= 0 or np.isnan(put_newton_vol):
            raise ValueError("Invalid implied volatility")
    except (RuntimeError, ValueError, OverflowError):
        put_newton_vol = np.nan  # Set IV as NaN when it fails

    return put_newton_vol

def compute_atm_call_iv(ticker, date, close_price, options_DTE = 7):
    quote_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 15, minutes = 55)).value
    close_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 16, minutes = 0)).value
    
    expiry_date = pd.to_datetime(date) + pd.Timedelta(days=options_DTE)
    expiry_date_str = expiry_date.strftime('%Y-%m-%d')
    calls = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&expiration_date.gte={expiry_date_str}&contract_type=call&as_of={date}&limit=1000&apiKey={API_KEY}").json()["results"])
    calls["days_to_exp"] = (pd.to_datetime(calls["expiration_date"]) - pd.to_datetime(date)).dt.days
    # print(calls)
    calls = calls[calls["days_to_exp"] >= options_DTE].copy()
    nearest_exp_date = calls["expiration_date"].iloc[0]
    
    calls = calls[calls["expiration_date"] == nearest_exp_date].copy()
    calls["distance_from_price"] = abs(round(((calls["strike_price"] - close_price) / close_price)*100, 2))
    
    atm_call = calls.nsmallest(1, "distance_from_price")
    atm_call_dte = atm_call['days_to_exp'].iloc[0]
    print(atm_call)
    
    call_trades = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/trades/{atm_call['ticker'].iloc[0]}?timestamp={date}&order=desc&limit=1000&sort=timestamp&apiKey={API_KEY}").json()["results"]).set_index("sip_timestamp")
    call_trades.index = pd.to_datetime(call_trades.index, unit = "ns", utc = True).tz_convert("America/New_York")
    # print('===========================================================')
    # print(call_trades)
    
    # call_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_call['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=desc&limit=50000&sort=timestamp&apiKey={API_KEY}").json()["results"]).set_index("sip_timestamp")
    # call_quotes.index = pd.to_datetime(call_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
    # call_quotes["mid_price"] = round((call_quotes["bid_price"] + call_quotes["ask_price"]) / 2, 2)
    # 604800 seconds -> 7 days
    time_to_expiration = (atm_call_dte * 86400 / 86400) / 252
    print('call price:' , call_trades["price"].iloc[0])
    atm_call_vol = call_implied_vol(S=close_price, K=atm_call["strike_price"].iloc[0], t=time_to_expiration, r=.045, option_price=call_trades["price"].iloc[0])
    return atm_call_vol

def compute_atm_put_iv(ticker, date, close_price, options_DTE = 7):
    quote_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 15, minutes = 55)).value
    close_timestamp = (pd.to_datetime(date).tz_localize("America/New_York") + timedelta(hours = 16, minutes = 0)).value

    expiry_date = pd.to_datetime(date) + pd.Timedelta(days=options_DTE)
    expiry_date_str = expiry_date.strftime('%Y-%m-%d')

    puts = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&expiration_date.gte={expiry_date_str}&contract_type=put&as_of={date}&limit=1000&apiKey={API_KEY}").json()["results"])
    # print(puts)
    puts["days_to_exp"] = (pd.to_datetime(puts["expiration_date"]) - pd.to_datetime(date)).dt.days
    puts = puts[puts["days_to_exp"] >= options_DTE].copy()
    nearest_exp_date = puts["expiration_date"].iloc[0]

    puts = puts[puts["expiration_date"] == nearest_exp_date].copy()
    puts["distance_from_price"] = abs(round(((close_price - puts["strike_price"]) / puts["strike_price"])*100, 2))
    
    atm_put = puts.nsmallest(1, "distance_from_price")
    atm_put_dte = atm_put['days_to_exp'].iloc[0]
    print(atm_put)
    time_to_expiration = (atm_put_dte * 86400 / 86400) / 252

    put_trades = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/trades/{atm_put['ticker'].iloc[0]}?timestamp={date}&order=desc&limit=1000&sort=timestamp&apiKey={API_KEY}").json()["results"]).set_index("sip_timestamp")
    put_trades.index = pd.to_datetime(put_trades.index, unit = "ns", utc = True).tz_convert("America/New_York")
    # print('===========================================================')
    # print(put_trades)

    # put_quotes = pd.json_normalize(requests.get(f"https://api.polygon.io/v3/quotes/{atm_put['ticker'].iloc[0]}?timestamp.gte={quote_timestamp}&timestamp.lt={close_timestamp}&order=desc&limit=100&sort=timestamp&apiKey={API_KEY}").json()["results"]).set_index("sip_timestamp")
    # put_quotes.index = pd.to_datetime(put_quotes.index, unit = "ns", utc = True).tz_convert("America/New_York")
    # put_quotes["mid_price"] = round((put_quotes["bid_price"] + put_quotes["ask_price"]) / 2, 2)
    print('put price:' , put_trades["price"].iloc[0])
    atm_put_vol = put_implied_vol(S=close_price, K=atm_put["strike_price"].iloc[0], t=time_to_expiration, r=.045, option_price=put_trades["price"].iloc[0])
    return atm_put_vol 

def get_expected_move(ticker, df, options_DTE):
    working_df = df.copy()
    working_df['call_iv'] = np.nan 
    working_df['put_iv'] = np.nan
    working_df['combined_iv'] = np.nan  
    working_df['expected_1d_move'] = np.nan
    working_df['expected_1w_move'] = np.nan 

    for i in tqdm(range(len(working_df)), desc="poopoo"):
        date_str = working_df['Date'].iloc[i].strftime('%Y-%m-%d')
        close_price = working_df['Close'].iloc[i]

        try:
            print('close price' , close_price)
            call_iv = compute_atm_call_iv(ticker, date_str, close_price, options_DTE)
            put_iv = compute_atm_put_iv(ticker, date_str, close_price, options_DTE)
        except Exception as error:
            print('error found', error)
            call_iv = 0
            put_iv = 0

        combined_iv = (call_iv + put_iv)/2
        expected_1d_move = close_price * combined_iv * np.sqrt(1/252)
        expected_1w_move = close_price * combined_iv * np.sqrt(5/252)

        working_df.at[working_df.index[i], 'call_iv'] = call_iv
        working_df.at[working_df.index[i], 'put_iv'] = put_iv
        working_df.at[working_df.index[i], 'combined_iv'] = combined_iv
        working_df.at[working_df.index[i], 'expected_1d_move'] = expected_1d_move
        working_df.at[working_df.index[i], 'expected_1w_move'] = expected_1w_move
        print(f'{call_iv=}', f'{put_iv=}', f'{expected_1d_move=}', f'{expected_1w_move=}')
    return working_df


def get_trades_df_highvol(df, trade_freq = 7):
    backtest_df = df.copy() 

    in_Trade = False 
    last_trade_date = pd.to_datetime("2020-01-01", format="%Y-%m-%d")

    trades_df = pd.DataFrame()
    for i in range(len(backtest_df)):
        date = backtest_df['Date'].iloc[i]
        close_price = backtest_df['Close'].iloc[i]
        trade_signal = backtest_df['trade_signal'].iloc[i]
        expected_1d_move = backtest_df['expected_1d_move'].iloc[i]
        expected_1w_move = backtest_df['expected_1w_move'].iloc[i]

        if trade_signal != 0 and in_Trade == False and (date - last_trade_date).days >= trade_freq :
            in_Trade == True 
            last_trade_date = date
            trade_dict = {"Date": date, "Close": close_price, "trade_signal": trade_signal, "expected_1d_move": expected_1d_move, "expected_1w_move": expected_1w_move}
            trade_df = pd.DataFrame(trade_dict, index=[0])
            trades_df = pd.concat([trades_df, trade_df], axis = 0)
    trades_df.reset_index(inplace=True, drop=True)
    return trades_df

def round_to_nearest(value, strike_width):
    """
    Rounds the value to the nearest strike width (5 or 10).
    """
    if strike_width == 10:
        return round(value / 10) * 10
    elif strike_width == 5:
        return round(value / 5) * 5
    return value 

def get_options_df(ticker, date, primary_strike, fallback_strike, option_type = 'call'):
    ''' 
    option_type either 'call' or 'put'
    '''
    if option_type == None:
        return None
    try:
        url_primary = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type={option_type}&as_of={date}&strike_price={primary_strike}&limit=1000&apiKey={API_KEY}"
        response_primary = requests.get(url_primary)
        # Check if primary request is successful
        if response_primary.status_code == 200:
            data_primary = response_primary.json()
            if "results" in data_primary and data_primary["results"]:
                print(f"✅ Found results for {ticker} on {date} at strike {primary_strike}")
                return pd.json_normalize(data_primary["results"])  
        
        print(f"⚠️ No results for {ticker} at {primary_strike}, retrying with rounded strike {fallback_strike}...")
        url_fallback = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&contract_type={option_type}&as_of={date}&strike_price={fallback_strike}&limit=1000&apiKey={API_KEY}"
        response_fallback = requests.get(url_fallback)

        if response_fallback.status_code == 200:
            data_fallback = response_fallback.json()
            if "results" in data_fallback and data_fallback["results"]:
                print(f"✅ Found results for {ticker} on {date} at fallback strike {fallback_strike}")
                return pd.json_normalize(data_fallback["results"]) 

        print(f"❌ No results found for {ticker} on {date} at both strikes ({primary_strike} and {fallback_strike})")
        return None

    except Exception as e:
        print(f"Error fetching options data: {e}")
        return None

def get_option(options_df, curr_date, option_dte = 5):
    working_df = options_df.copy()
    working_df["days_to_exp"] = (pd.to_datetime(working_df["expiration_date"]) - pd.to_datetime(curr_date)).dt.days
    working_df = working_df[working_df["days_to_exp"] >= option_dte].copy()
    nearest_exp_date = working_df["expiration_date"].iloc[0]
    option = working_df[working_df["expiration_date"] == nearest_exp_date].copy()
    return option

def get_strike_width(price):
    if price < 25:
        return 0.5
    elif price < 100:
        return 1
    elif price < 1000:
        return 5
    else:
        return 10
    
def get_options_names_df(ticker, df, option_dte, sell_atm = False):
    working_df = df.copy() 
    
    if sell_atm == True:
        working_df['sell_put_strike'] = np.floor(working_df['Close'] )
        working_df['strike_width'] = working_df['sell_put_strike'].apply(lambda x : get_strike_width(x))
        working_df['buy_put_strike'] = working_df['sell_put_strike'] - working_df['strike_width']
        working_df['sell_call_strike'] =  np.ceil(working_df['Close'] )
        working_df['buy_call_strike'] = working_df['sell_call_strike'] + working_df['strike_width']
    else:
        working_df['sell_put_strike'] = np.floor(working_df['Close'] - working_df['expected_1w_move'])
        working_df['strike_width'] = working_df['sell_put_strike'].apply(lambda x : get_strike_width(x))
        working_df['buy_put_strike'] = working_df['sell_put_strike'] - working_df['strike_width']
        working_df['sell_call_strike'] =  np.ceil(working_df['Close'] + working_df['expected_1w_move'])
        working_df['buy_call_strike'] = working_df['sell_call_strike'] + working_df['strike_width']
    
    #round to nearest 5
    
    working_df['sell_put_strike_rounded'] = working_df.apply(lambda row: round_to_nearest(row['sell_put_strike'], row['strike_width']), axis=1)
    working_df['buy_put_strike_rounded'] = working_df.apply(lambda row: round_to_nearest(row['buy_put_strike'], row['strike_width']), axis=1)
    working_df['sell_call_strike_rounded'] =  working_df.apply(lambda row: round_to_nearest(row['sell_call_strike'], row['strike_width']), axis=1)
    working_df['buy_call_strike_rounded'] = working_df.apply(lambda row: round_to_nearest(row['buy_call_strike'], row['strike_width']), axis=1)

    sell_option_ticker_ls = []
    buy_option_ticker_ls = []

    dte_ls = []

    for i in range(len(working_df)):
        date = working_df['Date'].iloc[i]
        date_str = date.strftime('%Y-%m-%d')
        trade_signal = working_df['trade_signal'].iloc[i]
        try:
            if trade_signal == 1: #1 = sell put spreads, # -1 = sell call spreads 
                sell_put_strike = working_df['sell_put_strike'].iloc[i]
                sell_put_strike_rounded = working_df['sell_put_strike_rounded'].iloc[i]
                buy_put_strike = working_df['buy_put_strike'].iloc[i]
                buy_put_strike_rounded = working_df['buy_put_strike_rounded'].iloc[i]

                sell_put_df = get_options_df(ticker, date, sell_put_strike, sell_put_strike_rounded, option_type = 'put')
                buy_put_df = get_options_df(ticker, date, buy_put_strike, buy_put_strike_rounded, option_type = 'put')

                sell_put_option = get_option(sell_put_df, date_str, option_dte)
                buy_put_option = get_option(buy_put_df, date_str, option_dte)

                sell_put_option_ticker = sell_put_option['ticker'].iloc[0]
                buy_put_option_ticker = buy_put_option['ticker'].iloc[0]

                sell_option_ticker_ls.append(sell_put_option_ticker)
                buy_option_ticker_ls.append(buy_put_option_ticker)
                
                dte = sell_put_option['days_to_exp'].iloc[0]

            elif trade_signal == -1:
                sell_call_strike = working_df['sell_call_strike'].iloc[i]
                sell_call_strike_rounded = working_df['sell_call_strike_rounded'].iloc[i]
                buy_call_strike = working_df['buy_call_strike'].iloc[i]
                buy_call_strike_rounded = working_df['buy_call_strike_rounded'].iloc[i]
            
                sell_call_df = get_options_df(ticker, date, sell_call_strike, sell_call_strike_rounded, option_type = 'call')
                buy_call_df = get_options_df(ticker, date, buy_call_strike, buy_call_strike_rounded, option_type = 'call')

                sell_call_option = get_option(sell_call_df, date_str, option_dte)
                buy_call_option = get_option(buy_call_df, date_str,option_dte)

                sell_call_option_ticker = sell_call_option['ticker'].iloc[0]
                buy_call_option_ticker = buy_call_option['ticker'].iloc[0]

                sell_option_ticker_ls.append(sell_call_option_ticker)
                buy_option_ticker_ls.append(buy_call_option_ticker)

                dte = sell_call_option['days_to_exp'].iloc[0]
            dte_ls.append(dte)
        except Exception as error:
            print(f'error : {error}') 
            sell_option_ticker_ls.append(np.nan)
            buy_option_ticker_ls.append(np.nan)
            dte_ls.append(np.nan)


    options_tickers_df = pd.DataFrame({'sell_option_ticker' : sell_option_ticker_ls,
                               'buy_option_ticker' : buy_option_ticker_ls, 
                               'days_to_exp' : dte_ls}, index= [x for x in range(len(working_df))])
    
    final_df = pd.concat([working_df, options_tickers_df], axis=1)
    final_df['option_type'] = np.where(final_df['trade_signal'] == 1, 'put', 'call')
    option_names_df = final_df[['Date', 'Close', 'option_type', 'sell_option_ticker', 'buy_option_ticker', 'days_to_exp']]
    return option_names_df

def get_open_close_price_option(option_ticker, date, dte):
    #get next trading day
    nyse = mcal.get_calendar('NYSE')
    next_open_day = nyse.valid_days(start_date=date, end_date=date + pd.Timedelta(days=10))[1]
    
    date_str = date.strftime('%Y-%m-%d')
    next_open_day_str = next_open_day.strftime('%Y-%m-%d')

    query_end_date = date + pd.Timedelta(days=dte)
    query_end_date_str = query_end_date.strftime('%Y-%m-%d')
    print(f'decision date: {date_str}. trade executed at open of {next_open_day_str}, end at {query_end_date_str}.')
    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{next_open_day_str}/{query_end_date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    # print(ohlcvdf)
    initial_price = ohlcvdf['o'].iloc[0]
    end_price = ohlcvdf['c'].iloc[-1]
    return initial_price, end_price

def get_option_close_price(option_ticker, date_str):

    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{date_str}/{date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    print(ohlcvdf)
    close_price = ohlcvdf['c'].iloc[0]

    print(f'Date: {date_str}. Put close price at {close_price}.')
    return close_price

def get_option_quotes(option_ticker, date_str):
    # url = f"https://api.polygon.io/v3/trades/{option_ticker}?timestamp={date_str}&order=asc&limit=50000&sort=timestamp&apiKey={API_KEY}"
    # url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{date_str}/{date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    
    url = f"https://api.polygon.io/v3/quotes/{option_ticker}?timestamp={date_str}&order=desc&limit=50000&sort=timestamp&apiKey={API_KEY}"
    trades_df = pd.json_normalize(requests.get(url).json()['results'])
    trades_df['sip_timestamp'] = pd.to_datetime(trades_df['sip_timestamp'], unit='ns')
        # Step 2: Localize the datetime to UTC
    trades_df['sip_timestamp'] = trades_df['sip_timestamp'].dt.tz_localize('UTC')

    # Step 3: Convert UTC to Eastern Time (NYSE time)
    trades_df['sip_timestamp'] = trades_df['sip_timestamp'].dt.tz_convert('US/Eastern')
    trades_df = trades_df.head(50)
    # ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    # print(ohlcvdf)
    # close_price = ohlcvdf['c'].iloc[0]

    # print(f'Date: {date_str}. Put close price at {close_price}.')
    return trades_df

def get_options_backtest(df_w_optionnames):
    backtest_df = df_w_optionnames.copy() 

    strategy_pnl_ls = []
    strategy_end_cost_ls = []
    strategy_initial_premium_ls = []
    for idx, row in backtest_df.iterrows():
        dte = row['days_to_exp']
        date = row['Date']
        option_type = row['option_type']
        sell_option_ticker = row['sell_option_ticker']
        buy_option_ticker = row['buy_option_ticker']
        try:    
            sell_open_price, buy_close_price = get_open_close_price_option(sell_option_ticker, date, dte)
            buy_open_price, sell_close_price = get_open_close_price_option(buy_option_ticker, date, dte)
        except Exception as error:
            print(f'{date} has {error}.')
            print('setting everything as 0.')
            sell_open_price, buy_close_price = 0, 0
            buy_open_price, sell_close_price = 0, 0

        strategy_initial_premium =  sell_open_price - buy_open_price
        strategy_end_cost  = buy_close_price - sell_close_price
        strategy_pnl = strategy_initial_premium - strategy_end_cost
        
        print(f'pnl from {option_type} credit spread: {strategy_pnl}')

        strategy_initial_premium_ls.append(strategy_initial_premium)
        strategy_end_cost_ls.append(strategy_end_cost)
        strategy_pnl_ls.append(strategy_pnl)


    option_trades = pd.DataFrame({
                                  'initial_premium' : strategy_initial_premium_ls,
                                  'closing_cost' : strategy_end_cost_ls,
                                  'strategy_pnl': strategy_pnl_ls
                                  }, index= [x for x in range(len(backtest_df))])
    
    final_df = pd.concat([backtest_df, option_trades], axis=1)
    return final_df


# 1. 1M Beta Adjusted Relative Performance 
def compute_1M_beta(ohlcv_df_stock, ohlcv_df_benchmark, beta_calc_window = 252):
    to_merge_ohlcv = ohlcv_df_stock[['Date', 'Close']].copy()
    to_merge_ohlcv['log_ret'] = np.log(to_merge_ohlcv['Close'] / to_merge_ohlcv['Close'].shift(1))
    
    to_merge_benchmark = ohlcv_df_benchmark[['Date', 'Close']].copy()
    to_merge_benchmark['log_ret'] = np.log(to_merge_benchmark['Close'] / to_merge_benchmark['Close'].shift(1))
    merged_df = to_merge_ohlcv.merge(to_merge_benchmark, on='Date', how='left', suffixes = ('_stock', '_benchmark'))
    merged_df.dropna(inplace=True)
    #compute beta 
    beta_ls = []  
    date_ls = merged_df['Date'].iloc[beta_calc_window:].values  

    for i in range(beta_calc_window, len(merged_df)):
        train_df = merged_df.iloc[i - beta_calc_window : i]
        X = train_df[['log_ret_benchmark']].values
        y = train_df['log_ret_stock'].values

        model = LinearRegression()
        model.fit(X,y)
        beta = model.coef_[0]
        # print(f"Computed Beta: {beta:.4f}")
        beta_ls.append(beta)
    
    beta_df = pd.DataFrame({'Date': date_ls, 'Beta': beta_ls})
    return beta_df

def compute_1M_beta_adj_rel_perf(ohlcv_df_stock, ohlcv_df_benchmark, beta_df):
    working_df = ohlcv_df_stock.copy()
    working_df['monthly_log_returns'] = np.log(working_df['Close']/working_df['Close'].shift(21) )  # ~21 trading days a month
    to_merge_df = working_df[['Date', 'monthly_log_returns']].copy() 

    workingdf2 = ohlcv_df_benchmark.copy()
    workingdf2['monthly_log_returns'] = np.log(workingdf2['Close']/workingdf2['Close'].shift(21) )  # ~21 trading days a month
    to_merge_df2 = workingdf2[['Date', 'monthly_log_returns']].copy() 

    merged_df = to_merge_df.merge(to_merge_df2, on='Date', how = 'left', suffixes = ('_stock', '_benchmark'))
    merged_df2 = merged_df.merge(beta_df, on='Date', how = 'left')

    merged_df2['1m_beta_adj_perf'] = merged_df2['monthly_log_returns_stock'] - (merged_df2['Beta'] * merged_df2['monthly_log_returns_benchmark'])
    return merged_df2[['Date', '1m_beta_adj_perf']]

# 4. BEst EPS 4w Change 
# 5. JPM/BEst EPS price upside


##### Vol Indicators ###### 
# 1. 3M implied vs realised vol to mkt 
# 2. 3M implied vs realised vol to sector

def get_trend_regime(ohlcv_df, benchmark_ticker, sma_period = 63, rsi_period = 14):
    working_df = ohlcv_df.copy()
    # 1. 1M Beta Adjusted Relative Performance 
    start_date = working_df['Date'].iloc[0].strftime('%Y-%m-%d')
    end_date = working_df['Date'].iloc[-1].strftime('%Y-%m-%d')
    ohlcv_df_benchmark = get_ohlcv_df(benchmark_ticker, start_date, end_date, timeframe='1/day')

    beta_df = compute_1M_beta(ohlcv_df, ohlcv_df_benchmark, beta_calc_window = 252)
    beta_adj_perf_df = compute_1M_beta_adj_rel_perf(ohlcv_df, ohlcv_df_benchmark, beta_df)
    
    # 2. 63D Price Z-score 
    working_df[f'SMA_{sma_period}'] = working_df[f'Close'].rolling(window=sma_period).mean()
    working_df[f'std_{sma_period}'] = working_df[f'Close'].rolling(window=sma_period).std()
    working_df[f'{sma_period}D_overbought'] = working_df[f'SMA_{sma_period}'] + 2 * working_df[f'std_{sma_period}'] 
    working_df[f'{sma_period}D_oversold'] = working_df[f'SMA_{sma_period}'] - 2 * working_df[f'std_{sma_period}'] 
    # 3. 14D RSI 
    working_df[f'RSI_{rsi_period}'] = ta.momentum.RSIIndicator(working_df["Close"], window=rsi_period).rsi()

    merged_w_beta = beta_adj_perf_df.merge(working_df, on='Date', how = 'left')
    merged_w_beta.dropna(inplace=True)

    
    merged_w_beta['1m_beta_adj_score'] = np.where(merged_w_beta['1m_beta_adj_perf'] > 0 , 1, 
                                                  np.where(merged_w_beta['1m_beta_adj_perf'] < 0, -1, 0)) 
    
    merged_w_beta['RSI_score'] = np.where(merged_w_beta[f'RSI_{rsi_period}'] >= 70 , -1, 
                                          np.where(merged_w_beta[f'RSI_{rsi_period}'] <= 30 , 1, 0))
    merged_w_beta['z_score'] = np.where(merged_w_beta['Close'] >= merged_w_beta[f'{sma_period}D_overbought'] , -1, 
                                          np.where(merged_w_beta['Close'] <= merged_w_beta[f'{sma_period}D_oversold'] , 1, 0))

    merged_w_beta['combined_score'] = merged_w_beta['1m_beta_adj_score'] + merged_w_beta['RSI_score'] + merged_w_beta['z_score']
    return merged_w_beta

def get_trend_regime_wo_beta(ohlcv_df, sma_period = 63, rsi_period = 14):
    working_df = ohlcv_df.copy()
    working_df['1y_rolling_max'] = working_df['High'].rolling(252).max()
    working_df['max_to_close_returns'] = (working_df['Close']  - working_df['1y_rolling_max'] )/ working_df['1y_rolling_max']
    # 2. 63D Price Z-score 
    working_df[f'SMA_{sma_period}'] = working_df[f'Close'].rolling(window=sma_period).mean()
    working_df[f'std_{sma_period}'] = working_df[f'Close'].rolling(window=sma_period).std()
    working_df[f'{sma_period}D_overbought'] = working_df[f'SMA_{sma_period}'] + 2 * working_df[f'std_{sma_period}'] 
    working_df[f'{sma_period}D_oversold'] = working_df[f'SMA_{sma_period}'] - 2 * working_df[f'std_{sma_period}'] 
    # 3. 14D RSI 
    working_df[f'RSI_{rsi_period}'] = ta.momentum.RSIIndicator(working_df["Close"], window=rsi_period).rsi()

    working_df['RSI_score'] = np.where(working_df[f'RSI_{rsi_period}'] >= 70 , -1, 
                                          np.where(working_df[f'RSI_{rsi_period}'] <= 30 , 1, 0))
    working_df['z_score'] = np.where(working_df['Close'] >= working_df[f'{sma_period}D_overbought'] , -1, 
                                          np.where(working_df['Close'] <= working_df[f'{sma_period}D_oversold'] , 1, 0))
    working_df['perc_from_high_score'] = np.where(working_df['max_to_close_returns'] >= 0.1, -1, 
                                          np.where(working_df['max_to_close_returns'] <= -0.1 , 1, 0))
    
    working_df['combined_score'] = working_df['RSI_score'] + working_df['z_score'] + working_df['perc_from_high_score'] 
    working_df.dropna(inplace=True)
    return working_df


def get_vol_regime(ohlcv_df, features_df, benchmark_ticker, std_dev_lookback=504):
    working_df = ohlcv_df.copy() 
    working_df['log_returns'] = np.log(working_df['Close']/working_df['Close'].shift(1))
    working_df['realised_vol'] = working_df['log_returns'].rolling(window=504).std() * np.sqrt(252)

    working_df = working_df[['realised_vol', 'Date']].merge(features_df.copy(), on='Date', how='right')
    working_df = working_df[working_df['combined_iv'] != 0]
    working_df['iv_rv_ratio'] = working_df['combined_iv']/working_df['realised_vol']

    start_date = working_df['Date'].iloc[0]
    end_date = working_df['Date'].iloc[-1]
    benchmark_df = get_ohlcv_df(benchmark_ticker, start_date, end_date, timeframe='1/day')
    benchmark_df['log_returns'] = np.log(benchmark_df['Close']/benchmark_df['Close'].shift(1))
    benchmark_df['realised_vol'] = benchmark_df['log_returns'].rolling(window=504).std() * np.sqrt(252)


    working_df['vol_regime'] = np.nan
    # Start loop at std_dev_lookback to ensure lookback_period has enough data
    for idx in range(std_dev_lookback, len(working_df)):
        lookback_period = working_df.iloc[idx - std_dev_lookback:idx]

        # Ensure lookback_period is not empty
        if lookback_period.empty:
            continue  # Skip iteration if no data available

        perc_75 = np.percentile(lookback_period['iv_rv_ratio'], 75)
        perc_25 = np.percentile(lookback_period['iv_rv_ratio'], 25)

        curr_iv_rv = working_df['iv_rv_ratio'].iloc[idx]

        if curr_iv_rv >= perc_75:
            working_df.at[working_df.index[idx], 'vol_regime'] = 1  # High vol regime
        elif curr_iv_rv <= perc_25: 
            working_df.at[working_df.index[idx], 'vol_regime'] = -1  # Low vol regime
        else:
            working_df.at[working_df.index[idx], 'vol_regime'] = 0
    return working_df



def compute_vix_expectedmove(df, normalisation_lookback = 60):
    working_df = df.copy()
    start_date = working_df['Date'].iloc[0]
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date = working_df['Date'].iloc[-1]
    end_date_str = end_date.strftime('%Y-%m-%d')

    vix_data = pd.json_normalize(requests.get(f"https://api.polygon.io/v2/aggs/ticker/I:VIX/range/1/day/{start_date_str}/{end_date_str}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}").json()["results"])
    vix_data['Date'] = pd.to_datetime(vix_data['t'], unit = 'ms')
    vix_data['date_str'] = vix_data['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # vix_data['Date'] = pd.to_datetime(vix_data['t'], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York")

    vix_data["vix_1d_expected_move"] = vix_data["c"] / np.sqrt(252)
    vix_data["vix_1w_expected_move"] = vix_data["c"] * np.sqrt(5/252)
    
    merged_df = pd.merge(working_df, vix_data[["date_str","vix_1d_expected_move", "vix_1w_expected_move"]], on = "date_str")
    merged_df['idiosyncratic_vol_flag'] = 0
    print(merged_df)

    for idx in range(normalisation_lookback, len(merged_df)):
        lookback_period = merged_df.iloc[idx - normalisation_lookback : idx].copy()
        #vix normalisation
        vix_1w_min = lookback_period['vix_1w_expected_move'].min()
        vix_1w_max = lookback_period['vix_1w_expected_move'].max()
        lookback_period['vix_1w_normalised'] = (lookback_period['vix_1w_expected_move'] - vix_1w_min) / (vix_1w_max - vix_1w_min)

        #stock normalisation
        stock_1w_min = lookback_period['expected_1w_move'].min()
        stock_1w_max = lookback_period['expected_1w_move'].max()
        lookback_period['stock_1w_normalised'] = (lookback_period['expected_1w_move'] - stock_1w_min) / (stock_1w_max - stock_1w_min)

        lookback_period['1w_normlised_spread'] = abs(lookback_period['vix_1w_normalised'] - lookback_period['stock_1w_normalised']) 

        curr_normalised_spread = lookback_period['1w_normlised_spread'].iloc[-1]

        if curr_normalised_spread >= 0.5:
            merged_df.at[merged_df.index[idx], 'idiosyncratic_vol_flag'] = 1
    return merged_df

def get_trade_signal(df):
    working_df = df.copy() 

    working_df['trade_signal'] = np.where((working_df['vol_score'] >= 75) & (working_df['combined_score'] > 0) , 1, 
                                          np.where((working_df['vol_score'] >= 75) & (working_df['combined_score'] < 0) , -1, 0))
    
    working_df['trade_signal_lowvol'] = np.where((working_df['vol_score'] <= 25) & (working_df['combined_score'] > 0) , 1, 
                                          np.where((working_df['vol_score'] <= 25) & (working_df['combined_score'] < 0) , -1, 0))
    
    working_df = working_df[['Date', 'Close','trade_signal', 'trade_signal_lowvol', 'expected_1d_move', 'expected_1w_move']]
    return working_df

def get_open_close_price_option_lowvol(option_ticker, entry_date, exit_date):
    #get next trading day
    nyse = mcal.get_calendar('NYSE')
    next_open_day = nyse.valid_days(start_date=entry_date, end_date=entry_date + pd.Timedelta(days=10))[1]
    date_str = entry_date.strftime('%Y-%m-%d')
    next_open_day_str = next_open_day.strftime('%Y-%m-%d')

    closing_date = nyse.valid_days(start_date=exit_date, end_date=exit_date + pd.Timedelta(days=10))[1]
    closing_date_str = closing_date.strftime('%Y-%m-%d')
    #entry price, open at next day. close price = open on next day
    print(f'decision date: {date_str}. trade executed at open of {next_open_day_str} and closed at the open of {closing_date_str} for {option_ticker}.')

    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{next_open_day_str}/{closing_date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    print(ohlcvdf)
    initial_price = ohlcvdf['o'].iloc[0]
    end_price = ohlcvdf['o'].iloc[-1]
    return initial_price, end_price

def get_trades_df_lowvol(trade_signals_df):
    options_bt_df = trade_signals_df.copy()

    trades = []
    in_trade = False

    for idx, row in options_bt_df.iterrows():
        signal = row['trade_signal_lowvol']
        date = row['Date']
        price = row['Close']

        if not in_trade:
            if signal == 1:
                # Enter long trade
                entry_date = date
                entry_price = price
                position = "long"
                in_trade = True

            elif signal == -1:
                # Enter short trade
                entry_date = date
                entry_price = price
                position = "short"
                in_trade = True

        else:
            if position == "long":
                if signal == 0:
                    # Close long trade
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "trade_type": "long"
                    })
                    in_trade = False
                    position = None

                elif signal == -1:
                    # Close long trade and enter short trade
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "trade_type": "long"
                    })
                    entry_date = date
                    entry_price = price
                    position = "short"

            elif position == "short":
                if signal == 0:
                    # Close short trade
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "trade_type": "short"
                    })
                    in_trade = False
                    position = None

                elif signal == 1:
                    # Close short trade and enter long trade
                    trades.append({
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "trade_type": "short"
                    })
                    entry_date = date
                    entry_price = price
                    position = "long"

    trades_df = pd.DataFrame(trades)
    return trades_df

def get_open_close_price_option_lowvol(option_ticker, date_str):
    #get next trading day
    nyse = mcal.get_calendar('NYSE')
    date = pd.to_datetime(date_str, format = '%Y-%m-%d')
    
    entry_date_plus5 = date+pd.Timedelta(days=5)
    entry_date_plus5_str = entry_date_plus5.strftime('%Y-%m-%d')

    next_open_day = nyse.valid_days(start_date=date_str, end_date=entry_date_plus5_str)[2]
    next_open_day_str = next_open_day.strftime('%Y-%m-%d')

    print(f'trade executed at open of {date_str}, end at open of {next_open_day_str}.')
    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{date_str}/{next_open_day_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    if len(ohlcvdf) == 1:
        initial_price = ohlcvdf['o'].iloc[0]
        end_price = ohlcvdf['c'].iloc[0]
        print('[bold red]using same day open & close price since couldnt get other prices[/bold red]')
    else:
        initial_price = ohlcvdf['o'].iloc[0]
        end_price = ohlcvdf['o'].iloc[1]

    return initial_price, end_price, next_open_day_str

def lowvol_options_backtest(lowvol_trades_df, ticker):
    all_trades_df = pd.DataFrame()
    for idx,row in lowvol_trades_df.iterrows():
        entry_date = row['entry_date']
        exit_date = row['exit_date']
        direction = row['trade_type']
        option_type = 'call' if direction == 'long' else 'put'
        entry_date_plus5 = entry_date+pd.Timedelta(days=5)
        entry_date_str = entry_date.strftime('%Y-%m-%d')
        entry_date_plus5_str = entry_date_plus5.strftime('%Y-%m-%d')

        exit_date_plus5 = exit_date+pd.Timedelta(days=5)
        exit_date_str = exit_date.strftime('%Y-%m-%d')
        exit_date_plus5_str = exit_date_plus5.strftime('%Y-%m-%d')


        nyse_cal = mcal.get_calendar('NYSE')
        options_entry_date = nyse_cal.valid_days(start_date=entry_date_str, end_date=entry_date_plus5_str)[1]
        options_exit_date = nyse_cal.valid_days(start_date=exit_date_str, end_date=exit_date_plus5_str)[1]

        trading_days_ls = nyse_cal.valid_days(start_date=options_entry_date.strftime('%Y-%m-%d'), end_date=options_exit_date.strftime('%Y-%m-%d'))

        trade_entry_date_ls = []
        trade_exit_date_ls = []
        trade_entry_price_ls = []
        trade_exit_price_ls = []
        trade_option_ticker_ls = []

        for trade_day in trading_days_ls:
            trade_day_str = trade_day.strftime('%Y-%m-%d')
            #get open price at trading day 
            ohlc = get_ohlcv_df(ticker, trade_day_str, trade_day_str, timeframe='1/day')
            underlying_open_price = ohlc.iloc[0]['Open']
            primary_strike = round(underlying_open_price / 5) * 5
            fallback_strike= round(underlying_open_price / 10) * 10
            options_df = get_options_df(ticker, trade_day_str, primary_strike, fallback_strike, option_type)
            actual_option = get_option(options_df, trade_day_str, 7)
            option_ticker = actual_option.iloc[0]['ticker']
            print(f'{option_ticker=}')
            option_entry_price, option_exit_price, trade_exit_date_str = get_open_close_price_option_lowvol(option_ticker, trade_day_str)
            
            print(f'{trade_day_str=}')
            print(f'{trade_exit_date_str=}')
            print(f'{option_entry_price=}')
            print(f'{option_exit_price=}')
            
            trade_entry_date_ls.append(trade_day_str)
            trade_exit_date_ls.append(trade_exit_date_str)
            trade_option_ticker_ls.append(option_ticker)
            trade_entry_price_ls.append(option_entry_price)
            trade_exit_price_ls.append(option_exit_price)

        trades_dfdf = pd.DataFrame({'entry_date' : trade_entry_date_ls,
                                    'exit_date' : trade_exit_date_ls,
                                    'ticker' : trade_option_ticker_ls,
                                    'entry_price' : trade_entry_price_ls,
                                    'exit_price' : trade_exit_price_ls}, index = [x for x in range(len(trade_entry_date_ls))])
        print(trades_dfdf)

        all_trades_df = pd.concat([all_trades_df, trades_dfdf], axis = 0)
    all_trades_df.reset_index(inplace=True, drop=True)
    return all_trades_df

def backtest_get_puts_contract_by_exp(ticker, query_date_str, sell_put_strike, desired_exp_date):
# sep/dec/apr
    lower_bound_strike = 0.3*sell_put_strike
    upper_bound_strike = 1.8*sell_put_strike

    url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={ticker}&strike_price.gte={lower_bound_strike}&strike_price.lte={upper_bound_strike}&contract_type=put&expiration_date={desired_exp_date}&as_of={query_date_str}&expired=false&order=asc&limit=1000&sort=expiration_date&apiKey={API_KEY}"
    contracts_data = pd.json_normalize(requests.get(url).json()["results"])
    contracts_data['strike_diff'] = abs(sell_put_strike - contracts_data['strike_price'])
    strike_diff_arr = np.unique(contracts_data['strike_diff'])
    smallest_strike = strike_diff_arr[0]
    # sec_smallest_strike = strike_diff_arr[1]
    return contracts_data[contracts_data['strike_diff'] == smallest_strike]

def find_closest_date(timestamp_ls, target_datetime) :
    """
    Finds the closest date in the list to the target datetime.

    Parameters:
        timestamps (List[datetime]): A list of datetime objects.
        target_datetime (datetime): The target datetime to compare against.

    Returns:
        datetime: The closest datetime from the list.
    """
    # Calculate the absolute difference for each timestamp and find the minimum
    closest_date = min(timestamp_ls, key=lambda x: abs(x - target_datetime))
    return closest_date

def run_options_expiry_ls(start_year, end_year):

    nyse = mcal.get_calendar('NYSE')
    def get_third_friday(year, month):
        """
        Get the 3rd Friday of a given year and month using pd.to_datetime.
        """
        # Create a date range for the month
        start_date = pd.to_datetime(f"{year}-{month}-01")
        end_date = (start_date + pd.offsets.MonthEnd(0)).normalize()  # Last day of the month
        
        # Filter for Fridays
        fridays = pd.date_range(start=start_date, end=end_date, freq='W-FRI')
        
        # Return the 3rd Friday
        return fridays[2]  # 3rd Friday (index 2)

    def get_options_expiry_dates(start_year, end_year):
        """
        Generate a list of options expiration dates for the 3rd Friday of each month,
        adjusting for US market holidays.
        """
        expiry_dates = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):  # Loop through all months
                # Get the 3rd Friday of the month
                third_friday = get_third_friday(year, month)
                
                # Check if the 3rd Friday is a market holiday
                schedule = nyse.valid_days(start_date=third_friday, end_date=third_friday)
                if len(schedule) == 0:  # No trading on this day (holiday)
                    # Adjust to the preceding Thursday
                    adjusted_date = third_friday - pd.Timedelta(days=1)
                    expiry_dates.append(adjusted_date)
                else:
                    # Use the 3rd Friday as the expiration date
                    expiry_dates.append(third_friday)
        
        return expiry_dates
    options_expiry_dates = get_options_expiry_dates(start_year, end_year)
    return options_expiry_dates

def get_third_thursdays(dates):
    # Group by year and month using tuples (hashable)
    grouped = dates.groupby(list(zip(dates.year, dates.month)))
    third_thursdays = []
    for year_month_tup, dates_ls in grouped.items():
        if len(dates_ls) >= 3:  # Ensure there are at least 3 Thursdays in the month
            third_thursdays.append(dates_ls[2])  # Index 2 corresponds to the 3rd Thursday
    return pd.to_datetime(third_thursdays).strftime('%Y-%m-%d').tolist()

def get_next_x_trading_date(curr_date, x):
    nyse = mcal.get_calendar("NYSE")
    trading_days = nyse.valid_days(start_date= pd.to_datetime(curr_date), end_date=pd.to_datetime(curr_date) +pd.Timedelta(days=x+100))
    next_x_day = trading_days[x]
    return next_x_day


def get_open_close_price_options_backtest(option_ticker, trade_date_str):

    trade_end = pd.to_datetime(trade_date_str , format = "%Y-%m-%d") + pd.Timedelta(days=250)
    trade_end_str = trade_end.strftime("%Y-%m-%d")
    print(f'trade executed at open of {trade_date_str}.')

    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{trade_date_str}/{trade_end_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    # url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{trade_date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    # print(requests.get(url_primary).json())
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    print(ohlcvdf)
    initial_price = ohlcvdf['o'].iloc[0]
    end_price = ohlcvdf['c'].iloc[-1]
    end_date = ohlcvdf.index[-1]
    end_date_str = end_date.strftime("%Y-%m-%d")
    end_date_dt = pd.to_datetime(end_date_str, format="%Y-%m-%d")
    return initial_price, end_price, end_date_dt


def get_open_close_price_options_backtest_quotes(option_ticker, trade_date_str, option_expiry_str):

    trade_end = pd.to_datetime(trade_date_str , format = "%Y-%m-%d") + pd.Timedelta(days=250)
    trade_end_str = trade_end.strftime("%Y-%m-%d")
    print(f'trade executed at open of {trade_date_str}.')
    
    trade_open_url = f"https://api.polygon.io/v3/quotes/{option_ticker}?timestamp={trade_date_str}&order=asc&limit=50000&sort=timestamp&apiKey={API_KEY}"
    trade_close_url = f"https://api.polygon.io/v3/quotes/{option_ticker}?timestamp={trade_end_str}&order=desc&limit=50000&sort=timestamp&apiKey={API_KEY}"

    # start_timestamp =  trade_date_str + " 14:30:00"
    # end_timestamp = trade_date_str + " 14:35:00"
    # trade_open_df = pd.json_normalize(requests.get(trade_open_url).json()['results'])
    # trade_open_df = trade_open_df[(trade_open_df['bid_size'] != 0) & (trade_open_df['ask_size'] != 0)]
    # trade_open_df['sip_timestamp'] = pd.to_datetime(trade_open_df['sip_timestamp'], unit='ns')
    # trade_open_df = trade_open_df[trade_open_df['sip_timestamp'].between(pd.Timestamp(start_timestamp), pd.Timestamp(end_timestamp))]
    # trade_open_df['bid_ask_spread'] = trade_open_df['ask_price'] - trade_open_df['bid_price'] 
    # mininum_bidask = trade_open_df['bid_ask_spread'].min()
    # trade_open_df = trade_open_df[trade_open_df['bid_ask_spread'] == mininum_bidask]
    # trade_open_df['midprice'] = (trade_open_df['ask_price'] + trade_open_df['bid_price'])/2
    # initial_premium = trade_open_df['midprice'].iloc[0] 

    start_timestamp_close =  trade_date_str + " 14:30:00"
    end_timestamp_close = trade_date_str + " 14:35:00"
    trade_close_df = pd.json_normalize(requests.get(trade_close_url).json()['results'])
    # trade_close_df = trade_close_df[(trade_close_df['bid_size'] != 0) & (trade_close_df['ask_size'] != 0)]
    print(trade_close_df)
    # trade_close_df['sip_timestamp'] = pd.to_datetime(trade_close_df['sip_timestamp'], unit='ns')
    # trade_close_df = trade_close_df[trade_close_df['sip_timestamp'].between(pd.Timestamp(start_timestamp_close), pd.Timestamp(end_timestamp_close))]
    # trade_close_df['bid_ask_spread'] = trade_close_df['ask_price'] - trade_close_df['bid_price'] 
    # mininum_bidask = trade_close_df['bid_ask_spread'].min()
    # trade_close_df = trade_close_df[trade_close_df['bid_ask_spread'] == mininum_bidask]
    # trade_close_df['midprice'] = (trade_close_df['ask_price'] + trade_close_df['bid_price'])/2
    # exit_price = trade_close_df['midprice'].iloc[0] 




    # end_date_str = end_date.strftime("%Y-%m-%d")
    # end_date_dt = pd.to_datetime(end_date_str, format="%Y-%m-%d")
    # return initial_price, end_price, end_date_dt

def get_close_price_at_date(stock_ticker, date_str):
    url_primary = f"https://api.polygon.io/v2/aggs/ticker/{stock_ticker}/range/1/day/{date_str}/{date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    ohlcvdf = pd.json_normalize(requests.get(url_primary).json()['results']).set_index("t")
    ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    print(ohlcvdf)
    close_price = ohlcvdf['c'].iloc[-1]
    return close_price


def get_option_quotes(option_ticker, date_str):
    # url = f"https://api.polygon.io/v3/trades/{option_ticker}?timestamp={date_str}&order=asc&limit=50000&sort=timestamp&apiKey={API_KEY}"
    # url_primary = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{date_str}/{date_str}?adjusted=true&sort=asc&apiKey={API_KEY}"
    
    url = f"https://api.polygon.io/v3/quotes/{option_ticker}?timestamp={date_str}&order=desc&limit=50000&sort=timestamp&apiKey={API_KEY}"
    trades_df = pd.json_normalize(requests.get(url).json()['results'])
    trades_df['sip_timestamp'] = pd.to_datetime(trades_df['sip_timestamp'], unit='ns')
        # Step 2: Localize the datetime to UTC
    trades_df['sip_timestamp'] = trades_df['sip_timestamp'].dt.tz_localize('UTC')

    # Step 3: Convert UTC to Eastern Time (NYSE time)
    trades_df['sip_timestamp'] = trades_df['sip_timestamp'].dt.tz_convert('US/Eastern')
    trades_df = trades_df.head(50)
    # ohlcvdf.index = pd.to_datetime(ohlcvdf.index, unit="ms", utc=True).tz_convert("America/New_York")
    # print(ohlcvdf)
    # close_price = ohlcvdf['c'].iloc[0]

    # print(f'Date: {date_str}. Put close price at {close_price}.')
    return trades_df


def get_options_greeks_and_iv(polygon_ticker, options_ticker):
    url = f"https://api.polygon.io/v3/snapshot/options/{polygon_ticker}/{options_ticker}?apiKey={API_KEY}"
    snapshot = pd.json_normalize(requests.get(url).json()['results'])
    delta = snapshot['greeks.delta'].iloc[0]
    gamma = snapshot['greeks.gamma'].iloc[0]
    theta = snapshot['greeks.theta'].iloc[0]
    vega = snapshot['greeks.vega'].iloc[0]
    iv = snapshot['implied_volatility'].iloc[0] * 100
    return delta, gamma, theta, vega, iv 

# df = get_puts_contract_by_exp("SPY", "2025-04-07", 400, month_to_exp = 12)
# print(df)
# quotes = get_option_quotes("O:SPY260320P00400000", "2025-04-07")
# print(quotes)
# get_open_close_price_options_backtest_quotes("O:SPY260320P00400000", "2025-04-07", "")

# delta, gamma, theta, vega, iv = get_options_greeks_and_iv("SPY", "O:SPY260320P00400000")
# print(delta, gamma, theta, vega, iv)