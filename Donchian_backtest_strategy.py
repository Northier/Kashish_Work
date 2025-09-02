import pandas as pd
import numpy as np
import os


def donchian_channel(df, window=50):
    df['Donchian_High'] = df['High'].rolling(window).max()
    df['Donchian_Low'] = df['Low'].rolling(window).min()
    return df

def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift(1))
    df['L-C'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(period).mean()
    return df

def backtest_donchian_trend(df, window=50, ma_period=200, atr_period=14, sl_atr=1.0, tp_atr=2.0, rsi_period=14):
    import talib
    
    df = donchian_channel(df, window)
    df['SMA'] = df['Close'].rolling(ma_period).mean()
    df = calculate_atr(df, atr_period)
    
    # RSI momentum filter
    df['RSI'] = talib.RSI(df['Close'], timeperiod=rsi_period)
    
    df['Signal'] = 0
    df.loc[(df['Close'] > df['Donchian_High'].shift(1)), 'Signal'] = 1  # long
    df.loc[(df['Close'] < df['Donchian_Low'].shift(1)), 'Signal'] = -1 # short

    # Apply trend filter
    df['Trend_Long'] = (df['Close'] > df['SMA']) & (df['RSI'] > 50)  # bullish trend
    df['Trend_Short'] = (df['Close'] < df['SMA']) & (df['RSI'] < 50) # bearish trend
    
    trades = []
    position = None
    entry_price = 0
    sl = 0
    tp = 0

    for i in range(1, len(df)):
        signal = df['Signal'].iloc[i]
        close = df['Close'].iloc[i]
        atr = df['ATR'].iloc[i]
        
        if np.isnan(atr):
            continue
        
        if signal == 1 and not df['Trend_Long'].iloc[i]:
            signal = 0
        if signal == -1 and not df['Trend_Short'].iloc[i]:
            signal = 0
        
        if position is None and signal != 0:
            entry_price = close
            position = 'Long' if signal == 1 else 'Short'
            sl = entry_price - sl_atr*atr if position=='Long' else entry_price + sl_atr*atr
            tp = entry_price + tp_atr*atr if position=='Long' else entry_price - tp_atr*atr
            trades.append({
                'Entry_Index': i,
                'Entry_Price': entry_price,
                'Position': position,
                'SL': sl,
                'TP': tp,
                'Exit_Index': None,
                'Exit_Price': None,
                'Result': None
            })
        
        if position is not None:
            if position == 'Long' and (close <= sl or close >= tp):
                trades[-1]['Exit_Index'] = i
                trades[-1]['Exit_Price'] = close
                trades[-1]['Result'] = close - entry_price
                position = None
            elif position == 'Short' and (close >= sl or close <= tp):
                trades[-1]['Exit_Index'] = i
                trades[-1]['Exit_Price'] = close
                trades[-1]['Result'] = entry_price - close
                position = None

    trade_log = pd.DataFrame(trades)
    trade_log['PnL'] = trade_log['Result'].fillna(0)
    trade_log['Total_PnL'] = trade_log['PnL'].sum()
    
    total_trades = len(trade_log)
    winning_trades = len(trade_log[trade_log['Result'] > 0])
    losing_trades = len(trade_log[trade_log['Result'] <= 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    avg_pnl = trade_log['PnL'].mean() if total_trades > 0 else 0
    gross_profit = trade_log[trade_log['PnL'] > 0]['PnL'].sum()
    gross_loss = -trade_log[trade_log['PnL'] < 0]['PnL'].sum()
    profit_factor = (gross_profit / gross_loss) if gross_loss != 0 else np.nan

    metrics = {
        'Total_Trades': total_trades,
        'Winning_Trades': winning_trades,
        'Losing_Trades': losing_trades,
        'Win_Rate_%': round(win_rate, 2),
        'Average_PnL': round(avg_pnl, 2),
        'Gross_Profit': round(gross_profit, 2),
        'Gross_Loss': round(gross_loss, 2),
        'Profit_Factor': round(profit_factor, 2) if not np.isnan(profit_factor) else np.nan,
        'Total_PnL': trade_log['Total_PnL'].iloc[0] if len(trade_log)>0 else 0
    }

    return trade_log, metrics



path = 'C:\\Users\\KASHISH RANA\\OneDrive\\Desktop\\stock_market'
files = [
    'resampled_5min_NSE_ABB.csv',
    'resampled_15min_NSE_ABB.csv',
    'resampled_30min_NSE_ABB.csv',
    'resampled_1hr_NSE_ABB.csv',
    'resampled_1D_NSE_ABB.csv',
    'resampled_5min_NSE_INDHOTEL.csv',
    'resampled_15min_NSE_INDHOTEL.csv',
    'resampled_30min_NSE_INDHOTEL.csv',
    'resampled_1hr_NSE_INDHOTEL.csv',
    'resampled_1D_NSE_INDHOTEL.csv',
]

sl_multipliers = [0.5, 1.0, 1.5, 2.0]
tp_multipliers = [1.0, 2.0, 3.0, 4.0]

all_trades = []
performance_summary = []

for file in files:
    filepath = os.path.join(path, file)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.sort_values('DateTime')

        df['Stock'] = df['Symbol'].apply(lambda x: x.split(':')[1])
        dataset_name = '_'.join(file.split('_')[1:6])
        df['Dataset'] = dataset_name

        best_total_pnl = -np.inf
        best_sl, best_tp = None, None
        best_trades = None
        best_metrics = None

        # Try all SL/TP combinations
        for sl in sl_multipliers:
            for tp in tp_multipliers:
                trades, metrics = backtest_donchian_trend(df, sl_atr=sl, tp_atr=tp)
                total_pnl = metrics['Total_PnL']
                if total_pnl > best_total_pnl:
                    best_total_pnl = total_pnl
                    best_sl, best_tp = sl, tp
                    best_trades = trades.copy()
                    best_metrics = metrics.copy()

        best_trades['SL_Mult'] = best_sl
        best_trades['TP_Mult'] = best_tp
        best_trades['File'] = file
        best_trades['Dataset'] = dataset_name
        best_trades['Stock'] = df['Stock'].iloc[0]
        all_trades.append(best_trades)

        best_metrics.update({
            'File': file,
            'Dataset': dataset_name,
            'Stock': df['Stock'].iloc[0],
            'SL_Mult': best_sl,
            'TP_Mult': best_tp
        })
        performance_summary.append(best_metrics)


all_trades_df = pd.concat(all_trades, ignore_index=True)
all_trades_df.to_csv('donchian_best_trades.csv', index=False)
print("All trades saved in donchian_best_trades.csv")

performance_df = pd.DataFrame(performance_summary)
performance_df.to_csv('donchian_best_performance_summary.csv', index=False)
print("Performance metrics saved in donchian_best_performance_summary.csv")
print(performance_df)


# Profit per timeframe
profit_per_timeframe = performance_df[['Stock', 'Dataset', 'Total_PnL', 'Winning_Trades', 'Losing_Trades']]
profit_per_timeframe.to_csv('profit_per_timeframe.csv', index=False)
print("Profit per timeframe saved → profit_per_timeframe.csv")

# Combined profit per stock across all timeframes
combined_profit = performance_df.groupby('Stock').agg({
    'Total_PnL': 'sum',
    'Winning_Trades': 'sum',
    'Losing_Trades': 'sum'
}).reset_index()

combined_profit['Win_Rate_%'] = (combined_profit['Winning_Trades'] / 
                                 (combined_profit['Winning_Trades'] + combined_profit['Losing_Trades']) * 100)

combined_profit.to_csv('combined_profit_per_stock.csv', index=False)
print("Combined profit per stock saved in combined_profit_per_stock.csv")
print(combined_profit)
