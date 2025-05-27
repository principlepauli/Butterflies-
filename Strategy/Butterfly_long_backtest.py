import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ButterflyBacktest:
    def __init__(self, options_data, underlying_data, pred_vol_series, spread_pct=0.01, commission=1.0, slippage=0.01, min_open_interest=10, min_volume=1):
        self.options_data = options_data
        self.underlying_data = underlying_data
        self.pred_vol_series = pred_vol_series
        self.spread_pct = spread_pct
        self.commission = commission  # per leg
        self.slippage = slippage      # as fraction of price per leg
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume

        self.underlying_data['realized_vol_90d'] = (
            self.underlying_data['close']
            .pct_change()
            .rolling(90)
            .std()
            * np.sqrt(252)
        )

    def entry_rule(self, day, butterfly, option_type='call'):
        # Predicted vol < realized vol
        pred_vol = self.pred_vol_series.loc[day]
        realized_vol_90d = self.underlying_data.loc[self.underlying_data['timestamp'] == day, 'realized_vol_90d'].values[0]
        if pred_vol >= realized_vol_90d:
            return False

        # Underlying volume filter
        underlying_row = self.underlying_data[self.underlying_data['timestamp'] == day]
        if underlying_row.empty or underlying_row['close'].values[0] <= 0:
            return False
        if 'volume' in underlying_row and underlying_row['volume'].values[0] < self.min_volume:
            return False

        # Option liquidity filter
        for leg in butterfly.values():
            if ('open_interest' in leg and (pd.isna(leg['open_interest']) or leg['open_interest'] < self.min_open_interest)):
                return False
            if ('close_price' in leg and (pd.isna(leg['close_price']) or leg['close_price'] <= 0)):
                return False

        # Cost filter: don't enter if butterfly is too expensive
        k1 = butterfly['long1']['strike_price']
        k2 = butterfly['short2']['strike_price']
        k3 = butterfly['long3']['strike_price']
        spread_width = k3 - k2  # assuming symmetric butterfly

        entry_price = butterfly['long1']['close_price'] + butterfly['long3']['close_price'] - 2 * butterfly['short2']['close_price']
        max_profit = spread_width  # max payoff if ATM at expiry

        if entry_price > 0.5 * max_profit:
            return False

        return True

    def select_options_for_day(self, day, option_type='call'):
        mask = (self.options_data['timestamp'] == day) & (self.options_data['dtm'] == 5) & (self.options_data['type'] == option_type)
        return self.options_data[mask]

    def construct_butterfly(self, options_day):
        strikes = np.sort(options_day['strike_price'].unique())
        if len(strikes) < 3:
            return None
        spot = options_day['SPY_close'].iloc[0]
        k2 = strikes[np.argmin(np.abs(strikes - spot))]
        spread = spot * self.spread_pct

        k1_candidates = strikes[strikes < k2]
        k3_candidates = strikes[strikes > k2]
        if len(k1_candidates) == 0 or len(k3_candidates) == 0:
            return None
        k1 = k1_candidates[np.argmin(np.abs(k1_candidates - (k2 - spread)))]
        k3 = k3_candidates[np.argmin(np.abs(k3_candidates - (k2 + spread)))]

        if len(set([k1, k2, k3])) < 3:
            return None

        legs = {
            'long1': options_day[options_day['strike_price'] == k1].iloc[0],
            'short2': options_day[options_day['strike_price'] == k2].iloc[0],
            'long3': options_day[options_day['strike_price'] == k3].iloc[0]
        }
        return legs

    def get_exit_prices(self, legs, expiry):
        exit_prices = {}
        for leg, row in legs.items():
            match = self.options_data[
                (self.options_data['expiration_date'] == expiry) &
                (self.options_data['strike_price'] == row['strike_price']) &
                (self.options_data['type'] == row['type']) &
                (self.options_data['timestamp'] == expiry)
            ]
            if match.empty:
                return None
            exit_prices[leg] = match.iloc[0]['close_price']
        return exit_prices

def run(self, option_type='call', starting_cash=1000):
    results = []
    cash = starting_cash
    equity_curve = []
    for day in self.underlying_data['timestamp']: # Iterate over each trading day
        if cash <= 0:
            print("Insufficient cash to continue trading.")
            break
        if day not in self.pred_vol_series.index: # Skip if no prediction for this day
            continue
        if pd.isna(self.underlying_data.loc[self.underlying_data['timestamp'] == day, 'realized_vol_90d']).all(): # Skip if no realized vol data
            continue
        options_day = self.select_options_for_day(day, option_type) 
        if options_day.empty: # No options data for this day
            continue
        butterfly = self.construct_butterfly(options_day) #
        if butterfly is None: # No valid butterfly found
            print(f"No valid butterfly found for {day}.")
            continue
        if not self.entry_rule(day, butterfly, option_type): # Entry rule not met
            print(f"Entry rule not met for {day}.")
            continue
        entry_price = butterfly['long1']['close_price'] + butterfly['long3']['close_price'] - 2 * butterfly['short2']['close_price']
        expiry = butterfly['long1']['expiration_date']
        exit_prices = self.get_exit_prices(butterfly, expiry) 
        if exit_prices is None: # No exit prices available for expiry
            continue
        exit_price = exit_prices['long1'] + exit_prices['long3'] - 2 * exit_prices['short2']

        # Trading costs: commission and slippage (per leg, both entry and exit)
        total_legs = 4  # 2 long, 2 short
        total_commission = self.commission * total_legs
        total_slippage = self.slippage * (abs(butterfly['long1']['close_price']) +
                                          abs(butterfly['long3']['close_price']) +
                                          2 * abs(butterfly['short2']['close_price']))
        total_slippage += self.slippage * (abs(exit_prices['long1']) +
                                           abs(exit_prices['long3']) +
                                           2 * abs(exit_prices['short2']))

        total_cost_per_butterfly = entry_price + total_commission + total_slippage

        # Find the minimum available contracts you can realistically buy
        min_oi = min(
            butterfly['long1'].get('open_interest', np.inf),
            butterfly['long3'].get('open_interest', np.inf),
            butterfly['short2'].get('open_interest', np.inf) // 2  # short 2 contracts
        )
        min_vol = min(
            butterfly['long1'].get('volume', np.inf),
            butterfly['long3'].get('volume', np.inf),
            butterfly['short2'].get('volume', np.inf) // 2
        )
        max_liquidity = int(min(min_oi, min_vol))

        # Calculate how many butterflies we can buy, considering both cash and liquidity
        n_butterflies = min(int(cash // total_cost_per_butterfly), max_liquidity)
        if n_butterflies < 1:
            continue

        # Scale all costs and P&L by n_butterflies
        total_entry_cost = total_cost_per_butterfly * n_butterflies
        total_exit_value = exit_price * n_butterflies
        total_commission_all = total_commission * n_butterflies
        total_slippage_all = total_slippage * n_butterflies
        pnl = (exit_price - entry_price - total_commission - total_slippage) * n_butterflies

        # Enter trade: subtract cost
        cash -= total_entry_cost

        # Exit trade: add proceeds
        cash += total_exit_value

        results.append({
            'entry_date': day,
            'expiry': expiry,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'n_butterflies': n_butterflies,
            'pnl': pnl,
            'underlying_price': self.underlying_data.loc[self.underlying_data['timestamp'] == day, 'close'].values[0],
            'strikes': (butterfly['long1']['strike_price'], butterfly['short2']['strike_price'], butterfly['long3']['strike_price']),
            'commission': total_commission_all,
            'slippage': total_slippage_all,
            'cash_after_trade': cash
        })
        equity_curve.append({'date': expiry, 'cash': cash})

    results_df = pd.DataFrame(results)
    equity_df = pd.DataFrame(equity_curve)

    # Add summary statistics
    if not results_df.empty:
        total_trades = len(results_df)
        total_profit = results_df['pnl'].sum()
        avg_profit = results_df['pnl'].mean()
        win_rate = (results_df['pnl'] > 0).mean()
        max_drawdown = (results_df['pnl'].cumsum().cummax() - results_df['pnl'].cumsum()).max()
        ending_cash = cash
        stats = {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'ending_cash': ending_cash
        }
        print("Backtest Statistics (1 Year):")
        for k, v in stats.items():
            print(f"{k}: {v}")
    else:
        print("No trades executed.")

    # Plot cash balance and trade signals
    if not equity_df.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['date'], equity_df['cash'], label='Cash Balance', color='blue')
        if not results_df.empty:
            plt.scatter(results_df['entry_date'], results_df['cash_after_trade'], color='red', marker='^', label='Trade Entry')
        plt.title('Account Cash Balance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cash ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No equity data to plot.")

    return results_df, equity_df