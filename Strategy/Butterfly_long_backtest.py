import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ButterflyBacktest:
    def __init__(self, options_data, underlying_data, vol_predictions, spread_pct=0.01, commission=1.0, slippage=0.01, min_open_interest=10, min_volume=1, vol_exit_threshold=0.5):
        self.options_data = options_data
        self.underlying_data = underlying_data
        self.vol_predictions = vol_predictions
        self.spread_pct = spread_pct
        self.commission = commission
        self.slippage = slippage
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume
        self.vol_exit_threshold = vol_exit_threshold  # e.g., 0.5 = 50% change triggers exit

        self.underlying_data['realized_vol_90d'] = (
            self.underlying_data['close']
            .pct_change()
            .rolling(90)
            .std()
            * np.sqrt(252)
        )

    def get_predicted_vol(self, entry_day, expiry_day):
        row = self.vol_predictions[
            (self.vol_predictions['prediction_day'] == entry_day) &
            (self.vol_predictions['predicted_day'] == expiry_day)
        ]
        if not row.empty:
            return row.iloc[0]['predicted_vol']
        return np.nan

    def entry_rule(self, entry_day, expiry_day, butterfly):
        pred_vol = self.get_predicted_vol(entry_day, expiry_day)
        realized_vol_90d = self.underlying_data.loc[self.underlying_data['timestamp'] == entry_day, 'realized_vol_90d'].values[0]
        if pd.isna(pred_vol) or pd.isna(realized_vol_90d):
            return False
        # enter if predicted vol < realized vol
        if pred_vol >= realized_vol_90d:
            print(f"Skipping entry on {entry_day}: Predicted vol {pred_vol} >= Realized vol {realized_vol_90d}")
            return False
        

        # Underlying volume filter
        underlying_row = self.underlying_data[self.underlying_data['timestamp'] == entry_day]
        if underlying_row.empty or underlying_row['close'].values[0] <= 0: # if no underlying data or price is zero
            return False
        if 'volume' in underlying_row and underlying_row['volume'].values[0] < self.min_volume: # if underlying volume is too low
            print(f"Skipping entry on {entry_day}: Underlying volume {underlying_row['volume'].values[0]} < {self.min_volume}")
            return False

        # Option liquidity filter
        for leg in butterfly.values():
            if ('open_interest' in leg and (pd.isna(leg['open_interest']) or leg['open_interest'] < self.min_open_interest)): # skip if open interest is too low
                print(f"Skipping entry on {entry_day}: Open interest {leg['open_interest']} < {self.min_open_interest}")
                return False
            if ('opt_close' in leg and (pd.isna(leg['opt_close']) or leg['opt_close'] <= 0)):
                print(f"Skipping entry on {entry_day}: Option price {leg['opt_close']} <= 0")
                return False

        # Cost filter: don't enter if butterfly is too expensive
        k1 = butterfly['long1']['strike_price']
        k2 = butterfly['short2']['strike_price']
        k3 = butterfly['long3']['strike_price']
        spread_width = k3 - k2  # assuming symmetric butterfly

        entry_price = butterfly['long1']['opt_close'] + butterfly['long3']['opt_close'] - 2 * butterfly['short2']['opt_close']
        max_profit = spread_width  # max payoff if ATM at expiry

        if entry_price > 0.7 * max_profit: # if entry price is more than 70% of max profit, skip
            print(f"Skipping entry on {entry_day}: Entry price {entry_price} > 70% of max profit {max_profit}")
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
    
    def should_exit_early(self, open_trade, current_day):
        # Check if new predicted vol for remaining expiry is radically different
        expiry_day = open_trade['expiry']
        pred_vol_now = self.get_predicted_vol(current_day, expiry_day)
        pred_vol_entry = open_trade['pred_vol_entry']
        if pd.isna(pred_vol_now) or pd.isna(pred_vol_entry):
            return False
        # Exit if predicted vol changes by more than threshold (relative)
        if abs(pred_vol_now - pred_vol_entry) / pred_vol_entry > self.vol_exit_threshold: 
            print(f"Early exit on {current_day}: Predicted vol changed from {pred_vol_entry} to {pred_vol_now}")
            return True
        return False
    
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
            exit_prices[leg] = match.iloc[0]['opt_close']
        return exit_prices

    def run(self, option_type='call', starting_cash=1000):
        results = []
        cash = starting_cash
        equity_curve = []
        open_trades = []

        # Combine all relevant dates (trading days and expiries)
        all_dates = sorted(set(self.underlying_data['timestamp']) | set(self.options_data['expiration_date']))

        for day in all_dates:
            # 1. Check for new trade opportunities (entry)
            if day in self.underlying_data['timestamp']:
                # Find expiry 5 trading days ahead
                expiry_candidates = self.options_data[
                    (self.options_data['timestamp'] == day) & (self.options_data['dtm'] == 5)
                ]['expiration_date'].unique()
                if len(expiry_candidates) == 0:
                    continue
                expiry = expiry_candidates[0]
                options_day = self.options_data[
                    (self.options_data['timestamp'] == day) &
                    (self.options_data['expiration_date'] == expiry) &
                    (self.options_data['type'] == option_type)
                ]
                if options_day.empty:
                    continue
                butterfly = self.construct_butterfly(options_day)
                if butterfly is None:
                    continue
                pred_vol_entry = self.get_predicted_vol(day, expiry)
                if not self.entry_rule(day, expiry, butterfly):
                    continue

                # Calculate entry price and costs
                entry_price = butterfly['long1']['opt_close'] + butterfly['long3']['opt_close'] - 2 * butterfly['short2']['opt_close']
                total_legs = 4
                total_commission = self.commission * total_legs
                total_slippage = self.slippage * (abs(butterfly['long1']['opt_close']) +
                                                abs(butterfly['long3']['opt_close']) +
                                                2 * abs(butterfly['short2']['opt_close']))
                total_cost_per_butterfly = entry_price + total_commission + total_slippage

                # Liquidity constraints
                min_oi = min(
                    butterfly['long1'].get('open_interest', np.inf),
                    butterfly['long3'].get('open_interest', np.inf),
                    butterfly['short2'].get('open_interest', np.inf) // 2
                )
                min_vol = min(
                    butterfly['long1'].get('volume', np.inf),
                    butterfly['long3'].get('volume', np.inf),
                    butterfly['short2'].get('volume', np.inf) // 2
                )
                max_liquidity = int(min(min_oi, min_vol))

                # Max butterflies to buy
                n_butterflies = min(int(cash // total_cost_per_butterfly), max_liquidity)
                if n_butterflies < 1:
                    continue

                total_entry_cost = total_cost_per_butterfly * n_butterflies

                # Enter trade: subtract cost, add to open trades
                cash -= total_entry_cost
                open_trades.append({
                    'entry_date': day,
                    'expiry': expiry,
                    'butterfly': butterfly,
                    'entry_price': entry_price,
                    'commission': total_commission,
                    'slippage': total_slippage,
                    'total_cost': total_entry_cost,
                    'n_butterflies': n_butterflies,
                    'pred_vol_entry': pred_vol_entry
                })

            # 2. Check for early exit or expiry for open trades
            to_remove = []
            for i, trade in enumerate(open_trades):
                # Early exit if volatility prediction changes radically
                early_exit = self.should_exit_early(trade, day)
                is_expiry = (trade['expiry'] == day)
                if early_exit or is_expiry:
                    exit_prices = self.get_exit_prices(trade['butterfly'], day)
                    if exit_prices is None:
                        continue
                    exit_price = exit_prices['long1'] + exit_prices['long3'] - 2 * exit_prices['short2']
                    exit_slippage = self.slippage * (abs(exit_prices['long1']) +
                                                    abs(exit_prices['long3']) +
                                                    2 * abs(exit_prices['short2']))
                    total_exit_value = exit_price * trade['n_butterflies']
                    total_slippage_all = (trade['slippage'] + exit_slippage) * trade['n_butterflies']
                    total_commission_all = trade['commission'] * trade['n_butterflies']
                    pnl = (exit_price - trade['entry_price'] - trade['commission'] - trade['slippage'] - exit_slippage) * trade['n_butterflies']
                    cash += total_exit_value

                    results.append({
                        'entry_date': trade['entry_date'],
                        'exit_date': day,
                        'expiry': trade['expiry'],
                        'entry_price': trade['entry_price'],
                        'exit_price': exit_price,
                        'n_butterflies': trade['n_butterflies'],
                        'pnl': pnl,
                        'underlying_price': self.underlying_data.loc[self.underlying_data['timestamp'] == trade['entry_date'], 'close'].values[0],
                        'strikes': (trade['butterfly']['long1']['strike_price'],
                                    trade['butterfly']['short2']['strike_price'],
                                    trade['butterfly']['long3']['strike_price']),
                        'commission': total_commission_all,
                        'slippage': total_slippage_all,
                        'cash_after_trade': cash,
                        'early_exit': early_exit
                    })
                    to_remove.append(i)
            for idx in sorted(to_remove, reverse=True):
                open_trades.pop(idx)

            # 3. Mark-to-market for equity
            mtm_value = 0
            for trade in open_trades:
                current_prices = self.get_exit_prices(trade['butterfly'], day)
                if current_prices is not None:
                    mtm_value += (current_prices['long1'] + current_prices['long3'] - 2 * current_prices['short2']) * trade['n_butterflies']
            equity_curve.append({'date': day, 'cash': cash, 'equity': cash + mtm_value})

        results_df = pd.DataFrame(results)
        equity_df = pd.DataFrame(equity_curve)

        # Add summary statistics
        if not results_df.empty:
            total_trades = len(results_df)
            total_profit = results_df['pnl'].sum()
            avg_profit = results_df['pnl'].mean()
            win_rate = (results_df['pnl'] > 0).mean()
            max_drawdown = (equity_df['equity'].cummax() - equity_df['equity']).max()
            ending_cash = cash
            ending_equity = equity_df['equity'].iloc[-1]
            stats = {
                'total_trades': total_trades,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'ending_cash': ending_cash,
                'ending_equity': ending_equity
            }
            print("Backtest Statistics (1 Year):")
            for k, v in stats.items():
                print(f"{k}: {v}")
        else:
            print("No trades executed.")

        # Plot cash and equity curve with trade signals
        if not equity_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df['date'], equity_df['cash'], label='Cash Balance', color='blue')
            plt.plot(equity_df['date'], equity_df['equity'], label='Total Equity', color='green')
            if not results_df.empty:
                plt.scatter(results_df['entry_date'], results_df['cash_after_trade'], color='red', marker='^', label='Trade Entry')
            plt.title('Account Cash & Equity Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value ($)')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No equity data to plot.")

        return results_df, equity_df


























# -----------------------------
# def run(self, option_type='call', starting_cash=1000):
#     results = []
#     cash = starting_cash
#     equity_curve = []
#     for day in self.underlying_data['timestamp']: # Iterate over each trading day
#         if cash <= 0:
#             print("Insufficient cash to continue trading.")
#             break
#         if day not in self.pred_vol_series.index: # Skip if no prediction for this day
#             continue
#         if pd.isna(self.underlying_data.loc[self.underlying_data['timestamp'] == day, 'realized_vol_90d']).all(): # Skip if no realized vol data
#             continue
#         options_day = self.select_options_for_day(day, option_type) 
#         if options_day.empty: # No options data for this day
#             continue
#         butterfly = self.construct_butterfly(options_day) #
#         if butterfly is None: # No valid butterfly found
#             print(f"No valid butterfly found for {day}.")
#             continue
#         if not self.entry_rule(day, butterfly, option_type): # Entry rule not met
#             print(f"Entry rule not met for {day}.")
#             continue
#         entry_price = butterfly['long1']['opt_close'] + butterfly['long3']['opt_close'] - 2 * butterfly['short2']['opt_close']
#         expiry = butterfly['long1']['expiration_date']
#         exit_prices = self.get_exit_prices(butterfly, expiry) 
#         if exit_prices is None: # No exit prices available for expiry
#             continue
#         exit_price = exit_prices['long1'] + exit_prices['long3'] - 2 * exit_prices['short2']

#         # Trading costs: commission and slippage (per leg, both entry and exit)
#         total_legs = 4  # 2 long, 2 short
#         total_commission = self.commission * total_legs
#         total_slippage = self.slippage * (abs(butterfly['long1']['opt_close']) +
#                                           abs(butterfly['long3']['opt_close']) +
#                                           2 * abs(butterfly['short2']['opt_close']))
#         total_slippage += self.slippage * (abs(exit_prices['long1']) +
#                                            abs(exit_prices['long3']) +
#                                            2 * abs(exit_prices['short2']))

#         total_cost_per_butterfly = entry_price + total_commission + total_slippage

#         # Find the minimum available contracts you can realistically buy
#         min_oi = min(
#             butterfly['long1'].get('open_interest', np.inf),
#             butterfly['long3'].get('open_interest', np.inf),
#             butterfly['short2'].get('open_interest', np.inf) // 2  # short 2 contracts
#         )
#         min_vol = min(
#             butterfly['long1'].get('volume', np.inf),
#             butterfly['long3'].get('volume', np.inf),
#             butterfly['short2'].get('volume', np.inf) // 2
#         )
#         max_liquidity = int(min(min_oi, min_vol))

#         # Calculate how many butterflies we can buy, considering both cash and liquidity
#         n_butterflies = min(int(cash // total_cost_per_butterfly), max_liquidity)
#         if n_butterflies < 1:
#             continue

#         # Scale all costs and P&L by n_butterflies
#         total_entry_cost = total_cost_per_butterfly * n_butterflies
#         total_exit_value = exit_price * n_butterflies
#         total_commission_all = total_commission * n_butterflies
#         total_slippage_all = total_slippage * n_butterflies
#         pnl = (exit_price - entry_price - total_commission - total_slippage) * n_butterflies

#         # Enter trade: subtract cost
#         cash -= total_entry_cost

#         # Exit trade: add proceeds
#         cash += total_exit_value

#         results.append({
#             'entry_date': day,
#             'expiry': expiry,
#             'entry_price': entry_price,
#             'exit_price': exit_price,
#             'n_butterflies': n_butterflies,
#             'pnl': pnl,
#             'underlying_price': self.underlying_data.loc[self.underlying_data['timestamp'] == day, 'close'].values[0],
#             'strikes': (butterfly['long1']['strike_price'], butterfly['short2']['strike_price'], butterfly['long3']['strike_price']),
#             'commission': total_commission_all,
#             'slippage': total_slippage_all,
#             'cash_after_trade': cash
#         })
#         equity_curve.append({'date': expiry, 'cash': cash})

#     results_df = pd.DataFrame(results)
#     equity_df = pd.DataFrame(equity_curve)

#     # Add summary statistics
#     if not results_df.empty:
#         total_trades = len(results_df)
#         total_profit = results_df['pnl'].sum()
#         avg_profit = results_df['pnl'].mean()
#         win_rate = (results_df['pnl'] > 0).mean()
#         max_drawdown = (results_df['pnl'].cumsum().cummax() - results_df['pnl'].cumsum()).max()
#         ending_cash = cash
#         stats = {
#             'total_trades': total_trades,
#             'total_profit': total_profit,
#             'avg_profit': avg_profit,
#             'win_rate': win_rate,
#             'max_drawdown': max_drawdown,
#             'ending_cash': ending_cash
#         }
#         print("Backtest Statistics (1 Year):")
#         for k, v in stats.items():
#             print(f"{k}: {v}")
#     else:
#         print("No trades executed.")

#     # Plot cash balance and trade signals
#     if not equity_df.empty:
#         plt.figure(figsize=(12, 6))
#         plt.plot(equity_df['date'], equity_df['cash'], label='Cash Balance', color='blue')
#         if not results_df.empty:
#             plt.scatter(results_df['entry_date'], results_df['cash_after_trade'], color='red', marker='^', label='Trade Entry')
#         plt.title('Account Cash Balance Over Time')
#         plt.xlabel('Date')
#         plt.ylabel('Cash ($)')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No equity data to plot.")

#     return results_df, equity_df