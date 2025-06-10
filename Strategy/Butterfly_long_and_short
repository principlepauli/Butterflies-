import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Butterfly long and short strategy backtest

class ButterflyBacktest_long_short:  
    def __init__(self, options_data, underlying_data, vol_predictions, spread_pct=0.01, commission=1.0, slippage=0.0, min_open_interest=10, min_volume=1, vol_exit_threshold=0.5, rolling_window=90):
        self.options_data = options_data # DataFrame with columns: timestamp, expiration_date, strike_price, type, opt_close, dtm, SPY_close, volume, open_interest
        self.underlying_data = underlying_data # DataFrame with columns: timestamp, close, volume
        self.vol_predictions = vol_predictions # DataFrame with columns: prediction_day, predicted_day, predicted_vol
        self.spread_pct = spread_pct # TUNE!!! percentage of underlying price to determine strike distances
        self.commission = commission # commission per option leg 0.1 $
        self.slippage = slippage # slippage per option leg # slippage ignoriable  
        self.min_open_interest = min_open_interest # minimum open interest for options to consider entering a trade
        self.min_volume = min_volume # minimum volume for underlying to consider entering a trade
        self.vol_exit_threshold = vol_exit_threshold #  TUNE!!! threshold for early exit based on volatility increase
        self.rolling_window = rolling_window # TUNE!!! number of days for rolling volatility calculation

        self.underlying_data['realized_vol'] = (
            self.underlying_data['close']
            .pct_change()
            .rolling(self.rolling_window)
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
    
    def check_liquidity_and_costs(self, entry_day, butterfly):
        # Underlying volume filter
        underlying_row = self.underlying_data[self.underlying_data['timestamp'] == entry_day]
        if underlying_row.empty or underlying_row['close'].values[0] <= 0:
            return False
        if 'volume' in underlying_row and underlying_row['volume'].values[0] < self.min_volume:
            print(f"Skipping entry on {entry_day}: Underlying volume {underlying_row['volume'].values[0]} < {self.min_volume}")
            return False

        # Option liquidity filter
        for leg in butterfly.values():
            if ('open_interest' in leg and (pd.isna(leg['open_interest']) or leg['open_interest'] < self.min_open_interest)):
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

        if entry_price > 0.7 * max_profit:
            print(f"Skipping entry on {entry_day}: Entry price {entry_price} > 70% of max profit {max_profit}")
            return False

        return True

    def entry_rule_long(self, entry_day, butterfly):
        # Get predicted vol for the next 5 days (single value)
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == entry_day]
        if pred_row.empty:
            return False
        pred_vol = pred_row.iloc[0]['predicted_vol']
        realized_vol = self.underlying_data.loc[self.underlying_data['timestamp'] == entry_day, 'realized_vol'].values[0]
        if pd.isna(pred_vol) or pd.isna(realized_vol):
            return False
        if pred_vol >= realized_vol:
            return False
        
        return self.check_liquidity_and_costs(entry_day, butterfly)
    
    def entry_rule_short(self, entry_day, butterfly):
        # Get predicted vol for the next 5 days (single value)
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == entry_day]
        if pred_row.empty:
            return False
        pred_vol = pred_row.iloc[0]['predicted_vol']
        realized_vol = self.underlying_data.loc[self.underlying_data['timestamp'] == entry_day, 'realized_vol'].values[0]
        if pd.isna(pred_vol) or pd.isna(realized_vol):
            return False
        # For short: only enter if predicted vol is HIGHER than realized vol
        if pred_vol <= realized_vol:
            return False

        return self.check_liquidity_and_costs(entry_day, butterfly)


    def select_options_for_day(self, day, option_type='call'):
        mask = (self.options_data['timestamp'] == day) & (self.options_data['dtm'] == 5) & (self.options_data['type'] == option_type)
        return self.options_data[mask]

    def construct_long_butterfly(self, options_day):
        """
        Construct a long butterfly: +1 lower strike, -2 middle strike, +1 higher strike.
        """
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

    def construct_short_butterfly(self, options_day):
        """
        Construct a short butterfly: -1 lower strike, +2 middle strike, -1 higher strike.
        """
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
            'short1': options_day[options_day['strike_price'] == k1].iloc[0],
            'long2': options_day[options_day['strike_price'] == k2].iloc[0],
            'short3': options_day[options_day['strike_price'] == k3].iloc[0]
        }
        return legs

    def should_exit_early(self, open_trade, current_day):
        """
        For long butterflies: exit if predicted vol increases significantly.
        For short butterflies: exit if predicted vol decreases significantly.
        """
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == current_day]
        if pred_row.empty:
            return False
        pred_vol_now = pred_row.iloc[0]['predicted_vol']
        pred_vol_entry = open_trade['pred_vol_entry']
        if pd.isna(pred_vol_now) or pd.isna(pred_vol_entry):
            return False

        # Check trade type
        trade_type = open_trade.get('trade_type', 'long')  # default to 'long' if not set

        if trade_type == 'long':
            # Exit if predicted vol increases by more than threshold
            if (pred_vol_now - pred_vol_entry) / pred_vol_entry > self.vol_exit_threshold:
                print(f"Early exit (LONG) on {current_day}: New 5d predicted vol {pred_vol_now} > entry vol {pred_vol_entry} by threshold")
                return True
        elif trade_type == 'short':
            # Exit if predicted vol decreases by more than threshold
            if (pred_vol_entry - pred_vol_now) / pred_vol_entry > self.vol_exit_threshold:
                print(f"Early exit (SHORT) on {current_day}: New 5d predicted vol {pred_vol_now} < entry vol {pred_vol_entry} by threshold")
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

    # def run(self, option_type='call', starting_cash=1000):
    #     results = []
    #     cash = starting_cash
    #     equity_curve = []
    #     open_trades = []

    #     # Combine all relevant dates (trading days and expiries)
    #     all_dates = sorted(set(self.underlying_data['timestamp']) | set(self.options_data['expiration_date']))

    #     for day in all_dates:
    #         # 1. Check for new trade opportunities (entry)
    #         total_max_loss = sum([trade['entry_price'] * trade['n_butterflies'] for trade in open_trades])
    #         margin_available = cash - total_max_loss
    #         max_invest_today = 0.2 * margin_available


    #         if day in self.underlying_data['timestamp']:
    #             expiry_candidates = self.options_data[
    #                 (self.options_data['timestamp'] == day) & (self.options_data['dtm'] == 5)
    #             ]['expiration_date'].unique()
    #             if len(expiry_candidates) == 0:
    #                 continue
    #             expiry = expiry_candidates[0]
    #             options_day = self.options_data[
    #                 (self.options_data['timestamp'] == day) &
    #                 (self.options_data['expiration_date'] == expiry) &
    #                 (self.options_data['type'] == option_type)
    #             ]
    #             if options_day.empty:
    #                 continue
    #             butterfly = self.construct_butterfly(options_day)
    #             if butterfly is None:
    #                 continue
    #             # Use only the prediction available on entry day
    #             pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == day]
    #             if pred_row.empty:
    #                 continue
    #             pred_vol_entry = pred_row.iloc[0]['predicted_vol']
    #             if not self.entry_rule(day, butterfly):
    #                 continue

    #             entry_price = butterfly['long1']['opt_close'] + butterfly['long3']['opt_close'] - 2 * butterfly['short2']['opt_close']
    #             total_legs = 4
    #             total_commission = self.commission * total_legs
    #             total_slippage = self.slippage * (abs(butterfly['long1']['opt_close']) +
    #                                             abs(butterfly['long3']['opt_close']) +
    #                                             2 * abs(butterfly['short2']['opt_close']))
    #             total_cost_per_butterfly = entry_price + total_commission + total_slippage

    #             # Liquidity constraints
    #             min_oi = min(
    #                 butterfly['long1'].get('open_interest', np.inf),
    #                 butterfly['long3'].get('open_interest', np.inf),
    #                 butterfly['short2'].get('open_interest', np.inf) // 2
    #             )
    #             min_vol = min(
    #                 butterfly['long1'].get('volume', np.inf),
    #                 butterfly['long3'].get('volume', np.inf),
    #                 butterfly['short2'].get('volume', np.inf) // 2
    #             )
    #             max_liquidity = int(min(min_oi, min_vol))

    #             # Limit number of butterflies by available margin, not just cash
    #             n_butterflies = min(int(max_invest_today // total_cost_per_butterfly), max_liquidity)
    #             if n_butterflies < 1:
    #                 continue

    #             total_entry_cost = total_cost_per_butterfly * n_butterflies

    #             cash -= total_entry_cost
    #             open_trades.append({
    #                 'entry_date': day,
    #                 'expiry': expiry,
    #                 'butterfly': butterfly,
    #                 'entry_price': entry_price,
    #                 'commission': total_commission,
    #                 'slippage': total_slippage,
    #                 'total_cost': total_entry_cost,
    #                 'n_butterflies': n_butterflies,
    #                 'pred_vol_entry': pred_vol_entry
    #             })


    #         # 2. Check for early exit or expiry for open trades
    #         to_remove = []
    #         for i, trade in enumerate(open_trades):
    #             # Early exit logic: check next 2 days' predictions
    #             alarm = False
    #             for offset in [1, 2]:
    #                 next_day = day + pd.Timedelta(days=offset)
    #                 pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == next_day]
    #                 if not pred_row.empty:
    #                     pred_vol_next = pred_row.iloc[0]['predicted_vol']
    #                     if (pred_vol_next - trade['pred_vol_entry']) / trade['pred_vol_entry'] > self.vol_exit_threshold:
    #                         alarm = True
    #                         print(f"Early exit on {day}: Next day {next_day} predicted vol {pred_vol_next} > entry vol {trade['pred_vol_entry']} by threshold")
    #                         break
    #             is_expiry = (trade['expiry'] == day)
    #             if alarm or is_expiry:
    #                 exit_prices = self.get_exit_prices(trade['butterfly'], day)
    #                 if exit_prices is None:
    #                     continue
    #                 exit_price = exit_prices['long1'] + exit_prices['long3'] - 2 * exit_prices['short2']
    #                 exit_slippage = self.slippage * (abs(exit_prices['long1']) +
    #                                                 abs(exit_prices['long3']) +
    #                                                 2 * abs(exit_prices['short2']))
    #                 total_exit_value = exit_price * trade['n_butterflies']
    #                 total_slippage_all = (trade['slippage'] + exit_slippage) * trade['n_butterflies']
    #                 total_commission_all = trade['commission'] * trade['n_butterflies']
    #                 pnl = (exit_price - trade['entry_price'] - trade['commission'] - trade['slippage'] - exit_slippage) * trade['n_butterflies']
    #                 cash += total_exit_value

    #                 results.append({
    #                     'entry_date': trade['entry_date'],
    #                     'exit_date': day,
    #                     'expiry': trade['expiry'],
    #                     'entry_price': trade['entry_price'],
    #                     'exit_price': exit_price,
    #                     'n_butterflies': trade['n_butterflies'],
    #                     'pnl': pnl,
    #                     'underlying_price': self.underlying_data.loc[self.underlying_data['timestamp'] == trade['entry_date'], 'close'].values[0],
    #                     'strikes': (
    #                         trade['butterfly']['long1']['strike_price'],
    #                         trade['butterfly']['short2']['strike_price'],
    #                         trade['butterfly']['long3']['strike_price']
    #                     ),
    #                     'option_prices_entry': (
    #                         trade['butterfly']['long1']['opt_close'],
    #                         trade['butterfly']['short2']['opt_close'],
    #                         trade['butterfly']['long3']['opt_close']
    #                     ),
    #                     'option_prices_exit': (
    #                         exit_prices['long1'],
    #                         exit_prices['short2'],
    #                         exit_prices['long3']
    #                     ),
    #                     'pred_vol_entry': trade['pred_vol_entry'],
    #                     'commission': total_commission_all,
    #                     'slippage': total_slippage_all,
    #                     'cash_after_trade': cash,
    #                     'early_exit': alarm
    #                 })
    #                 to_remove.append(i)
    #         for idx in sorted(to_remove, reverse=True):
    #             open_trades.pop(idx)

    #         # 3. Mark-to-market for equity
    #         mtm_value = 0
    #         for trade in open_trades:
    #             current_prices = self.get_exit_prices(trade['butterfly'], day)
    #             if current_prices is not None:
    #                 mtm_value += (current_prices['long1'] + current_prices['long3'] - 2 * current_prices['short2']) * trade['n_butterflies']
            


    #         equity_curve.append({
    #             'date': day,
    #             'cash': cash,
    #             'equity': cash + mtm_value,
    #             'margin_available': margin_available,
    #             'total_max_loss': total_max_loss
    #         })

    #     results_df = pd.DataFrame(results)
    #     equity_df = pd.DataFrame(equity_curve)

    #     # Add summary statistics
    #     if not results_df.empty:
    #         total_trades = len(results_df)
    #         total_profit = results_df['pnl'].sum()
    #         avg_profit = results_df['pnl'].mean()
    #         win_rate = (results_df['pnl'] > 0).mean()
    #         max_drawdown = (equity_df['equity'].cummax() - equity_df['equity']).max()
    #         ending_cash = cash
    #         ending_equity = equity_df['equity'].iloc[-1]
    #         stats = {
    #             'total_trades': total_trades,
    #             'total_profit': total_profit,
    #             'avg_profit': avg_profit,
    #             'win_rate': win_rate,
    #             'max_drawdown': max_drawdown,
    #             'ending_cash': ending_cash,
    #             'ending_equity': ending_equity
    #         }
    #         print("Backtest Statistics (1 Year):")
    #         for k, v in stats.items():
    #             print(f"{k}: {v}")
    #     else:
    #         print("No trades executed.")

    #     # Plot cash and equity curve with trade signals
    #     if not equity_df.empty:
    #         plt.figure(figsize=(12, 6))
    #         plt.plot(equity_df['date'], equity_df['cash'], label='Cash Balance', color='blue')
    #         plt.plot(equity_df['date'], equity_df['equity'], label='Total Equity', color='green')
    #         if not results_df.empty:
    #             plt.scatter(results_df['entry_date'], results_df['cash_after_trade'], color='red', marker='^', label='Trade Entry')
    #         plt.title('Account Cash & Equity Over Time')
    #         plt.xlabel('Date')
    #         plt.ylabel('Value ($)')
    #         plt.legend()
    #         plt.grid(True)
    #         plt.tight_layout()
    #         plt.show()
    #     else:
    #         print("No equity data to plot.")

    #     return results_df, equity_df



###################################################################################################
#--------------------------------------------------------------------------------
###################################################################################################


    def run(self, option_type='call', starting_cash=1000):
        results = []
        cash = starting_cash
        equity_curve = []
        open_trades = []

        # Combine all relevant dates (trading days and expiries)
        all_dates = sorted(set(self.underlying_data['timestamp']) | set(self.options_data['expiration_date']))

        for day in all_dates:
            # 1. Check for new trade opportunities (entry)
            total_max_loss = sum([trade['entry_price'] * trade['n_butterflies'] for trade in open_trades])
            margin_available = cash - total_max_loss
            max_invest_today = 0.2 * margin_available

            if day in self.underlying_data['timestamp']:
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

                # Try LONG butterfly
                butterfly_long = self.construct_long_butterfly(options_day)
                if butterfly_long is not None:
                    pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == day]
                    if not pred_row.empty:
                        pred_vol_entry = pred_row.iloc[0]['predicted_vol']
                        if self.entry_rule_long(day, butterfly_long):
                            entry_price = butterfly_long['long1']['opt_close'] + butterfly_long['long3']['opt_close'] - 2 * butterfly_long['short2']['opt_close']
                            total_legs = 4
                            total_commission = self.commission * total_legs
                            total_slippage = self.slippage * (abs(butterfly_long['long1']['opt_close']) +
                                                            abs(butterfly_long['long3']['opt_close']) +
                                                            2 * abs(butterfly_long['short2']['opt_close']))
                            total_cost_per_butterfly = entry_price + total_commission + total_slippage

                            min_oi = min(
                                butterfly_long['long1'].get('open_interest', np.inf),
                                butterfly_long['long3'].get('open_interest', np.inf),
                                butterfly_long['short2'].get('open_interest', np.inf) // 2
                            )
                            min_vol = min(
                                butterfly_long['long1'].get('volume', np.inf),
                                butterfly_long['long3'].get('volume', np.inf),
                                butterfly_long['short2'].get('volume', np.inf) // 2
                            )
                            max_liquidity = int(min(min_oi, min_vol))
                            n_butterflies = min(int(max_invest_today // total_cost_per_butterfly), max_liquidity)
                            if n_butterflies >= 1:
                                total_entry_cost = total_cost_per_butterfly * n_butterflies
                                cash -= total_entry_cost
                                open_trades.append({
                                    'entry_date': day,
                                    'expiry': expiry,
                                    'butterfly': butterfly_long,
                                    'entry_price': entry_price,
                                    'commission': total_commission,
                                    'slippage': total_slippage,
                                    'total_cost': total_entry_cost,
                                    'n_butterflies': n_butterflies,
                                    'pred_vol_entry': pred_vol_entry,
                                    'trade_type': 'long'
                                })

                # Try SHORT butterfly
                butterfly_short = self.construct_short_butterfly(options_day)
                if butterfly_short is not None:
                    pred_row = self.vol_predictions[self.vol_predictions['prediction_day'] == day]
                    if not pred_row.empty:
                        pred_vol_entry = pred_row.iloc[0]['predicted_vol']
                        if self.entry_rule_short(day, butterfly_short):
                            entry_price = -butterfly_short['short1']['opt_close'] + 2 * butterfly_short['long2']['opt_close'] - butterfly_short['short3']['opt_close']
                            total_legs = 4
                            total_commission = self.commission * total_legs
                            total_slippage = self.slippage * (abs(butterfly_short['short1']['opt_close']) +
                                                            abs(butterfly_short['short3']['opt_close']) +
                                                            2 * abs(butterfly_short['long2']['opt_close']))
                            total_cost_per_butterfly = entry_price + total_commission + total_slippage

                            min_oi = min(
                                butterfly_short['short1'].get('open_interest', np.inf),
                                butterfly_short['short3'].get('open_interest', np.inf),
                                butterfly_short['long2'].get('open_interest', np.inf) // 2
                            )
                            min_vol = min(
                                butterfly_short['short1'].get('volume', np.inf),
                                butterfly_short['short3'].get('volume', np.inf),
                                butterfly_short['long2'].get('volume', np.inf) // 2
                            )
                            max_liquidity = int(min(min_oi, min_vol))
                            n_butterflies = min(int(max_invest_today // abs(total_cost_per_butterfly)), max_liquidity)
                            if n_butterflies >= 1:
                                total_entry_cost = total_cost_per_butterfly * n_butterflies
                                cash -= total_entry_cost
                                open_trades.append({
                                    'entry_date': day,
                                    'expiry': expiry,
                                    'butterfly': butterfly_short,
                                    'entry_price': entry_price,
                                    'commission': total_commission,
                                    'slippage': total_slippage,
                                    'total_cost': total_entry_cost,
                                    'n_butterflies': n_butterflies,
                                    'pred_vol_entry': pred_vol_entry,
                                    'trade_type': 'short'
                                })

            # 2. Check for early exit or expiry for open trades
            to_remove = []
            for i, trade in enumerate(open_trades):
                alarm = self.should_exit_early(trade, day)
                is_expiry = (trade['expiry'] == day)
                if alarm or is_expiry:
                    exit_prices = self.get_exit_prices(trade['butterfly'], day)
                    if exit_prices is None:
                        continue
                    if trade['trade_type'] == 'long':
                        exit_price = exit_prices['long1'] + exit_prices['long3'] - 2 * exit_prices['short2']
                        exit_slippage = self.slippage * (abs(exit_prices['long1']) +
                                                        abs(exit_prices['long3']) +
                                                        2 * abs(exit_prices['short2']))
                    else:  # short butterfly
                        exit_price = -exit_prices['short1'] + 2 * exit_prices['long2'] - exit_prices['short3']
                        exit_slippage = self.slippage * (abs(exit_prices['short1']) +
                                                        abs(exit_prices['short3']) +
                                                        2 * abs(exit_prices['long2']))
                    total_exit_value = exit_price * trade['n_butterflies']
                    total_slippage_all = (trade['slippage'] + exit_slippage) * trade['n_butterflies']
                    total_commission_all = trade['commission'] * trade['n_butterflies']
                    pnl = (exit_price - trade['entry_price'] - trade['commission'] - trade['slippage'] - exit_slippage) * trade['n_butterflies']
                    cash += total_exit_value

                    # Save trade details
                    if trade['trade_type'] == 'long':
                        strikes = (
                            trade['butterfly']['long1']['strike_price'],
                            trade['butterfly']['short2']['strike_price'],
                            trade['butterfly']['long3']['strike_price']
                        )
                        option_prices_entry = (
                            trade['butterfly']['long1']['opt_close'],
                            trade['butterfly']['short2']['opt_close'],
                            trade['butterfly']['long3']['opt_close']
                        )
                        option_prices_exit = (
                            exit_prices['long1'],
                            exit_prices['short2'],
                            exit_prices['long3']
                        )
                    else:
                        strikes = (
                            trade['butterfly']['short1']['strike_price'],
                            trade['butterfly']['long2']['strike_price'],
                            trade['butterfly']['short3']['strike_price']
                        )
                        option_prices_entry = (
                            trade['butterfly']['short1']['opt_close'],
                            trade['butterfly']['long2']['opt_close'],
                            trade['butterfly']['short3']['opt_close']
                        )
                        option_prices_exit = (
                            exit_prices['short1'],
                            exit_prices['long2'],
                            exit_prices['short3']
                        )

                    results.append({
                        'entry_date': trade['entry_date'],
                        'exit_date': day,
                        'expiry': trade['expiry'],
                        'entry_price': trade['entry_price'],
                        'exit_price': exit_price,
                        'n_butterflies': trade['n_butterflies'],
                        'pnl': pnl, # profit and loss
                        'underlying_price': self.underlying_data.loc[self.underlying_data['timestamp'] == trade['entry_date'], 'close'].values[0],
                        'strikes': strikes,
                        'option_prices_entry': option_prices_entry,
                        'option_prices_exit': option_prices_exit,
                        'pred_vol_entry': trade['pred_vol_entry'],
                        'commission': total_commission_all,
                        'slippage': total_slippage_all,
                        'cash_after_trade': cash,
                        'early_exit': alarm,
                        'trade_type': trade['trade_type']
                    })
                    to_remove.append(i)
            for idx in sorted(to_remove, reverse=True):
                open_trades.pop(idx)

            # 3. Mark-to-market for equity
            mtm_value = 0
            for trade in open_trades:
                current_prices = self.get_exit_prices(trade['butterfly'], day)
                if current_prices is not None:
                    if trade['trade_type'] == 'long':
                        mtm_value += (current_prices['long1'] + current_prices['long3'] - 2 * current_prices['short2']) * trade['n_butterflies']
                    else:
                        mtm_value += (-current_prices['short1'] + 2 * current_prices['long2'] - current_prices['short3']) * trade['n_butterflies']

            equity_curve.append({
                'date': day,
                'cash': cash,
                'equity': cash + mtm_value,
                'margin_available': margin_available,
                'total_max_loss': total_max_loss
            })

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