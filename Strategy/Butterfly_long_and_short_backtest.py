import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Butterfly long and short strategy backtest

class ButterflyBacktest_long_short:  
    def __init__(self, options_data, underlying_data, vol_predictions, spread_pct=0.01, commission=1.0, slippage=0.0, min_open_interest=10, min_volume=1, vol_exit_threshold=0.5, rolling_window=90):
        self.options_data = options_data # DataFrame with columns: timestamp, expiration_date, strike_price, type, opt_close, dtm, SPY_close_at_current_day, volume, open_interest
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
        underlying_row = self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day]
        if underlying_row.empty or underlying_row['close'].values[0] <= 0:
            return False
        if 'volume' in underlying_row:
            volume_val = pd.to_numeric(underlying_row['volume'].values[0], errors='coerce')
            if np.isnan(volume_val) or volume_val < self.min_volume:
                print(f"Skipping entry on {entry_day}: Underlying volume {volume_val} < {self.min_volume}")
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
        # Detect butterfly type by keys
        if set(butterfly.keys()) == {'long1', 'short2', 'long3'}:
            # Long butterfly
            k1 = butterfly['long1']['strike_price']
            k2 = butterfly['short2']['strike_price']
            k3 = butterfly['long3']['strike_price']
            entry_price = butterfly['long1']['opt_close'] + butterfly['long3']['opt_close'] - 2 * butterfly['short2']['opt_close']
        elif set(butterfly.keys()) == {'short1', 'long2', 'short3'}:
            # Short butterfly
            k1 = butterfly['short1']['strike_price']
            k2 = butterfly['long2']['strike_price']
            k3 = butterfly['short3']['strike_price']
            entry_price = -butterfly['short1']['opt_close'] + 2 * butterfly['long2']['opt_close'] - butterfly['short3']['opt_close']
        else:
            print(f"Unknown butterfly structure: {butterfly.keys()}")
            return False

        spread_width = k3 - k2  # symmetric butterfly
        max_profit = spread_width  # max payoff if ATM at expiry

        if abs(entry_price) > max_profit: # possible to include threshold *0.7 eg. 
            print(f"Skipping entry on {entry_day}: Entry price {entry_price} higher than max profit {max_profit}")
            return False

        return True

    def entry_rule_long(self, entry_day, butterfly):
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == entry_day]
        print(f"vol_predictions rows for {entry_day}: {len(pred_row)}")
        print(f"underlying_data rows for {entry_day}: {len(self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day])}")

        if pred_row.empty:
            print(f"No prediction for {entry_day}")
            return False
        pred_vol = pred_row.iloc[0]['predicted_vol']
        realized_vol_row = self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day]
        if realized_vol_row.empty:
            print(f"No realized vol for {entry_day}")
            return False
        realized_vol = realized_vol_row['realized_vol'].values[0]
        print(f"ENTRY CHECK: day={entry_day}, pred_vol={pred_vol}, realized_vol={realized_vol}")

        if pd.isna(pred_vol) or pd.isna(realized_vol):
            print(f"NaN vol for {entry_day}")
            return False
        if pred_vol >= realized_vol:
            print(f"Predicted vol {pred_vol} >= realized vol {realized_vol} on {entry_day} (no LONG entry)")
            return False
        
        print(f"Predicted vol {pred_vol} < realized vol {realized_vol} on {entry_day} (LONG entry allowed)")

        return self.check_liquidity_and_costs(entry_day, butterfly)
    
    def entry_rule_short(self, entry_day, butterfly):
        # Get predicted vol for the next 5 days (single value)
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == entry_day]
        print(f"vol_predictions rows for {entry_day}: {len(pred_row)}")
        print(f"underlying_data rows for {entry_day}: {len(self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day])}")

        if pred_row.empty:
            print(f"No prediction for {entry_day}")
            return False
        pred_vol = pred_row.iloc[0]['predicted_vol']
        realized_vol_row = self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day]
        if realized_vol_row.empty:
            print(f"No realized vol for {entry_day}")
            return False
        realized_vol = realized_vol_row['realized_vol'].values[0]
        print(f"ENTRY CHECK: day={entry_day}, pred_vol={pred_vol}, realized_vol={realized_vol}")

        if pd.isna(pred_vol) or pd.isna(realized_vol):
            print(f"NaN vol for {entry_day}")
            return False
        # For short: only enter if predicted vol is HIGHER than realized vol
        if pred_vol <= realized_vol:
            print(f"Predicted vol {pred_vol} <= realized vol {realized_vol} on {entry_day} (no SHORT entry)")
            return False
        
        print(f"Predicted vol {pred_vol} > realized vol {realized_vol} on {entry_day} (SHORT entry allowed)")


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
        spot = options_day['SPY_close_at_current_day'].iloc[0]
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

        print(f"Constructed LONG butterfly on {options_day['timestamp'].iloc[0]}: ")
        return legs

    def construct_short_butterfly(self, options_day):
        """
        Construct a short butterfly: -1 lower strike, +2 middle strike, -1 higher strike.
        """
        strikes = np.sort(options_day['strike_price'].unique())
        if len(strikes) < 3:
            return None
        spot = options_day['SPY_close_at_current_day'].iloc[0]
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

        print(f"Constructed SHORT butterfly on {options_day['timestamp'].iloc[0]}: ")
        return legs

    def get_exit_prices(self, legs, exit_day):
        exit_prices = {}
        for leg, row in legs.items():
            print(f"\n[DEBUG] Looking for {leg}:")
            print(f"  expiry={row['expiration_date']} ({type(row['expiration_date'])})")
            print(f"  strike={row['strike_price']} ({type(row['strike_price'])})")
            print(f"  type={row['type']} ({type(row['type'])})")
            print(f"  exit_day={exit_day} ({type(exit_day)})")
            # If exiting on expiry, use close_price_on_expiry if available
            if exit_day == row['expiration_date'].date():
                mask = (
                    (self.options_data['expiration_date'].dt.date == row['expiration_date'].date()) &
                    (self.options_data['strike_price'] == row['strike_price']) &
                    (self.options_data['type'] == row['type']) &
                    (self.options_data['timestamp'].dt.date == exit_day)
                )
                match = self.options_data[mask]
                print(f"  Found {len(match)} rows for {leg} on {exit_day}")
                if not match.empty:
                    print(match[['timestamp', 'expiration_date', 'strike_price', 'type', 'close_price_on_expiry', 'opt_close']])
                if not match.empty and not pd.isna(match.iloc[0]['close_price_on_expiry']):
                    exit_prices[leg] = match.iloc[0]['close_price_on_expiry']
                elif not match.empty:
                    exit_prices[leg] = match.iloc[0]['opt_close']
                else:
                    print(f"No exit price for {leg} at expiry {exit_day}")
                    return None
            else:
                # Not expiry: use opt_close
                mask = (
                    (self.options_data['expiration_date'].dt.date == row['expiration_date'].date()) &
                    (self.options_data['strike_price'] == row['strike_price']) &
                    (self.options_data['type'] == row['type']) &
                    (self.options_data['timestamp'].dt.date == exit_day)
                )
                match = self.options_data[mask]
                print(f"  Found {len(match)} rows for {leg} on {exit_day}")
                if not match.empty:
                    print(match[['timestamp', 'expiration_date', 'strike_price', 'type', 'close_price_on_expiry', 'opt_close']])
                    exit_prices[leg] = match.iloc[0]['opt_close']
                else:
                    print(f"No exit price for {leg} on {exit_day}")
                    return None
        return exit_prices

    def should_exit_early(self, open_trade, current_day):
        """
        For long butterflies: exit if predicted vol increases significantly.
        For short butterflies: exit if predicted vol decreases significantly.
        """
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == current_day]
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
    
    # def get_exit_prices(self, legs, expiry):
    #     exit_prices = {}
    #     for leg, row in legs.items():
    #         match = self.options_data[
    #             (self.options_data['expiration_date'] == expiry) &
    #             (self.options_data['strike_price'] == row['strike_price']) &
    #             (self.options_data['type'] == row['type']) &
    #             (self.options_data['timestamp'] == expiry)
    #         ]
    #         if match.empty:
    #             return None
    #         exit_prices[leg] = match.iloc[0]['opt_close']
    #     return exit_prices


###################################################################################################
#--------------------------------------------------------------------------------
###################################################################################################


    def run(self, option_type='call', starting_cash=1000):
        results = []
        cash = starting_cash
        equity_curve = []
        open_trades = []

        # Combine all relevant dates (trading days and expiries)
        #all_dates = sorted(set(self.underlying_data['timestamp']) | set(self.options_data['expiration_date']))
        
        
        #####################
        self.underlying_data['date'] = self.underlying_data['timestamp'].dt.date
        all_dates = sorted(set(self.underlying_data['date']) | set(self.options_data['timestamp'].dt.date))
        underlying_ts_set = set(self.underlying_data['date'])
    


        ############
        print("Total dates to process:", len(all_dates))

        for day in all_dates:
            #print(f"Processing day: {day}")
            # 1. Check for new trade opportunities (entry)
            total_max_loss = sum([trade['entry_price'] * trade['n_butterflies'] for trade in open_trades])
            margin_available = cash - total_max_loss
            max_invest_today = 0.2 * margin_available

            print("Sample underlying_data['timestamp']:", self.underlying_data['timestamp'].head(10).tolist())

            #if day in self.underlying_data['timestamp']:
            if day in underlying_ts_set:
                print(f"Processing day: {day}")
                expiry_candidates = self.options_data[
                    (self.options_data['timestamp'].dt.date == day) & (self.options_data['dtm'] == 5)
                ]['expiration_date'].unique()

                print(f"Found expiry candidates: {len(expiry_candidates)} for day {day}")
                if len(expiry_candidates) == 0:
                    continue
                expiry = expiry_candidates[0]
                options_day = self.options_data[
                    (self.options_data['timestamp'].dt.date == day) &
                    (self.options_data['expiration_date'] == expiry) &
                    (self.options_data['type'] == option_type)
                ]
                print(f"Options for {day}: {len(options_day)} strikes: {options_day['strike_price'].unique()}")
                if options_day.empty:
                    print(f"No options data for {day} with expiry {expiry}")
                    continue

                # Try LONG butterfly
                butterfly_long = self.construct_long_butterfly(options_day)
                if butterfly_long is not None:
                    pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == day]
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
                    pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == day]
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
                if butterfly_long is None:
                    print(f"No valid LONG butterfly for {day}")
                if butterfly_short is None:
                    print(f"No valid SHORT butterfly for {day}")
            else: 
                print(f"Skipping day {day}: No underlying data available")
               

            # 2. Check for early exit or expiry for open trades
            to_remove = []
            for i, trade in enumerate(open_trades):
                alarm = self.should_exit_early(trade, day)
                is_expiry = (trade['expiry'].date() == day)
                if not (alarm or is_expiry):
                    continue  # Only try to exit if alarm or expiry!
                exit_prices = self.get_exit_prices(trade['butterfly'], day)
                if exit_prices is None:
                    print(f"Could not exit trade on {day} (no price for all legs)")
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
                    'underlying_price': self.underlying_data.loc[self.underlying_data['timestamp'].dt.date == trade['entry_date'], 'close'].values[0],
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
                print(f"Removed trade on {day} for {'LONG' if trade['trade_type'] == 'long' else 'SHORT'} butterfly")

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
######################################################################################
#-----------------------------------VISUALIZATION---------------------------------------------
#######################################################################################
        # Plot cash and equity curve with trade signals
        # Enhanced visualization
        if not equity_df.empty:
            fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

            # 1. Equity and cash
            axs[0].plot(equity_df['date'], equity_df['cash'], label='Cash', color='blue', alpha=0.7)
            axs[0].plot(equity_df['date'], equity_df['equity'], label='Equity', color='green', alpha=0.7)
            axs[0].set_ylabel('Account Value ($)')
            axs[0].legend(loc='upper left')
            axs[0].set_title('Account Equity & Cash with Trade Markers')

            # Mark trade entries and exits
            if not results_df.empty:
                long_trades = results_df[results_df['trade_type'] == 'long']
                short_trades = results_df[results_df['trade_type'] == 'short']

                # Entry markers
                axs[0].scatter(long_trades['entry_date'], 
                            equity_df.set_index('date').loc[long_trades['entry_date'], 'equity'],
                            marker='^', color='lime', label='Long Entry', zorder=5)
                axs[0].scatter(short_trades['entry_date'], 
                            equity_df.set_index('date').loc[short_trades['entry_date'], 'equity'],
                            marker='^', color='red', label='Short Entry', zorder=5)
                # Exit markers
                axs[0].scatter(long_trades['exit_date'], 
                            equity_df.set_index('date').reindex(long_trades['exit_date'])['equity'],
                            marker='v', color='green', label='Long Exit', zorder=5)
                axs[0].scatter(short_trades['exit_date'], 
                            equity_df.set_index('date').reindex(short_trades['exit_date'])['equity'],
                            marker='v', color='darkred', label='Short Exit', zorder=5)

            axs[0].grid(True)
            axs[0].legend()

            # 2. Underlying price
            axs[1].plot(self.underlying_data['timestamp'], self.underlying_data['close'], label='Underlying Price', color='black')
            axs[1].set_ylabel('Underlying Price')
            axs[1].legend(loc='upper left')
            axs[1].grid(True)

            # 3. Volatility
            axs[2].plot(self.underlying_data['timestamp'], self.underlying_data['realized_vol'], label='Realized Vol', color='blue')
            axs[2].plot(self.vol_predictions['prediction_day'], self.vol_predictions['predicted_vol'], label='Predicted Vol', color='orange')
            axs[2].set_ylabel('Volatility')
            axs[2].legend(loc='upper left')
            axs[2].grid(True)

            # Format x-axis as dates
            axs[2].xaxis.set_major_locator(mdates.MonthLocator())
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.xlabel('Date')
            plt.tight_layout()
            plt.show()
        else:
            print("No equity data to plot.")
        
        # Plot equity, underlying, trades, and open trades over time

        # --- Prepare open trades over time ---
        if not results_df.empty and not equity_df.empty:
            # Build a time series of open trades
            all_dates = pd.to_datetime(equity_df['date'])
            open_trades_series = pd.Series(0, index=all_dates)
            for _, row in results_df.iterrows():
                entry = pd.to_datetime(row['entry_date'])
                exit_ = pd.to_datetime(row['exit_date'])
                # Increment for each day trade is open
                mask = (open_trades_series.index >= entry) & (open_trades_series.index <= exit_)
                open_trades_series[mask] += 1

            fig, axs = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

            # 1. Equity curve and trade markers
            axs[0].plot(equity_df['date'], equity_df['equity'], label='Equity', color='green')
            axs[0].plot(equity_df['date'], equity_df['cash'], label='Cash', color='blue', alpha=0.5)
            axs[0].set_ylabel('Account Value ($)')
            axs[0].set_title('Equity Curve with Trade Markers')

            # Mark trade entries/exits
            long_trades = results_df[results_df['trade_type'] == 'long']
            short_trades = results_df[results_df['trade_type'] == 'short']

            axs[0].scatter(long_trades['entry_date'], 
                        equity_df.set_index('date').reindex(long_trades['entry_date'])['equity'],
                        marker='^', color='lime', label='Long Entry', zorder=5)
            axs[0].scatter(short_trades['entry_date'], 
                        equity_df.set_index('date').reindex(short_trades['entry_date'])['equity'],
                        marker='^', color='red', label='Short Entry', zorder=5)
            axs[0].scatter(long_trades['exit_date'], 
                        equity_df.set_index('date').reindex(long_trades['exit_date'])['equity'],
                        marker='v', color='green', label='Long Exit', zorder=5)
            axs[0].scatter(short_trades['exit_date'], 
                        equity_df.set_index('date').reindex(short_trades['exit_date'])['equity'],
                        marker='v', color='darkred', label='Short Exit', zorder=5)
            axs[0].legend()
            axs[0].grid(True)

            # 2. Underlying price
            axs[1].plot(self.underlying_data['timestamp'], self.underlying_data['close'], label='Underlying Price', color='black')
            axs[1].set_ylabel('Underlying Price')
            axs[1].legend()
            axs[1].grid(True)

            # 3. Open trades over time
            axs[2].bar(open_trades_series.index, open_trades_series.values, width=1, color='purple', alpha=0.6)
            axs[2].set_ylabel('Open Trades')
            axs[2].set_xlabel('Date')
            axs[2].set_title('Number of Open Trades Over Time')
            axs[2].grid(True)

            # Format x-axis as dates
            axs[2].xaxis.set_major_locator(mdates.MonthLocator())
            axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            plt.tight_layout()
            plt.show()
        else:
            print("No results or equity data to plot.")
        
        plt.plot(self.underlying_data['timestamp'], self.underlying_data['realized_vol'], label='Realized Vol')
        plt.plot(self.vol_predictions['prediction_day'], self.vol_predictions['predicted_vol'], label='Predicted Vol')
        plt.legend()
        plt.show()

        return results_df, equity_df