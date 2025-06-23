import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Butterfly long and short strategy backtest

class ButterflyBacktest_long_short:  
    def __init__(
        self, options_data, underlying_data, vol_predictions, 
        spread_pct=0.01, commission=1.0, slippage=0.0, min_open_interest=10, min_volume=1, 
        vol_exit_threshold=0.5, rolling_window=90, 
        pct_below=0.98, pct_above=1.02 
    ):
        self.options_data = options_data
        self.underlying_data = underlying_data
        self.vol_predictions = vol_predictions
        self.spread_pct = spread_pct
        self.commission = commission
        self.slippage = slippage
        self.min_open_interest = min_open_interest
        self.min_volume = min_volume
        self.vol_exit_threshold = vol_exit_threshold
        self.rolling_window = rolling_window
        self.pct_below = pct_below  # e.g. 0.98 means 2% below realized vol for long
        self.pct_above = pct_above  # e.g. 1.02 means 2% above realized vol for short

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
                #print(f"Skipping entry on {entry_day}: Underlying volume {volume_val} < {self.min_volume}")
                return False

        # Option liquidity filter
        for leg in butterfly.values():
            if ('open_interest' in leg and (pd.isna(leg['open_interest']) or leg['open_interest'] < self.min_open_interest)):
                #print(f"Skipping entry on {entry_day}: Open interest {leg['open_interest']} < {self.min_open_interest}")
                return False
            if ('opt_close' in leg and (pd.isna(leg['opt_close']) or leg['opt_close'] <= 0)):
                #print(f"Skipping entry on {entry_day}: Option price {leg['opt_close']} <= 0")
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
            #print(f"Skipping entry on {entry_day}: Entry price {entry_price} higher than max profit {max_profit}")
            return False

        return True

    def entry_rule_long(self, entry_day, butterfly):
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == entry_day]
        #print(f"vol_predictions rows for {entry_day}: {len(pred_row)}")
        #print(f"underlying_data rows for {entry_day}: {len(self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day])}")

        if pred_row.empty:
            #print(f"No prediction for {entry_day}")
            return False
        pred_vol = pred_row.iloc[0]['predicted_vol']
        realized_vol_row = self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day]
        if realized_vol_row.empty:
            #print(f"No realized vol for {entry_day}")
            return False
        realized_vol = realized_vol_row['realized_vol'].values[0]
        #print(f"ENTRY CHECK: day={entry_day}, pred_vol={pred_vol}, realized_vol={realized_vol}")

        if pd.isna(pred_vol) or pd.isna(realized_vol):
            #print(f"NaN vol for {entry_day}")
            return False
        # Use threshold for long entry
        if pred_vol >= self.pct_below * realized_vol:
            #print(f"Predicted vol {pred_vol} >= {self.pct_below} * realized vol {realized_vol} on {entry_day} (no LONG entry)")
            return False

        #print(f"Predicted vol {pred_vol} < {self.pct_below} * realized vol {realized_vol} on {entry_day} (LONG entry allowed)")
        return self.check_liquidity_and_costs(entry_day, butterfly)

    
    def entry_rule_short(self, entry_day, butterfly):
        pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == entry_day]
        #print(f"vol_predictions rows for {entry_day}: {len(pred_row)}")
        #print(f"underlying_data rows for {entry_day}: {len(self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day])}")

        if pred_row.empty:
            #print(f"No prediction for {entry_day}")
            return False
        pred_vol = pred_row.iloc[0]['predicted_vol']
        realized_vol_row = self.underlying_data[self.underlying_data['timestamp'].dt.date == entry_day]
        if realized_vol_row.empty:
            #print(f"No realized vol for {entry_day}")
            return False
        realized_vol = realized_vol_row['realized_vol'].values[0]
        #print(f"ENTRY CHECK: day={entry_day}, pred_vol={pred_vol}, realized_vol={realized_vol}")

        if pd.isna(pred_vol) or pd.isna(realized_vol):
            #print(f"NaN vol for {entry_day}")
            return False
        # Use threshold for short entry
        if pred_vol <= self.pct_above * realized_vol:
            #print(f"Predicted vol {pred_vol} <= {self.pct_above} * realized vol {realized_vol} on {entry_day} (no SHORT entry)")
            return False

        #print(f"Predicted vol {pred_vol} > {self.pct_above} * realized vol {realized_vol} on {entry_day} (SHORT entry allowed)")
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
            mask = (
                (self.options_data['expiration_date'].dt.date == row['expiration_date'].date()) &
                (self.options_data['strike_price'] == row['strike_price']) &
                (self.options_data['type'] == row['type']) &
                (self.options_data['timestamp'].dt.date == exit_day)
            )
            match = self.options_data[mask]

            # Fallback: Try to use the last known price before exit_day if no match
            if match.empty:
                fallback_mask = (
                    (self.options_data['expiration_date'].dt.date == row['expiration_date'].date()) &
                    (self.options_data['strike_price'] == row['strike_price']) &
                    (self.options_data['type'] == row['type']) &
                    (self.options_data['timestamp'].dt.date < exit_day)
                )
                fallback_matches = self.options_data[fallback_mask]
                if not fallback_matches.empty:
                    fallback_match = fallback_matches.sort_values('timestamp').iloc[-1]
                    exit_prices[leg] = fallback_match['opt_close']
                    continue
                else:
                    print(f"[WARN] No fallback match for leg {leg} on {exit_day} (strike={row['strike_price']}, type={row['type']})")
                    exit_prices[leg] = np.nan
                    continue

            if exit_day == row['expiration_date'].date():
                if not pd.isna(match.iloc[0]['close_price_on_expiry']):
                    exit_prices[leg] = match.iloc[0]['close_price_on_expiry']
                else:
                    exit_prices[leg] = match.iloc[0]['opt_close']
            else:
                exit_prices[leg] = match.iloc[0]['opt_close']

        return exit_prices

    # def get_exit_prices(self, legs, exit_day):
    #     exit_prices = {}
    #     for leg, row in legs.items():
    #         # print(f"\n[DEBUG] Looking for {leg}:")
    #         # print(f"  expiry={row['expiration_date']} ({type(row['expiration_date'])})")
    #         # print(f"  strike={row['strike_price']} ({type(row['strike_price'])})")
    #         # print(f"  type={row['type']} ({type(row['type'])})")
    #         # print(f"  exit_day={exit_day} ({type(exit_day)})")
    #         # If exiting on expiry, use close_price_on_expiry if available
    #         if exit_day == row['expiration_date'].date():
    #             mask = (
    #                 (self.options_data['expiration_date'].dt.date == row['expiration_date'].date()) &
    #                 (self.options_data['strike_price'] == row['strike_price']) &
    #                 (self.options_data['type'] == row['type']) &
    #                 (self.options_data['timestamp'].dt.date == exit_day)
    #             )
    #             match = self.options_data[mask]
    #             #print(f"  Found {len(match)} rows for {leg} on {exit_day}")
    #             if not match.empty:
    #                 print(match[['timestamp', 'expiration_date', 'strike_price', 'type', 'close_price_on_expiry', 'opt_close']])

    #             if not match.empty and not pd.isna(match.iloc[0]['close_price_on_expiry']):
    #                 exit_prices[leg] = match.iloc[0]['close_price_on_expiry']
    #             elif not match.empty:
    #                 exit_prices[leg] = match.iloc[0]['opt_close']
    #             else:
    #                 print(f"No exit price for {leg} at expiry {exit_day}")
    #                 return None
    #         else:
    #             # Not expiry: use opt_close
    #             mask = (
    #                 (self.options_data['expiration_date'].dt.date == row['expiration_date'].date()) &
    #                 (self.options_data['strike_price'] == row['strike_price']) &
    #                 (self.options_data['type'] == row['type']) &
    #                 (self.options_data['timestamp'].dt.date == exit_day)
    #             )
    #             match = self.options_data[mask]
    #             #print(f"  Found {len(match)} rows for {leg} on {exit_day}")
    #             if not match.empty:
    #                 #print(match[['timestamp', 'expiration_date', 'strike_price', 'type', 'close_price_on_expiry', 'opt_close']])
    #                 exit_prices[leg] = match.iloc[0]['opt_close']
    #             else:
    #                 print(f"No exit price for {leg} on {exit_day}")
    #                 return None
    #     return exit_prices

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


# Full plotting logic
    def plot_results(self, results_df, equity_df):  
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd

        if not equity_df.empty and not results_df.empty:
            if 'date' in equity_df.columns:
                all_dates = pd.to_datetime(equity_df['date'])
            else:
                all_dates = pd.to_datetime(equity_df.index)
            all_dates = pd.DatetimeIndex(all_dates)
            if all_dates.tz is None:
                all_dates = all_dates.tz_localize('UTC')
            else:
                all_dates = all_dates.tz_convert('UTC')

            first_trade_date = pd.to_datetime(results_df['entry_date']).min()
            if first_trade_date.tzinfo is None:
                first_trade_date = first_trade_date.tz_localize('UTC')
            else:
                first_trade_date = first_trade_date.tz_convert('UTC')

            all_dates = all_dates[all_dates >= first_trade_date]

            equity_df['date'] = pd.to_datetime(equity_df['date'])
            if equity_df['date'].dt.tz is None:
                equity_df['date'] = equity_df['date'].dt.tz_localize('UTC')
            else:
                equity_df['date'] = equity_df['date'].dt.tz_convert('UTC')

            equity_df_plot = equity_df[equity_df['date'].isin(all_dates)].copy()

            entries = pd.to_datetime(results_df['entry_date'])
            exits = pd.to_datetime(results_df['exit_date'])
            if entries.dt.tz is None:
                entries = entries.dt.tz_localize('UTC')
            else:
                entries = entries.dt.tz_convert('UTC')
            if exits.dt.tz is None:
                exits = exits.dt.tz_localize('UTC')
            else:
                exits = exits.dt.tz_convert('UTC')

            open_trades_series = pd.Series(0, index=all_dates)
            for entry, exit_ in zip(entries, exits):
                mask = (open_trades_series.index >= entry) & (open_trades_series.index < exit_)
                open_trades_series[mask] += 1

            fig, axs = plt.subplots(4, 1, figsize=(16, 16), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1, 1]})

            axs[0].plot(equity_df_plot['date'], equity_df_plot['cash'], label='Cash', color='blue', alpha=0.7)
            axs[0].plot(equity_df_plot['date'], equity_df_plot['equity'], label='Equity', color='green', alpha=0.7)
            axs[0].set_ylabel('Account Value ($)')
            axs[0].set_title('Account Equity & Cash with Trade Markers')
            axs[0].grid(True)
            axs[0].legend(loc='upper left')

            long_trades = results_df[results_df['trade_type'] == 'long'].copy()
            short_trades = results_df[results_df['trade_type'] == 'short'].copy()

            equity_df_plot = equity_df_plot.sort_values('date')
            long_trades = long_trades.sort_values('entry_date')
            short_trades = short_trades.sort_values('entry_date')

            # Ensure all datetime columns are proper timezone-aware datetime64[ns, UTC]
            equity_df_plot['date'] = pd.to_datetime(equity_df_plot['date'])
            long_trades['entry_date'] = pd.to_datetime(long_trades['entry_date'])
            short_trades['entry_date'] = pd.to_datetime(short_trades['entry_date'])
            long_trades['exit_date'] = pd.to_datetime(long_trades['exit_date'])
            short_trades['exit_date'] = pd.to_datetime(short_trades['exit_date'])

            for col in ['entry_date', 'exit_date']:
                for df in [long_trades, short_trades]:
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize('UTC')
                    else:
                        df[col] = df[col].dt.tz_convert('UTC')

            long_entry_equity = pd.merge_asof(long_trades[['entry_date']], equity_df_plot[['date', 'equity']], left_on='entry_date', right_on='date', direction='backward')['equity']
            short_entry_equity = pd.merge_asof(short_trades[['entry_date']], equity_df_plot[['date', 'equity']], left_on='entry_date', right_on='date', direction='backward')['equity']
            long_exit_equity = pd.merge_asof(long_trades[['exit_date']], equity_df_plot[['date', 'equity']], left_on='exit_date', right_on='date', direction='backward')['equity']
            short_exit_equity = pd.merge_asof(short_trades[['exit_date']], equity_df_plot[['date', 'equity']], left_on='exit_date', right_on='date', direction='backward')['equity']

            axs[0].scatter(long_trades['entry_date'], long_entry_equity, marker='^', color='lime', label='Long Entry', zorder=5)
            axs[0].scatter(short_trades['entry_date'], short_entry_equity, marker='^', color='red', label='Short Entry', zorder=5)
            axs[0].scatter(long_trades['exit_date'], long_exit_equity, marker='v', color='green', label='Long Exit', zorder=5)
            axs[0].scatter(short_trades['exit_date'], short_exit_equity, marker='v', color='darkred', label='Short Exit', zorder=5)
            axs[0].legend(loc='upper left', ncol=2)

            underlying_filtered = self.underlying_data[self.underlying_data['timestamp'] >= first_trade_date]
            axs[1].plot(underlying_filtered['timestamp'], underlying_filtered['close'], label='Underlying Price', color='black')
            axs[1].set_ylabel('Underlying Price')
            axs[1].set_title('Underlying Price')
            axs[1].legend(loc='upper left')
            axs[1].grid(True)

            vol_realized_filtered = self.underlying_data[self.underlying_data['timestamp'] >= first_trade_date]
            vol_pred_filtered = self.vol_predictions[self.vol_predictions['prediction_day'] >= first_trade_date]
            axs[2].plot(vol_realized_filtered['timestamp'], vol_realized_filtered['realized_vol'], label='Realized Vol', color='blue')
            axs[2].plot(vol_pred_filtered['prediction_day'], vol_pred_filtered['predicted_vol'], label='Predicted Vol', color='orange')
            axs[2].set_ylabel('Volatility')
            axs[2].set_title('Volatility (Realized & Predicted)')
            axs[2].legend(loc='upper left')
            axs[2].grid(True)

            axs[3].step(open_trades_series.index, open_trades_series.values, where='post', color='purple', alpha=0.8, label='Open Trades')
            axs[3].fill_between(open_trades_series.index, open_trades_series.values, step='post', color='purple', alpha=0.2)
            axs[3].set_ylabel('Open Trades')
            axs[3].set_xlabel('Date')
            axs[3].set_title('Number of Open Trades Over Time')
            axs[3].grid(True)
            axs[3].legend(loc='upper left')

            axs[3].xaxis.set_major_locator(mdates.MonthLocator())
            axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print("No equity data to plot.")


    
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


    def run(self, option_type='call', starting_cash=1000, risk_free_rate=0.045):
        results = []
        cash = starting_cash
        equity_curve = []
        open_trades = []

        self.underlying_data['date'] = self.underlying_data['timestamp'].dt.date
        # Only consider dates starting 6 days before the first options_data date
        first_option_date = self.options_data['timestamp'].dt.date.min()
        min_date = first_option_date - pd.Timedelta(days=6)
        all_dates = sorted(
            d for d in (set(self.underlying_data['date']) | set(self.options_data['timestamp'].dt.date))
            if d >= min_date
        )
        underlying_ts_set = set(self.underlying_data['date'])

        # --- Only use business days for trading ---
        # Get all business days in the date range
        business_days = pd.bdate_range(start=min(all_dates), end=max(all_dates)).date
        business_days_set = set(business_days)


        print("Total dates to process:", len(all_dates))

        for day in all_dates:
            # Only trade on business days
            if day not in business_days_set:
                continue
            # Set margin and max invest per trade (customize as needed)
            margin_available = cash 
            max_invest_today = margin_available * 0.2
            total_max_loss = 0

            if day in underlying_ts_set:
                print(f"Processing day: {day}")
                expiry_candidates = self.options_data[
                    (self.options_data['timestamp'].dt.date == day) & (self.options_data['dtm'] == 5)
                ]['expiration_date'].unique()

                if len(expiry_candidates) == 0:
                    continue
                expiry = expiry_candidates[0]
                options_day = self.options_data[
                    (self.options_data['timestamp'].dt.date == day) &
                    (self.options_data['expiration_date'] == expiry) &
                    (self.options_data['type'] == option_type)
                ]
                if options_day.empty:
                    print(f"No options data for {day} with expiry {expiry}")
                    continue

                # Decide which butterfly to try based on volatility
                pred_row = self.vol_predictions[self.vol_predictions['prediction_day'].dt.date == day]
                if not pred_row.empty:
                    pred_vol = pred_row.iloc[0]['predicted_vol']
                    realized_vol_row = self.underlying_data[self.underlying_data['timestamp'].dt.date == day]
                    if not realized_vol_row.empty:
                        realized_vol = realized_vol_row['realized_vol'].values[0]
                        if not pd.isna(pred_vol) and not pd.isna(realized_vol):
                            if pred_vol < self.pct_below * realized_vol:
                                # Try LONG butterfly only
                                butterfly_long = self.construct_long_butterfly(options_day)
                                if butterfly_long is not None and self.entry_rule_long(day, butterfly_long):
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
                                            'pred_vol_entry': pred_vol,
                                            'trade_type': 'long'
                                        })
                                else:
                                    print(f"No valid LONG butterfly for {day}")
                            elif pred_vol >  self.pct_above * realized_vol:
                                # Try SHORT butterfly only
                                butterfly_short = self.construct_short_butterfly(options_day)
                                if butterfly_short is not None and self.entry_rule_short(day, butterfly_short):
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
                                            'pred_vol_entry': pred_vol,
                                            'trade_type': 'short'
                                        })
                                else:
                                    print(f"No valid SHORT butterfly for {day}")
                            else:
                                print(f"Predicted vol not far enough from realized vol on {day}, no trade.")
                        else:
                            print(f"NaN vol for {day}")
                    else:
                        print(f"No realized vol for {day}")
                else:
                    print(f"No prediction for {day}")
            else:
                print(f"Skipping day {day}: No underlying data available")

            # 2. Check for early exit or expiry for open trades
            to_remove = []

            for i, trade in enumerate(open_trades):
                expiry_date = trade['expiry'].date() if hasattr(trade['expiry'], 'date') else trade['expiry']

                if expiry_date < day:
                    print(f"Force-removing expired trade from {trade['entry_date']} on {day}")
                    to_remove.append(i)
                    continue

                alarm = self.should_exit_early(trade, day)
                is_expiry = (expiry_date == day)
                if not (alarm or is_expiry):
                    continue

                exit_prices = self.get_exit_prices(trade['butterfly'], day)
                incomplete = any(pd.isna(exit_prices[leg]) for leg in exit_prices)

                if incomplete:
                    print(f"Skipping trade entered {trade['entry_date']} - incomplete price on {day}. Force-close.")
                    cash += trade['total_cost']
                    to_remove.append(i)
                    continue

                if trade['trade_type'] == 'long':
                    exit_price = exit_prices['long1'] + exit_prices['long3'] - 2 * exit_prices['short2']
                    exit_slippage = self.slippage * (abs(exit_prices['long1']) + abs(exit_prices['long3']) + 2 * abs(exit_prices['short2']))
                else:
                    exit_price = -exit_prices['short1'] + 2 * exit_prices['long2'] - exit_prices['short3']
                    exit_slippage = self.slippage * (abs(exit_prices['short1']) + abs(exit_prices['short3']) + 2 * abs(exit_prices['long2']))

                total_exit_value = exit_price * trade['n_butterflies']
                total_slippage_all = (trade['slippage'] + exit_slippage) * trade['n_butterflies']
                total_commission_all = trade['commission'] * trade['n_butterflies']
                pnl = (exit_price - trade['entry_price'] - trade['commission'] - trade['slippage'] - exit_slippage) * trade['n_butterflies']
                cash += total_exit_value

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
                    'pnl': pnl,
                    'underlying_price_entry': self.underlying_data.loc[self.underlying_data['timestamp'].dt.date == trade['entry_date'], 'close'].values[0],
                    'underlying_price_exit': self.underlying_data.loc[self.underlying_data['timestamp'].dt.date == day, 'close'].values[0],
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

            print(f"Open trades after removal on {day}: {len(open_trades)}")


            # 3. Mark-to-market for equity
            mtm_value = 0
            total_max_loss = 0  # <-- Reset before calculation
            for trade in open_trades:
                # Initialize last_prices if not present
                if 'last_prices' not in trade:
                    trade['last_prices'] = None

                current_prices = self.get_exit_prices(trade['butterfly'], day)
                if current_prices is not None:
                    trade['last_prices'] = current_prices  # Update last known prices
                else:
                    current_prices = trade['last_prices']  # Use last known prices

                if current_prices is None:
                    print(f"[WARNING] No available prices for MTM on {day} for trade entered {trade['entry_date']}. Skipping MTM for this trade.")
                    continue  # Still no prices available

                # Calculate mark-to-market value
                if trade['trade_type'] == 'long':
                    mtm_value += (current_prices['long1'] + current_prices['long3'] - 2 * current_prices['short2']) * trade['n_butterflies']
                else:
                    mtm_value += (-current_prices['short1'] + 2 * current_prices['long2'] - current_prices['short3']) * trade['n_butterflies']

                # Calculate max loss for each trade (unchanged)
                if trade['trade_type'] == 'long':
                    strikes = [
                        trade['butterfly']['long1']['strike_price'],
                        trade['butterfly']['short2']['strike_price'],
                        trade['butterfly']['long3']['strike_price']
                    ]
                    spread = max(strikes) - min(strikes)
                    max_loss = abs(trade['entry_price']) * trade['n_butterflies']
                else:
                    strikes = [
                        trade['butterfly']['short1']['strike_price'],
                        trade['butterfly']['long2']['strike_price'],
                        trade['butterfly']['short3']['strike_price']
                    ]
                    spread = max(strikes) - min(strikes)
                    max_loss = spread * trade['n_butterflies'] - abs(trade['entry_price']) * trade['n_butterflies']
                total_max_loss += max_loss

            margin_available = cash - total_max_loss 

            equity_curve.append({
                'date': day,
                'cash': cash,
                'equity': cash + mtm_value,
                'margin_available': margin_available,
                'total_max_loss': total_max_loss
            })

        results_df = pd.DataFrame(results)
        equity_df = pd.DataFrame(equity_curve)


        # --- Add open trades column to results_df ---
        if not results_df.empty:
            # Sort by exit_date to ensure correct order
            results_df = results_df.sort_values(['exit_date', 'entry_date']).reset_index(drop=True)
            open_trades_count = []
            currently_open = 0
            last_exit = None
            for idx, row in results_df.iterrows():
                # Count how many trades are open at this trade's exit_date
                # A trade is open if its entry_date <= this exit_date and its exit_date >= this exit_date
                open_count = ((results_df['entry_date'] <= row['exit_date']) & (results_df['exit_date'] >= row['exit_date'])).sum()
                open_trades_count.append(open_count)
            results_df['open_trades_at_exit'] = open_trades_count


        # Count trades that could not be closed at expiration due to missing data
        unclosed_at_expiry = 0
        for trade in open_trades:
            expiry_date = trade['expiry'].date() if hasattr(trade['expiry'], 'date') else trade['expiry']
            # If the trade's expiry is before or equal to the last backtest day, and it's still open, it couldn't be closed
            if expiry_date <= all_dates[-1]:
                unclosed_at_expiry += 1


        # Add summary statistics
        if not results_df.empty:
            total_trades = len(results_df)
            total_profit = results_df['pnl'].sum()
            avg_profit = results_df['pnl'].mean()
            win_rate = (results_df['pnl'] > 0).mean()
            max_drawdown = (equity_df['equity'].cummax() - equity_df['equity']).max()
            ending_cash = cash
            ending_equity = equity_df['equity'].iloc[-1]

            # --- Financial statistics ---
            equity_df = equity_df.sort_values('date')
            equity_df['returns'] = equity_df['equity'].pct_change().fillna(0)
            n_days = len(equity_df)
            n_years = n_days / 252  # Approximate trading days in a year

            # Annualized return (CAGR)
            start_equity = equity_df['equity'].iloc[0]
            end_equity = equity_df['equity'].iloc[-1]
            cagr = (end_equity / start_equity) ** (252 / n_days) - 1 if n_days > 1 else np.nan

            # Annualized volatility
            ann_vol = equity_df['returns'].std() * np.sqrt(252)

            # Sharpe ratio (risk-free rate as parameter)
            excess_returns = equity_df['returns'] - (risk_free_rate / 252)
            sharpe = (excess_returns.mean() / equity_df['returns'].std()) * np.sqrt(252) if equity_df['returns'].std() > 0 else np.nan

            # Sortino ratio (downside deviation, risk-free rate as parameter)
            downside = np.where(excess_returns < 0, excess_returns, 0)
            downside_std = np.sqrt(np.mean(downside ** 2))
            sortino = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else np.nan

            # Calmar ratio
            calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan

            stats = {
                'total_trades': total_trades,
                'total_profit': total_profit,
                'avg_profit per trade': avg_profit,
                'win_rate (Percentage of profitable trades)': win_rate,
                'max_drawdown': max_drawdown,
                'ending_cash': ending_cash,
                'ending_equity': ending_equity,
                'annualized_return (CAGR)': cagr,
                'annualized_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'unclosed_trades_at_expiry_due_to_missing_data': unclosed_at_expiry
            }
            print("Backtest Statistics (1 Year):")
            for k, v in stats.items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

            # --- Long vs Short trade statistics ---
            long_trades = results_df[results_df['trade_type'] == 'long']
            short_trades = results_df[results_df['trade_type'] == 'short']

            print("\nLong Butterfly Trades:")
            if not long_trades.empty:
                print(f"  Count: {len(long_trades)}")
                print(f"  Total PnL: {long_trades['pnl'].sum():.2f}")
                print(f"  Avg PnL: {long_trades['pnl'].mean():.2f}")
                print(f"  Win rate: {(long_trades['pnl'] > 0).mean():.2%}")
                print(f"  Max Drawdown: {(equity_df.set_index('date').loc[long_trades['exit_date']]['equity'].cummax() - equity_df.set_index('date').loc[long_trades['exit_date']]['equity']).max():.2f}")
            else:
                print("  No long butterfly trades.")

            print("\nShort Butterfly Trades:")
            if not short_trades.empty:
                print(f"  Count: {len(short_trades)}")
                print(f"  Total PnL: {short_trades['pnl'].sum():.2f}")
                print(f"  Avg PnL: {short_trades['pnl'].mean():.2f}")
                print(f"  Win rate: {(short_trades['pnl'] > 0).mean():.2%}")
                print(f"  Max Drawdown: {(equity_df.set_index('date').loc[short_trades['exit_date']]['equity'].cummax() - equity_df.set_index('date').loc[short_trades['exit_date']]['equity']).max():.2f}")
            else:
                print("  No short butterfly trades.")

        else:
            print("No trades executed.")


        # Visualization 
        self.plot_results(results_df, equity_df)



        return results_df, equity_df