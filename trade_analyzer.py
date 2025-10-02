#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trade Analyzer - Analyze trades from alerts.csv
Extracts: Symbol, buyPrice, sellPrice, maxPump, maxDump, pnl%, and advanced metrics
"""

import pandas as pd
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradeAnalyzer:
    def __init__(self, alerts_file=None, historical_data_folder=None, target_profit=None):
        self.alerts_file = alerts_file or 'alerts.csv'
        self.historical_data_folder = historical_data_folder or 'historical_market_data'
        self.target_profit = target_profit
        self.results = []
        
    def load_alerts(self):
        """Load alerts file"""
        try:
            df = pd.read_csv(self.alerts_file)
            # Filter records with id = 1111 only
            signals_df = df[df['id'] == 1111].copy()
            signals_df['candel_date'] = pd.to_datetime(signals_df['candel_date']).dt.tz_localize(None)
            return signals_df.sort_values(['symbol', 'candel_date'])
        except Exception as e:
            print(f"Error loading file: {e}")
            return pd.DataFrame()
    
    def extract_trade_pairs(self, df):
        """Extract buy-sell pairs for each symbol"""
        trade_pairs = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            # Find buy-sell pairs
            buy_orders = []
            
            for _, row in symbol_data.iterrows():
                if row['side'] == 'buy':
                    buy_orders.append(row)
                elif row['side'] == 'sell' and buy_orders:
                    # Match sell with last buy
                    buy_order = buy_orders.pop()
                    trade_pairs.append({
                        'symbol': symbol,
                        'buy_date': buy_order['candel_date'],
                        'sell_date': row['candel_date'],
                        'buy_price': buy_order['alert_price'],
                        'sell_price': row['alert_price']
                    })
        
        return trade_pairs
    
    def load_historical_data(self, symbol):
        """Load historical data for symbol"""
        file_path = os.path.join(self.historical_data_folder, f"{symbol}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_trade_metrics(self, trade, all_trades):
        """Calculate trade metrics with new advanced metrics"""
        symbol = trade['symbol']
        buy_date = trade['buy_date']
        sell_date = trade['sell_date']
        buy_price = trade['buy_price']
        sell_price = trade['sell_price']
        
        # Load historical data
        historical_df = self.load_historical_data(symbol)
        
        # Calculate hold duration in minutes
        hold_duration_minutes = (sell_date - buy_date).total_seconds() / 60
        
        # Calculate PnL
        pnl_percent = ((sell_price - buy_price) / buy_price) * 100
        
        if historical_df.empty:
            return {
                'symbol': symbol,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'max_pump': 0,
                'max_dump': 0,
                'pnl_percent': pnl_percent,
                'original_pnl_percent': pnl_percent,
                'missed_profits': 0,
                'hold_duration_minutes': hold_duration_minutes,
                'efficiency_score': 0,
                'risk_reward_ratio': 0,
                'avg_volume': 0,
                'trade_count': 0,
                'time_to_max_pump_minutes': 0,
                'exit_timing_score': 50
            }
        
        # Filter data during trade period
        if 'timestamp' in historical_df.columns:
            trade_period = historical_df[
                (historical_df['timestamp'] >= buy_date) & 
                (historical_df['timestamp'] <= sell_date)
            ].copy()
        else:
            trade_period = historical_df.copy()
        
        if trade_period.empty:
            return {
                'symbol': symbol,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'max_pump': 0,
                'max_dump': 0,
                'pnl_percent': pnl_percent,
                'original_pnl_percent': pnl_percent,
                'missed_profits': 0,
                'hold_duration_minutes': hold_duration_minutes,
                'efficiency_score': 0,
                'risk_reward_ratio': 0,
                'avg_volume': 0,
                'trade_count': 0,
                'time_to_max_pump_minutes': 0,
                'exit_timing_score': 50
            }
        
        # Calculate max profit and max loss
        if 'high' in trade_period.columns and 'low' in trade_period.columns:
            max_high = trade_period['high'].max()
            min_low = trade_period['low'].min()
            max_high_idx = trade_period['high'].idxmax()
        elif 'close' in trade_period.columns:
            max_high = trade_period['close'].max()
            min_low = trade_period['close'].min()
            max_high_idx = trade_period['close'].idxmax()
        else:
            max_high = buy_price
            min_low = buy_price
            max_high_idx = 0
        
        max_pump = ((max_high - buy_price) / buy_price) * 100 if buy_price != 0 else 0
        max_dump = ((min_low - buy_price) / buy_price) * 100 if buy_price != 0 else 0
        
        # Calculate time to reach max pump
        time_to_max_pump_minutes = 0
        exit_timing_score = 50  # Default neutral score
        
        if max_high_idx in trade_period.index:
            timestamp_col = 'timestamp'
            if timestamp_col in trade_period.columns:
                max_pump_time = trade_period.loc[max_high_idx, timestamp_col]
                time_to_max_pump_minutes = (max_pump_time - buy_date).total_seconds() / 60
                
                # Calculate exit timing score (0-100)
                # Perfect timing = selling at max pump time = 100
                # Selling too early or too late = lower score
                total_duration = hold_duration_minutes
                optimal_time = time_to_max_pump_minutes
                
                if total_duration > 0 and optimal_time >= 0:
                    # If we sold at optimal time, score = 100
                    # If we sold at start/end, score approaches 0
                    time_efficiency = 1 - abs(total_duration - optimal_time) / max(total_duration, optimal_time, 1)
                    exit_timing_score = max(0, time_efficiency * 100)
        
        # Calculate efficiency_score
        efficiency_score = (pnl_percent / max_pump * 100) if max_pump != 0 else 0
        
        # Calculate risk_reward_ratio
        risk_reward_ratio = abs(max_dump / pnl_percent) if pnl_percent != 0 else 0
        
        # Calculate average volume and trade count
        avg_volume = trade_period['volume'].mean() if 'volume' in trade_period.columns else 0
        trade_count_total = trade_period['trade_count'].sum() if 'trade_count' in trade_period.columns else 0
        
        # Calculate missed profits (difference between max_pump and actual PnL)
        missed_profits = 0
        if max_pump > 0 and pnl_percent < max_pump:
            missed_profits = max_pump - pnl_percent
        
        # Apply target profit simulation if specified
        simulated_pnl = pnl_percent
        original_pnl = pnl_percent
        
        if self.target_profit is not None:
            required_max_pump = self.target_profit + 1  # Add 1% buffer as requested
            if max_pump >= required_max_pump:
                # Would have sold at target profit
                simulated_pnl = self.target_profit
                # Recalculate efficiency for simulated exit
                efficiency_score = (simulated_pnl / max_pump * 100) if max_pump != 0 else 0
                # Recalculate missed profits for simulation
                if max_pump > simulated_pnl:
                    missed_profits = max_pump - simulated_pnl
        
        return {
            'symbol': symbol,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'max_pump': max_pump,
            'max_dump': max_dump,
            'pnl_percent': simulated_pnl,
            'original_pnl_percent': original_pnl,
            'missed_profits': missed_profits,
            'hold_duration_minutes': hold_duration_minutes,
            'efficiency_score': efficiency_score,
            'risk_reward_ratio': abs(max_dump / simulated_pnl) if simulated_pnl != 0 else 0,
            'avg_volume': avg_volume,
            'trade_count': trade_count_total,
            'time_to_max_pump_minutes': time_to_max_pump_minutes,
            'exit_timing_score': exit_timing_score
        }
    
    def analyze_trades(self):
        """Analyze all trades"""
        print("Loading alerts file...")
        alerts_df = self.load_alerts()
        
        if alerts_df.empty:
            print("No data to analyze")
            return
        
        print("Extracting trade pairs...")
        trade_pairs = self.extract_trade_pairs(alerts_df)
        
        if not trade_pairs:
            print("No trade pairs found")
            return
        
        print(f"Found {len(trade_pairs)} trades")
        
        # Analyze each trade
        for i, trade in enumerate(trade_pairs, 1):
            print(f"Analyzing trade {i}/{len(trade_pairs)}: {trade['symbol']}")
            metrics = self.calculate_trade_metrics(trade, trade_pairs)
            metrics['buy_date'] = trade['buy_date']  # Add buy_date to results for sorting
            self.results.append(metrics)
        
        # Calculate additional metrics
        self.calculate_additional_metrics()
        
        # Save results
        self.save_results()
    
    def calculate_additional_metrics(self):
        """Calculate additional metrics like win_rate and consecutive_losses"""
        if not self.results:
            return
        
        # Calculate win_rate
        winning_trades = sum(1 for result in self.results if result['pnl_percent'] > 0)
        total_trades = len(self.results)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate consecutive_losses per symbol
        symbol_losses = {}
        
        # Sort results by symbol and date (assume current order is correct)
        current_symbol = None
        consecutive_count = 0
        
        for result in self.results:
            symbol = result['symbol']
            is_loss = result['pnl_percent'] < 0
            
            if symbol != current_symbol:
                # New symbol
                current_symbol = symbol
                consecutive_count = 1 if is_loss else 0
                symbol_losses[symbol] = consecutive_count
            else:
                # Same symbol
                if is_loss:
                    consecutive_count += 1
                    symbol_losses[symbol] = max(symbol_losses.get(symbol, 0), consecutive_count)
                else:
                    consecutive_count = 0
        
        # Add new metrics to each result
        for result in self.results:
            result['win_rate'] = win_rate
            result['consecutive_losses'] = symbol_losses.get(result['symbol'], 0)
    
    def calculate_portfolio_simulation(self, initial_capital=1000):
        """Simulate trading with fixed capital"""
        if not self.results:
            return initial_capital, []
        
        # Sort trades by buy_date for chronological simulation
        sorted_results = sorted(self.results, key=lambda x: x.get('buy_date', pd.Timestamp.min))
        
        capital = initial_capital
        trade_log = []
        
        print(f"\nPortfolio Simulation (Starting Capital: ${initial_capital})")
        print("=" * 60)
        
        for i, trade in enumerate(sorted_results, 1):
            pnl_percent = trade['pnl_percent']
            
            # Apply the percentage change to current capital
            trade_pnl = capital * (pnl_percent / 100)
            capital += trade_pnl
            
            trade_log.append({
                'trade_num': i,
                'symbol': trade['symbol'],
                'pnl_percent': pnl_percent,
                'trade_pnl': trade_pnl,
                'capital_after': capital
            })
            
            # Print all trades
            print(f"Trade {i:2}: {trade['symbol']} | PnL: {pnl_percent:6.2f}% | ${trade_pnl:7.2f} | Capital: ${capital:8.2f}")
        
        return capital, trade_log
    
    def save_results(self):
        """Save results to CSV file"""
        if not self.results:
            print("No results to save")
            return
        
        results_df = pd.DataFrame(self.results)
        
        # Reorder columns to match requested format
        column_order = [
            'symbol', 'buy_date', 'buy_price', 'sell_price', 'max_pump', 'max_dump', 'pnl_percent',
            'original_pnl_percent', 'missed_profits', 'hold_duration_minutes', 'efficiency_score', 'risk_reward_ratio', 
            'avg_volume', 'trade_count', 'time_to_max_pump_minutes', 
            'exit_timing_score', 'consecutive_losses'
        ]
        
        # Calculate win rate before removing it from dataframe  
        win_rate = results_df['win_rate'].iloc[0] if 'win_rate' in results_df.columns else 0
        
        # Keep only available columns
        available_columns = [col for col in column_order if col in results_df.columns]
        results_df = results_df[available_columns]
        
        output_file = f"trade_analysis_results.csv"
        results_df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to: {output_file}")
        print("\nResults Summary:")
        print(f"Total trades: {len(results_df)}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Average P&L: {results_df['pnl_percent'].mean():.2f}%")
        print(f"Best trade: {results_df['pnl_percent'].max():.2f}%")
        print(f"Worst trade: {results_df['pnl_percent'].min():.2f}%")
        print(f"Average hold duration: {results_df['hold_duration_minutes'].mean():.1f} minutes")
        print(f"Average efficiency: {results_df['efficiency_score'].mean():.1f}%")
        
        # Missed profits analysis
        total_missed = results_df['missed_profits'].sum()
        avg_missed = results_df['missed_profits'].mean()
        max_missed = results_df['missed_profits'].max()
        
        print(f"\nMissed Profits Analysis:")
        print(f"Total missed profits: {total_missed:.2f}%")
        print(f"Average missed per trade: {avg_missed:.2f}%")
        print(f"Biggest missed opportunity: {max_missed:.2f}%")
        
        # Portfolio simulation
        final_capital, trade_log = self.calculate_portfolio_simulation(1000)
        total_return = ((final_capital - 1000) / 1000) * 100
        print(f"\nPortfolio Performance:")
        print(f"Starting Capital: $1,000.00")
        print(f"Final Capital: ${final_capital:.2f}")
        print(f"Total Return: {total_return:.2f}%")
        
        # Print all trades sorted by buy_date chronologically
        print("\nAll trades:")
        # Sort by buy_date if available
        if 'buy_date' in results_df.columns:
            results_df_sorted = results_df.sort_values('buy_date').reset_index(drop=True)
        else:
            results_df_sorted = results_df
        
        basic_cols = ['symbol', 'buy_price', 'sell_price', 'max_pump', 'max_dump', 'pnl_percent', 
                     'missed_profits', 'hold_duration_minutes', 'time_to_max_pump_minutes', 'exit_timing_score', 'consecutive_losses']
        available_cols = [col for col in basic_cols if col in results_df_sorted.columns]
        print(results_df_sorted[available_cols].to_string(index=False))

def main():
    folder_name = None
    target_profit = None
    
    # Parse arguments
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--') and not arg[2:].isdigit():
            # Archive folder like --alerts_archive_2025-09-24
            folder_name = arg[2:]
        elif arg.startswith('--') and arg[2:].isdigit():
            # Target profit like --5 or --10
            target_profit = int(arg[2:])
    
    if folder_name:
        # Set up file paths for new structure
        alerts_file = f"{folder_name}/{folder_name}.csv"
        historical_data_folder = f"{folder_name}/archive_historical_market_data"
        
        print(f"Using alerts file: {alerts_file}")
        print(f"Using historical data folder: {historical_data_folder}")
        if target_profit:
            print(f"Target Profit Simulation: {target_profit}% (requires {target_profit + 1}% max pump)")
        
        # Check if files exist
        if not os.path.exists(alerts_file):
            print(f"Error: {alerts_file} not found!")
            sys.exit(1)
        
        if not os.path.exists(historical_data_folder):
            print(f"Error: {historical_data_folder} folder not found!")
            sys.exit(1)
            
        analyzer = TradeAnalyzer(alerts_file, historical_data_folder, target_profit)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        # Regular argument without --
        folder_name = sys.argv[1]
        
        # Set up file paths for new structure
        alerts_file = f"{folder_name}/{folder_name}.csv"
        historical_data_folder = f"{folder_name}/archive_historical_market_data"
        
        print(f"Using alerts file: {alerts_file}")
        print(f"Using historical data folder: {historical_data_folder}")
        
        # Check if files exist
        if not os.path.exists(alerts_file):
            print(f"Error: {alerts_file} not found!")
            sys.exit(1)
        
        if not os.path.exists(historical_data_folder):
            print(f"Error: {historical_data_folder} folder not found!")
            sys.exit(1)
            
        analyzer = TradeAnalyzer(alerts_file, historical_data_folder, target_profit)
    else:
        # Show usage instructions when no arguments provided
        print("Usage:")
        print("  python3 trade_analyzer.py <folder_name>")
        print()
        print("Examples:")
        print("  python3 trade_analyzer.py alerts_archive_2025-09-25")
        print()
        print("Available folders:")
        
        # List available archive folders
        import glob
        archive_folders = glob.glob("alerts_archive_*")
        if archive_folders:
            for folder in sorted(archive_folders):
                print(f"  - {folder}")
        else:
            print("  No archive folders found")
        
        sys.exit(0)
    
    analyzer.analyze_trades()

if __name__ == "__main__":
    main()