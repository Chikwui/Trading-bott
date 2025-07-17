"""
Backtesting Results Module

This module contains the BacktestResult class which handles storage and analysis
of backtesting results, including performance metrics and trade analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import seaborn as sns
import pyfolio as pf
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

class BacktestResult:
    """
    Container for backtest results and performance metrics.
    
    This class stores and analyzes the results of a backtest, including:
    - Trade history
    - Equity curve
    - Performance metrics (Sharpe ratio, drawdown, etc.)
    - Trade analysis (win rate, profit factor, etc.)
    """
    
    def __init__(self):
        """Initialize the BacktestResult with empty containers."""
        self.signals = []
        self.positions = []
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        self.trade_analysis = {}
        self.daily_returns = None
        self.monthly_returns = None
        self.yearly_returns = None
        self.drawdown = None
        self.drawdown_duration = None
    
    def calculate_metrics(self):
        """Calculate performance metrics for the backtest."""
        if not self.trades or len(self.trades) == 0:
            return
        
        # Calculate returns
        self.daily_returns = self.equity_curve['equity'].pct_change().dropna()
        self.monthly_returns = self.equity_curve['equity'].resample('M').last().pct_change().dropna()
        self.yearly_returns = self.equity_curve['equity'].resample('Y').last().pct_change().dropna()
        
        # Calculate drawdown
        rolling_max = self.equity_curve['equity'].cummax()
        self.drawdown = (self.equity_curve['equity'] - rolling_max) / rolling_max
        
        # Calculate drawdown duration
        self.drawdown_duration = (self.drawdown != 0).astype(int).groupby(
            (self.drawdown == 0).astype(int).cumsum()
        ).cumsum()
        
        # Basic metrics
        total_return = (self.equity_curve['equity'].iloc[-1] / self.equity_curve['equity'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(self.daily_returns)) - 1
        annual_volatility = self.daily_returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * self.daily_returns.mean() / (self.daily_returns.std() + 1e-10)
        sortino_ratio = self._calculate_sortino_ratio()
        max_drawdown = self.drawdown.min()
        calmar_ratio = annual_return / (abs(max_drawdown) + 1e-10)
        
        # Trade analysis
        winning_trades = [t for t in self.trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in self.trades if t['pnl_pct'] <= 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = abs(np.mean([t['pnl_pct'] for t in losing_trades])) if losing_trades else 0
        profit_factor = (len(winning_trades) * avg_win) / (len(losing_trades) * avg_loss + 1e-10)
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_trade_duration': np.mean([t.get('duration', 0) for t in self.trades]) if self.trades else 0,
            'max_trade_duration': max([t.get('duration', 0) for t in self.trades]) if self.trades else 0,
            'min_trade_duration': min([t.get('duration', 0) for t in self.trades]) if self.trades else 0,
        }
        
        # Trade analysis
        self.trade_analysis = {
            'by_month': self._analyze_trades_by_month(),
            'by_weekday': self._analyze_trades_by_weekday(),
            'by_hour': self._analyze_trades_by_hour(),
            'by_holding_period': self._analyze_trades_by_holding_period(),
            'consecutive_wins_losses': self._analyze_consecutive_trades()
        }
    
    def _calculate_sortino_ratio(self, target_return=0.0, annualize=True):
        """Calculate the Sortino ratio."""
        if self.daily_returns is None or len(self.daily_returns) == 0:
            return 0.0
            
        # Calculate downside returns
        downside_returns = self.daily_returns[self.daily_returns < target_return]
        if len(downside_returns) == 0:
            return float('inf')
            
        # Calculate downside deviation
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        
        if downside_deviation == 0:
            return float('inf')
            
        # Calculate excess returns
        excess_returns = self.daily_returns - target_return
        avg_excess_return = np.mean(excess_returns)
        
        # Annualize if needed
        if annualize:
            avg_excess_return *= 252
            downside_deviation *= np.sqrt(252)
            
        return avg_excess_return / downside_deviation
    
    def _analyze_trades_by_month(self) -> Dict[int, Dict[str, Any]]:
        """Analyze trades by month of the year."""
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        trades_df['month'] = trades_df['exit_time'].dt.month
        
        result = {}
        for month in range(1, 13):
            month_trades = trades_df[trades_df['month'] == month]
            if len(month_trades) > 0:
                result[month] = {
                    'count': len(month_trades),
                    'win_rate': len(month_trades[month_trades['pnl_pct'] > 0]) / len(month_trades),
                    'avg_pnl': month_trades['pnl_pct'].mean(),
                    'total_pnl': month_trades['pnl_pct'].sum(),
                }
        return result
    
    def _analyze_trades_by_weekday(self) -> Dict[int, Dict[str, Any]]:
        """Analyze trades by day of the week."""
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        trades_df['weekday'] = trades_df['exit_time'].dt.weekday
        
        result = {}
        for day in range(5):  # 0=Monday, 4=Friday
            day_trades = trades_df[trades_df['weekday'] == day]
            if len(day_trades) > 0:
                result[day] = {
                    'count': len(day_trades),
                    'win_rate': len(day_trades[day_trades['pnl_pct'] > 0]) / len(day_trades),
                    'avg_pnl': day_trades['pnl_pct'].mean(),
                    'total_pnl': day_trades['pnl_pct'].sum(),
                }
        return result
    
    def _analyze_trades_by_hour(self) -> Dict[int, Dict[str, Any]]:
        """Analyze trades by hour of the day."""
        if not self.trades:
            return {}
            
        trades_df = pd.DataFrame(self.trades)
        trades_df['hour'] = trades_df['entry_time'].dt.hour
        
        result = {}
        for hour in range(24):
            hour_trades = trades_df[trades_df['hour'] == hour]
            if len(hour_trades) > 0:
                result[hour] = {
                    'count': len(hour_trades),
                    'win_rate': len(hour_trades[hour_trades['pnl_pct'] > 0]) / len(hour_trades),
                    'avg_pnl': hour_trades['pnl_pct'].mean(),
                    'total_pnl': hour_trades['pnl_pct'].sum(),
                }
        return result
    
    def _analyze_trades_by_holding_period(self, bins=None) -> Dict[str, Dict[str, Any]]:
        """Analyze trades by holding period in days."""
        if not self.trades:
            return {}
            
        if bins is None:
            bins = [0, 1, 5, 10, 20, 50, 100, 252, float('inf')]
            
        trades_df = pd.DataFrame(self.trades)
        trades_df['holding_period'] = (trades_df['exit_time'] - trades_df['entry_time']).dt.days
        trades_df['holding_bin'] = pd.cut(trades_df['holding_period'], bins)
        
        result = {}
        for bin_name in trades_df['holding_bin'].unique():
            if pd.isna(bin_name):
                continue
                
            bin_trades = trades_df[trades_df['holding_bin'] == bin_name]
            result[str(bin_name)] = {
                'count': len(bin_trades),
                'win_rate': len(bin_trades[bin_trades['pnl_pct'] > 0]) / len(bin_trades),
                'avg_pnl': bin_trades['pnl_pct'].mean(),
                'total_pnl': bin_trades['pnl_pct'].sum(),
                'avg_holding_period': bin_trades['holding_period'].mean(),
            }
        return result
    
    def _analyze_consecutive_trades(self) -> Dict[str, int]:
        """Analyze consecutive winning and losing trades."""
        if not self.trades:
            return {}
            
        trades = sorted(self.trades, key=lambda x: x['exit_time'])
        
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade['pnl_pct'] > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return {
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'win_streak': current_wins if current_wins > 0 else 0,
            'losing_streak': current_losses if current_losses > 0 else 0
        }
    
    def plot_equity_curve(self, title='Equity Curve', figsize=(12, 8), save_path=None):
        """
        Plot the equity curve with drawdown.
        
        Args:
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Path to save the plot (optional)
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            print("No equity curve data to plot")
            return
            
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Plot equity curve
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.equity_curve.index, self.equity_curve['equity'], label='Equity', color='blue')
        ax1.set_title(title)
        ax1.set_ylabel('Equity')
        ax1.grid(True)
        
        # Plot drawdown
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.fill_between(self.equity_curve.index, self.drawdown * 100, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.set_ylabel('Drawdown %')
        ax2.grid(True)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_trade_distribution(self, figsize=(12, 8), save_path=None):
        """
        Plot distribution of trade returns.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Path to save the plot (optional)
        """
        if not self.trades:
            print("No trades to plot")
            return
            
        trades_df = pd.DataFrame(self.trades)
        
        plt.figure(figsize=figsize)
        
        # Plot histogram of trade returns
        plt.subplot(2, 1, 1)
        sns.histplot(trades_df['pnl_pct'], kde=True, bins=30)
        plt.axvline(0, color='r', linestyle='--')
        plt.title('Distribution of Trade Returns')
        plt.xlabel('Return %')
        plt.ylabel('Frequency')
        
        # Plot cumulative returns by trade
        plt.subplot(2, 1, 2)
        trades_df['cumulative_pnl'] = trades_df['pnl_pct'].cumsum()
        plt.plot(trades_df.index, trades_df['cumulative_pnl'])
        plt.title('Cumulative Returns by Trade')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Return %')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, title='Backtest Report', output_dir='reports') -> str:
        """
        Generate a comprehensive backtest report.
        
        Args:
            title: Report title
            output_dir: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save equity curve plot
        equity_plot_path = os.path.join(output_dir, 'equity_curve.png')
        self.plot_equity_curve(title=title, save_path=equity_plot_path)
        
        # Save trade distribution plot
        trade_dist_path = os.path.join(output_dir, 'trade_distribution.png')
        self.plot_trade_distribution(save_path=trade_dist_path)
        
        # Generate HTML report
        report_path = os.path.join(output_dir, 'backtest_report.html')
        
        with open(report_path, 'w') as f:
            f.write(f"<html><head><title>{title}</title></head><body>")
            f.write(f"<h1>{title}</h1>")
            f.write(f"<h2>Performance Summary</h2>")
            
            # Performance metrics table
            f.write("<h3>Key Metrics</h3>")
            f.write("<table border='1' style='border-collapse: collapse; width: 100%;'>")
            f.write("<tr><th>Metric</th><th>Value</th></tr>")
            
            metrics = {
                'Total Return': f"{self.metrics.get('total_return', 0) * 100:.2f}%",
                'Annual Return': f"{self.metrics.get('annual_return', 0) * 100:.2f}%",
                'Annual Volatility': f"{self.metrics.get('annual_volatility', 0) * 100:.2f}%",
                'Sharpe Ratio': f"{self.metrics.get('sharpe_ratio', 0):.2f}",
                'Sortino Ratio': f"{self.metrics.get('sortino_ratio', 0):.2f}",
                'Max Drawdown': f"{self.metrics.get('max_drawdown', 0) * 100:.2f}%",
                'Calmar Ratio': f"{self.metrics.get('calmar_ratio', 0):.2f}",
                'Win Rate': f"{self.metrics.get('win_rate', 0) * 100:.2f}%",
                'Profit Factor': f"{self.metrics.get('profit_factor', 0):.2f}",
                'Total Trades': self.metrics.get('total_trades', 0),
                'Winning Trades': self.metrics.get('winning_trades', 0),
                'Losing Trades': self.metrics.get('losing_trades', 0),
            }
            
            for name, value in metrics.items():
                f.write(f"<tr><td>{name}</td><td>{value}</td></tr>")
            
            f.write("</table>")
            
            # Add equity curve image
            f.write(f"<h3>Equity Curve</h3>")
            f.write(f"<img src='equity_curve.png' style='max-width: 100%;'>")
            
            # Add trade distribution
            f.write(f"<h3>Trade Distribution</h3>")
            f.write(f"<img src='trade_distribution.png' style='max-width: 100%;'>")
            
            # Trade analysis
            f.write("<h3>Trade Analysis</h3>")
            
            # Monthly performance
            f.write("<h4>Monthly Performance</h4>")
            f.write("<table border='1' style='border-collapse: collapse; width: 100%;'>")
            f.write("<tr><th>Month</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th><th>Total Return</th></tr>")
            
            for month, stats in self.trade_analysis.get('by_month', {}).items():
                month_name = datetime(2020, month, 1).strftime('%B')
                f.write(f"<tr>")
                f.write(f"<td>{month_name}</td>")
                f.write(f"<td>{stats['count']}</td>")
                f.write(f"<td>{stats['win_rate'] * 100:.1f}%</td>")
                f.write(f"<td>{stats['avg_pnl'] * 100:.2f}%</td>")
                f.write(f"<td>{stats['total_pnl'] * 100:.2f}%</td>")
                f.write("</tr>")
            
            f.write("</table>")
            
            # Weekday performance
            f.write("<h4>Weekday Performance</h4>")
            f.write("<table border='1' style='border-collapse: collapse; width: 100%;'>")
            f.write("<tr><th>Day</th><th>Trades</th><th>Win Rate</th><th>Avg Return</th><th>Total Return</th></tr>")
            
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            for day, stats in self.trade_analysis.get('by_weekday', {}).items():
                f.write(f"<tr>")
                f.write(f"<td>{weekdays[day]}</td>")
                f.write(f"<td>{stats['count']}</td>")
                f.write(f"<td>{stats['win_rate'] * 100:.1f}%</td>")
                f.write(f"<td>{stats['avg_pnl'] * 100:.2f}%</td>")
                f.write(f"<td>{stats['total_pnl'] * 100:.2f}%</td>")
                f.write("</tr>")
            
            f.write("</table>")
            
            # Consecutive trades
            consec = self.trade_analysis.get('consecutive_wins_losses', {})
            f.write("<h4>Consecutive Trades</h4>")
            f.write("<ul>")
            f.write(f"<li>Max Consecutive Wins: {consec.get('max_consecutive_wins', 0)}</li>")
            f.write(f"<li>Max Consecutive Losses: {consec.get('max_consecutive_losses', 0)}</li>")
            f.write(f"<li>Current Win Streak: {consec.get('win_streak', 0)}</li>")
            f.write(f"<li>Current Losing Streak: {consec.get('losing_streak', 0)}</li>")
            f.write("</ul>")
            
            # Trades table
            f.write("<h3>Trade History</h3>")
            f.write("<table border='1' style='border-collapse: collapse; width: 100%;'>")
            f.write("<tr><th>#</th><th>Entry</th><th>Exit</th><th>Symbol</th><th>Side</th><th>Return %</th><th>Holding Period</th></tr>")
            
            for i, trade in enumerate(self.trades, 1):
                f.write(f"<tr>")
                f.write(f"<td>{i}</td>")
                f.write(f"<td>{trade['entry_time'].strftime('%Y-%m-%d %H:%M')}</td>")
                f.write(f"<td>{trade['exit_time'].strftime('%Y-%m-%d %H:%M')}</td>")
                f.write(f"<td>{trade.get('symbol', 'N/A')}</td>")
                f.write(f"<td>{'Long' if trade.get('side', '') == 'long' else 'Short'}</td>")
                f.write(f"<td style='color: {'green' if trade['pnl_pct'] > 0 else 'red'}'>{trade['pnl_pct'] * 100:.2f}%</td>")
                f.write(f"<td>{(trade['exit_time'] - trade['entry_time']).days} days</td>")
                f.write("</tr>")
            
            f.write("</table>")
            
            # Footer
            f.write(f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            f.write("</body></html>")
        
        print(f"Report generated: {report_path}")
        return report_path
