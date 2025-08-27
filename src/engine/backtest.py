"""High-performance backtesting engine for yes-no markets."""

import numpy as np
import pandas as pd
from numba import jit, vectorize, float64
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from dataclasses import dataclass
import cupy as cp  # GPU acceleration

@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: List[Dict]
    equity_curve: np.ndarray

class YesNoBacktester:
    """GPU-accelerated backtesting engine."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
    @jit(nopython=True, parallel=True)
    def calculate_returns(self, prices: np.ndarray, signals: np.ndarray) -> np.ndarray:
        """Calculate returns using Numba JIT compilation."""
        returns = np.zeros(len(prices))
        position = 0.0
        
        for i in range(1, len(prices)):
            if signals[i] != position:
                position = signals[i]
            returns[i] = position * (prices[i] / prices[i-1] - 1)
        
        return returns
    
    def run_backtest(self, data: pd.DataFrame, strategy) -> BacktestResult:
        """Run backtest on historical data."""
        # Convert to GPU arrays if available
        prices = self.xp.array(data['price'].values)
        
        # Generate trading signals
        signals = strategy.generate_signals(data)
        
        # Calculate returns
        returns = self.calculate_returns(
            prices.get() if self.use_gpu else prices,
            signals
        )
        
        # Calculate metrics
        equity_curve = (1 + returns).cumprod()
        total_return = equity_curve[-1] - 1
        sharpe_ratio = self._calculate_sharpe(returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Extract trades
        trades = self._extract_trades(signals, prices, returns)
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=trades,
            equity_curve=equity_curve
        )
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_sharpe(returns: np.ndarray, risk_free: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free / 252
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return np.min(drawdown)
    
    def _extract_trades(self, signals: np.ndarray, prices: np.ndarray, returns: np.ndarray) -> List[Dict]:
        """Extract individual trades from signals."""
        trades = []
        position = 0
        entry_price = 0
        
        for i in range(len(signals)):
            if signals[i] != position:
                if position != 0:
                    # Close position
                    trades.append({
                        'entry': entry_price,
                        'exit': prices[i],
                        'pnl': (prices[i] - entry_price) / entry_price * position,
                        'duration': i
                    })
                
                position = signals[i]
                if position != 0:
                    entry_price = prices[i]
        
        return trades

class ParallelBacktester:
    """Parallel backtesting for multiple strategies."""
    
    def __init__(self, n_workers: int = mp.cpu_count()):
        self.n_workers = n_workers
        
    def run_parallel(self, data: pd.DataFrame, strategies: List) -> List[BacktestResult]:
        """Run multiple strategies in parallel."""
        with mp.Pool(self.n_workers) as pool:
            results = pool.starmap(
                self._run_single,
                [(data, strategy) for strategy in strategies]
            )
        return results
    
    @staticmethod
    def _run_single(data: pd.DataFrame, strategy) -> BacktestResult:
        """Run single backtest."""
        backtester = YesNoBacktester()
        return backtester.run_backtest(data, strategy)
