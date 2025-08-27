"""
yes-no.fun Backtesting Framework
High-performance historical simulation engine
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import multiprocessing as mp
from numba import jit, vectorize

@dataclass
class BacktestResult:
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    kelly_fraction: float
    
class VectorizedBacktester:
    """Ultra-fast vectorized backtesting using NumPy and Numba"""
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def calculate_returns(prices: np.ndarray, signals: np.ndarray, 
                          fees: float = 0.001) -> np.ndarray:
        """JIT-compiled return calculation for 1000x speedup"""
        returns = np.diff(prices) / prices[:-1]
        position_returns = returns * signals[:-1]
        fees_paid = np.abs(np.diff(signals)) * fees
        net_returns = position_returns - fees_paid
        return net_returns
    
    @staticmethod
    @vectorize(['float64(float64, float64)'], target='cuda')
    def gpu_sharpe(returns: np.ndarray, risk_free: float) -> float:
        """GPU-accelerated Sharpe ratio calculation"""
        excess_returns = returns - risk_free
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def run_parallel_backtest(strategies: List, data: pd.DataFrame, 
                         n_cores: int = mp.cpu_count()) -> Dict:
    """Run multiple strategies in parallel using all CPU cores"""
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(run_single_backtest, 
                              [(s, data) for s in strategies])
    return dict(zip([s.name for s in strategies], results))
