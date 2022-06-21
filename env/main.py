import os
import scipy
from math import inf

import numpy as np
import pandas as pd

from numba import njit

from utility_structures import PositionsQueue
from graph_tools import lineplot_by_y

def get_data(directory_name: str, sample_size=+inf) -> tuple:
    tmp = {'ask': [], 'bid': []}
    for filename in os.listdir("pickles"):
        tmp_frame = pd.read_pickle((os.path.join("pickles", filename)))
        tmp['ask'].append(np.array(tmp_frame.ask_price)[:sample_size])
        tmp['bid'].append(np.array(tmp_frame.bid_price)[:sample_size])
    ask_prices = np.stack(tmp['ask'], axis=0)
    bid_prices = np.stack(tmp['bid'], axis=0)
    return (ask_prices, bid_prices)

@njit
def get_weights(size, q):
    weights = np.zeros(0) 
    weight = 1
    for _ in range(size):
        weights = np.append(weights, [weight])
        weight *= q
    return weights[::-1]

@njit
def get_ewma(data_array: np.ndarray, window_size, q: float):
    length = data_array.shape[0]
    weights = get_weights(window_size, q)
    divider = weights.sum()
    result = np.zeros(0)
    for i in range(length - window_size + 1):
        subsequence = data_array[i:i+window_size]
        new_ewma = (subsequence*weights).sum() / divider
        result = np.append(result, [new_ewma])
    return result

@njit
def generate_signal(ask_prices, bid_prices, long_threshold, short_threshold, short_window, long_window, q):
    current_prices = (ask_prices + bid_prices) / 2
    signals_array = np.zeros((current_prices.shape[0], current_prices.shape[1]))
    if long_window < short_window:
        long_window, short_window = short_window, long_window
    for active_id in range(current_prices.shape[0]):
        active = current_prices[active_id]
        short_ewmas = get_ewma(active, short_window, q)
        long_ewmas = get_ewma(active, long_window, q)
        max_window_length = max(short_window, long_window)
        active_signals = np.zeros(max_window_length)
        for timestamp in range(max_window_length, active.shape[0]):
            error = long_ewmas[timestamp-long_window] - short_ewmas[timestamp-short_window]
#             print(round(error, 3))
            if error > long_threshold:
                new_signal = np.zeros(1) + 1
            elif error < short_threshold:
                new_signal = np.zeros(1) - 1
            else:
                new_signal = np.zeros(1) 
            active_signals = np.append(active_signals, new_signal)
        signals_array[active_id] = active_signals
    return signals_array

@njit
def get_UPL(balance, currencies, ask_prices, timestamp):
    UPL = balance
    for active_id in range(ask_prices.shape[0]):
        UPL += currencies[active_id] * ask_prices[active_id][timestamp]
    return UPL

def run_backtest(ask_prices, bid_prices, signals_array, max_position_usd, commission, delay, sample_size=1000):             
    balance = max_position_usd
    pq = PositionsQueue(delay)
    actives_count = ask_prices.shape[0]
    currencies = np.zeros(actives_count)
    UPLs = np.array([balance])
    for timestamp in range(min(sample_size, signals_array.shape[1])):
        position_amount = min(balance, max_position_usd) / actives_count
        to_handle = pq.step()
        for active_id in range(actives_count):
            if signals_array[active_id][timestamp] == 1:
                currency_amount = position_amount / bid_prices[active_id][timestamp]
                balance -= position_amount
                pq.add(active_id, currency_amount, True)
                currencies[active_id] -= currency_amount
                balance -= commission
            elif signals_array[active_id][timestamp] == -1:
                currency_amount = position_amount / ask_prices[active_id][timestamp]
                balance += position_amount
                pq.add(active_id, currency_amount, False)
                currencies[active_id] += currency_amount
                balance -= commission
            if active_id in to_handle:
                for handling in to_handle[active_id]:
                    amount = handling[0]
                    if handling[1]:
                        currencies[active_id] -= amount
                        balance_change = amount * ask_prices[active_id][timestamp]
                    else:
                        currencies[active_id] += amount
                        balance_change = -amount * bid_prices[active_id][timestamp]
                    balance += balance_change - commission
        tmp_UPL = get_UPL(float(balance), currencies, ask_prices, timestamp)
        UPLs = np.append(UPLs, (tmp_UPL))
    return UPLs

# short_window_size, long_window_size, long_threshold, short_threshold, q
def to_maximaze(x, ask_prices, bid_prices, max_position_usd, commission, delay):
    short_window_size, long_window_size, long_threshold, short_threshold, q = x
    signals_array = generate_signal(ask_prices, bid_prices, long_threshold, short_threshold, int(short_window_size), int(long_window_size), q)
    UPLs = run_backtest(ask_prices, bid_prices, signals_array, max_position_usd, commission, delay)
    sharp_coefficient = sharp_coef(UPLs)
    return -sharp_coefficient

def get_arguments(ask_prices, bid_prices, max_position_usd, commission, delay):
    # short_window_size, long_window_size, long_threshold, short_threshold, q
    # ДО
    # bounds = ((8, 250), (8, 250), (-10000, 10000), (-10000, 10000), (0.4, 1.1))
    # ПОСЛЕ
    bounds = [(10, 200), (20, 200), (-0.1, 0.1), (-0.1, 0.1), (0.1, 0.9)] 
    result = scipy.optimize.differential_evolution(to_maximaze, bounds, args=(ask_prices, bid_prices, max_position_usd, commission, delay), disp=False, polish=False, updating='deferred')
    return result

@njit
def sharp_coef(UPLs):
    if UPLs.std() == 0:
        return 0
    return ((UPLs-UPLs[0]).mean()/UPLs.std())                

def main(directory_name: str):
    max_position_usd = 500
    commission = 2
    delay = 15
    ask_prices, bid_prices = get_data(directory_name, sample_size = 500)
    result = get_arguments(ask_prices, bid_prices, max_position_usd, commission, delay)
    short_window_size, long_window_size, long_threshold, short_threshold, q = result.x
    signals_array = generate_signal(ask_prices, bid_prices, long_threshold, short_threshold, int(short_window_size), int(long_window_size), q)
    UPLs = run_backtest(ask_prices, bid_prices, signals_array, max_position_usd, commission, delay)
    print(-result.fun)
    lineplot_by_y(UPLs)

if __name__ == '__main__':
    DIRECTORY_NAME = 'pickles'
    main(DIRECTORY_NAME)

