import os
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
def get_weights(size, last_weight):
    weights = np.zeros(0)
    weight = last_weight
    for _ in range(size):
        weights = np.append(weights, [weight])
        weight *= 2
    return weights

@njit
def get_ewma(data_array: np.ndarray, window_size: int, last_weight: float):
    length = data_array.shape[0]
    weights = get_weights(window_size, last_weight)
    divider = weights.sum()
    result = np.zeros(0)
    for i in range(length - window_size + 1):
        subsequence = data_array[i:i+window_size]
        new_ewma = (subsequence*weights).sum() / divider
        result = np.append(result, [new_ewma])
    return result

@njit
def generate_signal(ask_prices, bid_prices, long_threshold, short_threshold, window_size, last_weight):
    current_prices = (ask_prices + bid_prices) / 2
    signals_array = np.zeros((current_prices.shape[0], current_prices.shape[1]), np.float64)
    for active_id in range(current_prices.shape[0]):
        active = current_prices[active_id]
        ewmas = get_ewma(active, window_size, last_weight)
        tmp_signals = np.zeros(0)
        ewmas_length = ewmas.shape[0]
        for i in range(active.shape[0]):
            if i >= ewmas_length:
                new_signal = np.zeros(1)  
            else:
                ewma = ewmas[i]
                price = active[i]
                error = price-ewma
                if error > long_threshold:
                    new_signal = np.zeros(1) + 1
                elif error < short_threshold:
                    new_signal = np.zeros(1) - 1
                else:
                    new_signal = np.zeros(1) 
            tmp_signals = np.append(tmp_signals, new_signal)
        signals_array[active_id] = tmp_signals
    return signals_array

def get_UPL(balance, currencies, ask_prices, timestamp):
    UPL = balance
    for active_id in range(ask_prices.shape[0]):
        UPL += currencies[active_id] * ask_prices[active_id][timestamp]
    return UPL

def run_backtest(ask_prices, bid_prices, signals_array, max_position_usd, commission, delay):
    balance = max_position_usd
    sample_size = 1000
    pq = PositionsQueue(delay)
    actives_count = ask_prices.shape[0]
    currencies = [1000] * actives_count
    zero_UPL = get_UPL(balance, currencies, ask_prices, 0)
    UPLs = np.array([zero_UPL])
    for timestamp in range(sample_size):
        position_amount = balance / actives_count
        to_handle = pq.step()
        for active_id in range(actives_count):
            if signals_array[active_id][timestamp] == 1:
                currency_amount = position_amount / bid_prices[active_id][timestamp]
                balance -= position_amount - commission
                pq.add(active_id, currency_amount, True)
                currencies[active_id] -= currency_amount
            elif signals_array[active_id][timestamp] == -1:
                currency_amount = position_amount / ask_prices[active_id][timestamp]
                balance += position_amount - commission
                pq.add(active_id, position_amount, False)
                currencies[active_id] += currency_amount
            if active_id in to_handle:
                for handling in to_handle[active_id]:
                    amount = handling[0]
                    if handling[1]:
                        currencies[active_id] -= amount
                        balance += amount * ask_prices[active_id][timestamp] - commission
                    else:
                        currencies[active_id] += amount
                        balance -= amount * bid_prices[active_id][timestamp] - commission
            tmp_UPL = get_UPL(balance, currencies, ask_prices, timestamp)
            UPLs = np.append(UPLs, (tmp_UPL))
    return UPLs - UPLs[0]                        

def main(directory_name: str):
    max_position_usd = 5000
    commission = 5
    delay = 10
    ask_prices, bid_prices = get_data(directory_name, sample_size = 10000)
    long_threshold = +inf
    short_threshold = -inf
    window_size = 5
    last_weight = 0.25
    signals_array = generate_signal(ask_prices, bid_prices, long_threshold, short_threshold, window_size, last_weight)
    UPLs = run_backtest(ask_prices, bid_prices, signals_array, max_position_usd, commission, delay)
    lineplot_by_y(UPLs)

if __name__ == '__main__':
    DIRECTORY_NAME = 'pickles'
    main(DIRECTORY_NAME)

