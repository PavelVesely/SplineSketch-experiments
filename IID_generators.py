import numpy as np
import matplotlib.pyplot as plt
import subprocess
import bisect
import math
import uuid
import os


# FUNCTION FOR LOADING THE DATA
###############################

SYNTHETIC_DATA_SIZE_DEFAULT = 1000000 # 10**6
np.random.seed(42)

def signed_lognormal(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    r = np.random.lognormal(mean=0, sigma=1, size=n)
    if np.random.random() < 0.5:
        r = -r
    return r

def lognormal(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.random.lognormal(mean=0, sigma=1, size=n)

def pareto(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.random.pareto(1, n)

def normal(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.random.normal(0, 1, n)

def gumbel(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.random.gumbel(0, 1, n)

LOGUNIFORM_MAX_EXPONENT = 10
LOGUNIFORM_MAX_EXPONENT_EXTREME = 60

def loguniform(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.power(10, (np.random.random(n)-0.5)*2*LOGUNIFORM_MAX_EXPONENT)

def signed_loguniform(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    r = np.power(10, (np.random.random(n)-0.5)*2*LOGUNIFORM_MAX_EXPONENT)
    if np.random.random() < 0.5:
        r = -r
    return r

def signed_loguniform_extreme(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    r = np.power(10, (np.random.random(n)-0.5)*2*LOGUNIFORM_MAX_EXPONENT_EXTREME)
    if np.random.random() < 0.5:
        r = -r
    return r

def uniform(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.random.random(n)*10**6 # the multiplication by 10**6 is perhaps not necessary

########################### INPUTS WITH FEW DISTINCT ELEMENTS, ALL WITH HIGH FREQUENCY ###################

def distinct_values_42(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    distinct_values = [x - 21 for x in range(42)] #1000000*np.random.random(42)
    return [np.random.choice(distinct_values) for _ in range(n)]

def distinct_values_5(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    distinct_values = range(5) #1000000*np.random.random(5)
    return [np.random.choice(distinct_values) for _ in range(n)]

def distinct_values_150(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    distinct_values = 1000000*np.random.random(150)
    return [np.random.choice(distinct_values) for _ in range(n)]

############################ MIXED DISTRIBUTIONS ########################################################

def normal_with_1_large_change(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.concatenate((np.random.normal(0, 1, n//2), np.random.normal(100, 10, n - n//2)))

def normal_with_1_small_change(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    return np.concatenate((np.random.normal(0, 1, n//2), np.random.normal(1, 1, n - n//2)))

def normal_and_distinct_42(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    distinct_values = [x - 21 for x in range(42)] 
    return np.concatenate((np.random.normal(0, 1, n//2), np.random.normal(1, 1, n - n//2), [np.random.choice(distinct_values) for _ in range(n - n//2)]))


############################ SORTED WITH FREQUENT ITEMS ########################################################


def sorted_with_frequent(n=SYNTHETIC_DATA_SIZE_DEFAULT):
    max_freq = max(1, n // 50)
    
    # Estimate number of unique values needed
    # Start with a guess and add until we reach or exceed n items
    unique_values = []
    frequencies = []
    total = 0
    value = 0
    
    while total < n:
        freq = np.random.randint(1, max_freq + 1)
        unique_values.append(value)
        frequencies.append(freq)
        total += freq
        value += 1

    # Trim to exact size
    values = np.repeat(unique_values, frequencies)[:n]
    sorted_array = np.sort(values)

    return sorted_array