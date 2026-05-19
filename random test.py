import os
from platform import machine
import numpy as np
import pandas as pd

PATH = r"C:\Users\grulovicma\Matija Grulovic\GitHub\SimRLFab_mata\log\80 states throughput test\episode_log.csv"
sim_log = pd.read_csv(PATH)

empty_list = [[1,2]]

if not empty_list:
    print("List is empty")
