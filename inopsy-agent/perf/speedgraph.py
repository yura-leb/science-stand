"""
This program is an .ipybn notebook utility.

This tool is designed to calculate amount of client data sent.
Note, that total packets calculated can be more, than content delivered,
because we calculate content size of packets, that include content.
"""

#%%
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

#%%
def pckt_dict(path, parti, fnd):
    '''Extract pckts per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd)
    d = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[:6]) // parti
        d[time_period] += 1
    return(d)

#%%
def byte_dict(path, parti, fnd):
    '''Extract bytes per second and put in dict'''
    with open(path, 'r') as f:
        text = f.read()
    '''This reg. expr works for less than 10K seconds experiments'''
    patt = re.compile(r"I00(.*)[^I00]*"+fnd+"(.*)\n")
    d = defaultdict(int)
    for i in patt.findall(text):
        time_period = int(i[0][:6]) // parti
        d[time_period] += int(re.search(r" (\d*) bytes", i[1]).group(1))
    return(d)

#%%
# INPUT:
parti = 100         # Partition
statfile = "logsrv" # Name of file

d1 = pckt_dict(statfile, parti, "Sent packet:")
d2 = pckt_dict(statfile, parti, "Received packet:")

# %%
plt.figure(figsize=(12, 7))
plt.plot([parti*i for i in range(len(d1))], [d1.setdefault(i, 0) for i in range(len(d1))], label="Sent")
plt.plot([parti*i for i in range(len(d2))], [d2.setdefault(i, 0) for i in range(len(d2))], label="Received")
plt.xlabel('Timestamp')
plt.ylabel('Packets sent/recieved')
plt.title(f'Packets sent/recieved by [{statfile}] in timestamp I..')
plt.legend()
plt.show()

#%%
# INPUT:
parti = 50          # Partition
statfile = "logsrv" # Name of file

d1 = byte_dict(statfile, parti, "Sent packet:")
d2 = byte_dict(statfile, parti, "Received packet:")

# %%
plt.figure(figsize=(12, 7))
plt.plot([parti*i for i in range(len(d1))], [d1.setdefault(i, 0) for i in range(len(d1))], label="Sent")
plt.plot([parti*i for i in range(len(d2))], [d2.setdefault(i, 0) for i in range(len(d2))], label="Received")
plt.xlabel('Timestamp')
plt.ylabel('Bytes sent/recieved')
plt.title(f'Bytes sent/recieved by [{statfile}] in timestamp I..')
plt.legend()
plt.show()
# %%
