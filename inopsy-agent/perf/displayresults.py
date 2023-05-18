"""
This program is an .ipybn notebook utility, used to convert special data into excel.

"""

#%%
from __future__ import annotations
from json.tool import main
import re
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import math
import json
import yaml
import os
from collections import defaultdict
sns.set_theme(style="whitegrid")

# %%
'''Define pathes'''

'''Txt files to parse data from'''
path_statfiles = []
'''Dirs with yaml files to parse data from'''
path_statdirs = []

'''data_format_type equals following numbers depending on the task:
1. Speed calculation on channel parameters;
   Data collection advice:   Use inopsy-stand/RunAllExperiments.py. 
2. Speed calculation on channel parameters with BBR parameters changed;
   Data collection advice:   Use RunBBRParamGridSearch.py. 
3. BBR parameters calculation on channel parameters;
   Data collection advice:   Use FullMinimizeBBRParam.py. 
4. Speed graph on a specific directory (can be applied for any type of experiments that include additional_data + content strcucture);
   Data collection advice:   Use inopsy-stand/RunAllExperiments.py.
'''
data_format_type = 2

'''Minimal amount of experiments with each parameter set'''
minimum_exp_count = 3

experiments_max = 2000
if data_format_type == 1:
    algos = ["CUBIC", "RENO", "BBR2"]
    bbrparam = False
    onlyinput = False
    speedisestimated = False
    paths_txt = ["perfres_for_speed/perfres_w3_v2", 
                    "perfres_for_speed/perfres_w3_deviation_p1", 
                    "perfres_for_speed/perfres_w3_deviation_p2", 
                    "perfres_for_speed/perfres_w3_deviation_p3", 
                    "perfres_for_speed/perfres_w3_deviation_p4"] # Experiments for a speed regression
    paths_yaml = ["perfres_for_speed/perfres_w3_deviation_p5"] # Experiments for a speed regression
if data_format_type == 2:
    algos = ["BBR2"]
    bbrparam = True
    onlyinput = False
    speedisestimated = False
    paths_txt = []
    paths_yaml = ["perfres_for_bbr/perfres_w3_bbr_p14/perfres_w3_bbr_p14",
                  "perfres_for_bbr/perfres_w3_bbr_p20_bindp14/perfres_w3_bbr_p20_bindp14",
                  "perfres_for_bbr/perfres_w3_bbr_p21_bindp14/perfres_w3_bbr_p21_bindp14",
                  "perfres_for_bbr/perfres_w3_bbr_p22_bindp14/perfres_w3_bbr_p22_bindp14",
                  "perfres_for_bbr/perfres_w3_bbr_p23_bindp14/perfres_w3_bbr_p23_bindp14"] # Long experiments on a standart agent and stand
    # paths_yaml = ["perfres_for_bbr/perfres_w3_bbr_p16/perfres_w3_bbr_p16"] # Long experiments on a march-agent and august-stand (server-client) made to assume
    # paths_yaml = ["perfres_for_bbr/perfres_w3_bbr_p3",
    #                  "perfres_for_bbr/perfres_w3_bbr_p7",
    #                  "perfres_for_bbr/perfres_w3_bbr_p8"]
if data_format_type == 3:
    algos = ["BBR2"]
    bbrparam = True
    onlyinput = True
    speedisestimated = True
    paths_txt = []
    paths_yaml = ["perfres_for_bbr_minimize/perfresBBRv2Fri_Jan_6_02.37.56_2023.optimal",
                  "perfres_for_bbr_minimize/perfresBBRv2Fri_Jan_13_12.32.26_2023.optimal",
                  "perfres_for_bbr_minimize/perfresBBRv2Mon_Jan_23_18.00.21_2023.optimal",
                  "perfres_for_bbr_minimize/perfresBBRv2Fri_Jan_27_12.38.48_2023.optimal",
                  "perfres_for_bbr_minimize/perfresBBRv2Thu_Feb__2_20.06.57_2023.optimal"] # Experiments on Powell minimization for regression 40 seconds
if data_format_type == 4:
    algos = ["BBR2"]
    with_averaging = False
    goodness_tresh = 299
    paths_yaml = ["perfres_for_bbr/perfres_w3_bbr_p14/perfres_w3_bbr_p14",
                  "perfres_for_bbr/perfres_w3_bbr_p20_bindp14/perfres_w3_bbr_p20_bindp14"] # Long experiments on a standart agent and stand

'''Path to json file, where we save parsed data'''
path_savejson = "../../../../mean_deviation_ML.json"
'''Path to directory with xlsx files'''
path_savexlsxs = "cc_perf_tests"

#%%
'''Functions, that process files with multiple statlog.py runs outputs into data.'''
def convert_speed(fl):
    '''Converts big float speeds into 2^n format string'''
    if fl >= 1024**3//8:
        return f"{int(fl*8//(1024**3))} Gbit/s"
    if fl >= 1024**2//8:
        return f"{int(fl*8//(1024**2))} Mbit/s"
    if fl >= 1024//8:
        return f"{int(fl*8//(1024))} Kbit/s"

def convert_speed_to_mbit(fl):
    '''Converts big float speeds into Mbit format'''
    return int(fl*8//(1024**2))

def convert_speed_to_kbit(fl):
    '''Converts big float speeds into Kbit format'''
    return int(fl*8//1024)

def calculate_accuracy(th, pr):
    '''Calculates accuracy'''
    return math.fabs((th - pr) / th)

def get_data_txt(paths, algos, speed, formulas):
    '''
    Analyse experiments results:

BBR2experiment. p_rtt: 70 p_loss: 0.25 p_bw: 415 p_jt: 1
Estimating speed in BYTE mode:
Second 0.0-1.0:     bytes sent: 1480700     mean s_rtt: 120.921182  mean jitter: 14.596059  mean loss: 0.045594     mean loss2: 0.07923     mean max_bw: 3607256.162562 
Second 1.0-2.0:     bytes sent: 1399200     mean s_rtt: 103.263158  mean jitter: 19.285714  mean loss: 0.190058     mean loss2: 0.106186    mean max_bw: 4173011.714286 
General mean values by each 1.0 second:
                    bytes sent: 1495208.852459 mean s_rtt: 79.015855   mean jitter: 6.628816   mean loss: 0.110859     mean loss2: 0.084434    mean max_bw: 2602494.440661 
    '''
    data = []
    bw_unit_str = " Kbit/s"
    maintext = ""
    for path in paths:
        with open(path, 'r') as f:
            text = f.read()
        maintext += text
    patt = re.compile(r"("+"|".join(algos)+r")experiment. (.*)(\n|.)*?(\d*)-(.*)\nGeneral mean values by each (.*)\n(.*)\n")
    good_data_amount = 0
    bad_data_amount = 0
    loss1_accuracy = 0
    loss2_accuracy = 0
    compound_loss_accuracy = 0
    formulas_accuracy = [0]*6
    for path in paths:
        with open(path, 'r') as f:
            text = f.read()
        for i in patt.findall(text):
            '''Analyse a single experiment'''
            data_sample = []
            data_sample.append(i[0])
            '''Add parameters'''
            theor_rtt = float(i[1].split()[1])
            data_sample.append(theor_rtt)
            theor_loss_percent = float(i[1].split()[3])
            data_sample.append(theor_loss_percent)
            theor_speed_in_kbit = float(i[1].split()[5]) * 1024
            data_sample.append(theor_speed_in_kbit)
            data_sample.append(float(i[1].split()[7])) # Channel jitter (ms)
            if (speed):
                '''Add channel speed to statistics'''
                data_sample.append(i[1].split()[9])
            parti = float(i[5].split()[0])
            real_speed_in_kbit = convert_speed_to_kbit(float(i[6].split()[2]) / parti)
            data_sample.append(real_speed_in_kbit)
            real_rtt = float(i[6].split()[5])
            data_sample.append(real_rtt)
            real_loss1 = float(i[6].split()[11])
            data_sample.append(real_loss1)
            real_loss2 = float(i[6].split()[14])
            data_sample.append(real_loss2)
            data_sample.append(float(i[6].split()[8])) # Sender jitter (ms)
            experiment_duration = float(i[4].split()[0][:-1])
            data_sample.append(experiment_duration)
            '''Skip if bad data (experiment was too fast). 
            Minimum experiment duration is currently 1min, accuracy is 1 second.'''
            if (((real_loss1 == 0) or (real_loss2 == 0)) and (formulas)) or (experiment_duration < 59):
                bad_data_amount += 1
                continue
            else:
                good_data_amount += 1
            '''Theoretical formulas from 1-stage report'''
            theor_square_root_bw = 8 * (3/2)**(1/2) / ((theor_rtt/1000) * (theor_loss_percent/100)**(1/2)) / 1024**2 * 1200
            if (real_loss1 != 0):
                cubic_formula_bw_1 = 8 * 1.05 * (1 / (((real_rtt/1000) * (real_loss1/100)**3))) ** (1/4) / 1024**2 * 1200
                prac_square_root_bw_1 = 8 * (3/2)**(1/2) / ((real_rtt/1000) * (real_loss1/100)**(1/2)) / 1024**2 * 1200
            else:
                prac_square_root_bw_1 = prac_square_root_bw_2 = 0
            cubic_formula_bw = 8 * 1.05 * (1 / (((theor_rtt/1000) * (theor_loss_percent/100)**3))) ** (1/4) / 1024**2 * 1200
            if (real_loss2 != 0):
                cubic_formula_bw_2 = 8 * 1.05 * (1 / (((real_rtt/1000) * (real_loss2/100)**3))) ** (1/4) / 1024**2 * 1200
                prac_square_root_bw_2 = 8 * (3/2)**(1/2) / ((real_rtt/1000) * (real_loss2/100)**(1/2)) / 1024**2 * 1200
            else:
                cubic_formula_bw_1 = cubic_formula_bw_2 = 0
            if (formulas):
                '''Add formulas data to statistics'''
                data_sample.append(str(round(theor_square_root_bw, 3)) + bw_unit_str)
                data_sample.append(str(round(prac_square_root_bw_1, 3)) + bw_unit_str)
                data_sample.append(str(round(prac_square_root_bw_2, 3)) + bw_unit_str)
                data_sample.append(str(round(cubic_formula_bw, 3)) + bw_unit_str)
                data_sample.append(str(round(cubic_formula_bw_1, 3)) + bw_unit_str)
                data_sample.append(str(round(cubic_formula_bw_2, 3)) + bw_unit_str)
            '''Calculate accuracy'''
            if real_loss1 != 0:
                loss1_accuracy += calculate_accuracy(float(i[1].split()[3]), real_loss1)
            if real_loss2 != 0:  
                loss2_accuracy += calculate_accuracy(float(i[1].split()[3]), real_loss2)
            if real_loss2 + real_loss1 != 0:
                compound_loss_accuracy += calculate_accuracy(float(i[1].split()[3]), (real_loss1 + real_loss2) / 2)
            if (formulas):
                formulas_accuracy[0] += calculate_accuracy(real_speed_in_kbit, theor_square_root_bw)
                formulas_accuracy[1] += calculate_accuracy(real_speed_in_kbit, prac_square_root_bw_1)
                formulas_accuracy[2] += calculate_accuracy(real_speed_in_kbit, prac_square_root_bw_2)
                formulas_accuracy[3] += calculate_accuracy(real_speed_in_kbit, cubic_formula_bw)
                formulas_accuracy[4] += calculate_accuracy(real_speed_in_kbit, cubic_formula_bw_1)
                formulas_accuracy[5] += calculate_accuracy(real_speed_in_kbit, cubic_formula_bw_2)
            
            data.append(data_sample)

    print("Good data samples amount: ", good_data_amount, " bad: ", bad_data_amount)
    good_data_amount = 1 if good_data_amount == 0 else good_data_amount
    return data, loss1_accuracy, loss2_accuracy, compound_loss_accuracy, formulas_accuracy, bad_data_amount / good_data_amount

def get_data_yaml_dft1(paths, algos, speed=False, bbrparam=False, minimum_goodness_tresh=290):
    '''
    Each file in each dir is:

{additional_info: {channel_bw: 375, channel_congestion_control: bbr2, channel_jitt: 1,
    channel_loss: 0.4, channel_rtt: 89}, content: [{'bytes sent: ': 765238, 'mean cwnd: ': 95540.631179,
      'mean jitter: ': 0.802281, 'mean loss2: ': 0.003118, 'mean loss: ': 0.464879,

        ...

      'mean loss: ': 0.395708, 'mean s_rtt: ': 89.0, time: 5.0-6.0}, {'bytes sent: ': 1391159.833333,
      'mean cwnd: ': 297459.917504, 'mean jitter: ': 0.133714, 'mean loss2: ': 0.001569,
      'mean loss: ': 1.338168, 'mean s_rtt: ': 88.200832, time: global}]}
    '''
    data = []
    good_data_amount = 0
    bad_data_amount = 0
    for directory in paths:
        for filename in os.listdir(directory):
            dir_file = os.path.join(directory, filename)
            if os.path.isfile(dir_file) and filename not in ['TooLongERRORs', 'OtherERRORs']:
                try:
                    with open(dir_file, 'r') as f:
                        '''Analyse a single experiment'''
                        data_sample = []
                        datadict = yaml.safe_load(f)
                        '''Add parameters'''
                        conj_c = datadict['additional_info']['channel_congestion_control'].upper()
                        theor_rtt = datadict['additional_info']['channel_rtt']
                        theor_loss_percent = datadict['additional_info']['channel_loss']
                        theor_speed_in_kbit = datadict['additional_info']['channel_bw'] * 1024
                        theor_jitt = datadict['additional_info']['channel_jitt']
                        data_sample.extend([conj_c, theor_rtt, theor_loss_percent, theor_speed_in_kbit, theor_jitt])
                        if (speed):
                            '''Add channel speed to statistics'''
                            data_sample.append(datadict['additional_info']['agent_speed'])
                        parti = float(datadict['content'][0]['time'].split('-')[1]) - float(datadict['content'][0]['time'].split('-')[0])
                        '''The statlog.py puts global data in the end of the list => [-1]'''
                        real_speed_in_kbit = convert_speed_to_kbit(datadict['content'][-1]['bytes sent: '] / parti)
                        real_rtt = datadict['content'][-1]['mean s_rtt: ']
                        real_loss1 = datadict['content'][-1]['mean loss: ']
                        real_loss2 = datadict['content'][-1]['mean loss2: ']
                        real_jitt = datadict['content'][-1]['mean jitter: ']
                        experiment_duration = float(datadict['content'][-2]['time'].split('-')[1])
                        data_sample.extend([real_speed_in_kbit, real_rtt, real_loss1, real_loss2, real_jitt, experiment_duration])
                        if (bbrparam):
                            bbr_beta = datadict['additional_info']['bbr_beta']
                            bbr_loss_tresh = datadict['additional_info']['bbr_loss_tresh']
                            bbr_probe_rtt_cwnd_gain = datadict['additional_info']['bbr_probe_rtt_cwnd_gain']
                            bbr_probe_rtt_duration = datadict['additional_info']['bbr_probe_rtt_duration']
                            data_sample.extend([bbr_beta, bbr_loss_tresh, bbr_probe_rtt_cwnd_gain, bbr_probe_rtt_duration])
                    '''Skip if bad data (experiment was too fast).'''
                    if (experiment_duration < minimum_goodness_tresh):
                        bad_data_amount += 1
                        continue
                    else:
                        good_data_amount += 1
                    
                    log_output_interval = 100
                    if (good_data_amount % log_output_interval == 0):
                        print("Current progress..: ", good_data_amount)
                    if (bad_data_amount % log_output_interval == 0):
                        print("BAD!: ", bad_data_amount)
                    
                    data.append(data_sample)
                except Exception as E:
                    print(E)

    print("Good data samples amount: ", good_data_amount, " bad: ", bad_data_amount)
    good_data_amount = 1 if good_data_amount == 0 else good_data_amount
    return data, bad_data_amount / good_data_amount

def get_data_yaml_dft3(paths):
    '''
    Each file in each dir is:

[{bbr_beta: 0.6390096630005888, bbr_loss_tresh: 0.038541019662496845, bbr_probe_rtt_cwnd_gain: 0.6704355065429269,
    bbr_probe_rtt_duration: 114.95627540454254, channel_bw: 211, channel_congestion_control: bbr2,
    channel_jitt: 1, channel_loss: 0.005, channel_rtt: 47, estimated_opt_speed: -23746734.861538403,

        ...

    channel_jitt: 1, channel_loss: 0.3, channel_rtt: 63, estimated_opt_speed: -17702643.98076917,
    res_nfev: 92.0}]
    '''
    data = []
    for filename in paths:
        if filename not in ['TooLongERRORs', 'OtherERRORs']:
            continue
        else:
            with open(filename, 'r') as f:
                datalist = yaml.safe_load(f)
                for datadict in datalist:
                    '''Analyse a single experiment'''
                    data_sample = []
                    '''Add parameters'''
                    conj_c = datadict['channel_congestion_control'].upper()
                    theor_rtt = float(datadict['channel_rtt'])
                    theor_loss_percent = float(datadict['channel_loss'])
                    theor_speed_in_kbit = float(datadict['channel_bw'] * 1024)
                    theor_jitt = float(datadict['channel_jitt'])
                    estimated_opt_speed = convert_speed_to_kbit(- datadict['estimated_opt_speed'])
                    res_nfev = datadict['res_nfev']
                    bbr_beta = datadict['bbr_beta']
                    bbr_loss_tresh = datadict['bbr_loss_tresh']
                    bbr_probe_rtt_cwnd_gain = datadict['bbr_probe_rtt_cwnd_gain']
                    bbr_probe_rtt_duration = datadict['bbr_probe_rtt_duration']
                    data_sample = [conj_c, theor_rtt, theor_loss_percent, theor_speed_in_kbit,
                                   theor_jitt, estimated_opt_speed, res_nfev, bbr_beta,
                                   bbr_loss_tresh, bbr_probe_rtt_cwnd_gain, bbr_probe_rtt_duration]

                    data.append(data_sample)

    return data

def get_data_yaml_dict_times(paths, goodness_tresh=299, experiments_max=100000, with_averaging=True):
    '''
    Each file in each dir is:

{additional_info: {bbr_beta: 0.8, bbr_loss_tresh: 0.05, bbr_probe_rtt_cwnd_gain: 0.8,
    bbr_probe_rtt_duration: 200, channel_bw: 180, channel_congestion_control: bbr2,
    channel_jitt: 1, channel_loss: 1, channel_rtt: 40}, content: [{'bytes sent: ': 2439059,
      'mean cwnd: ': 245635.52422, 'mean jitter: ': 0.277538, 'mean loss2: ': 0.005505,

        ...

      'mean jitter: ': 0.285098, 'mean loss2: ': 0.002163, 'mean loss: ': 0.206585,
      'mean s_rtt: ': 46.20917, time: global}]}

    Additional information must be same on each file => Otherwise dict keys decomposition is not guaranteed.
    '''
    data = dict()
    data_keys = []
    good_data_amount = 0
    bad_data_amount = 0
    exp_count = 0
    for directory in paths:
        for filename in os.listdir(directory):
            if exp_count >= experiments_max:
                break
            dir_file = os.path.join(directory, filename)
            if os.path.isfile(dir_file) and filename not in ['TooLongERRORs', 'OtherERRORs']:
                exp_count += 1
                with open(dir_file, 'r') as f:
                    '''Analyse a single experiment'''
                    datadict = yaml.safe_load(f)

                    experiment_duration = float(datadict['content'][-2]['time'].split('-')[1])
                    '''Skip if bad data (experiment was too fast).'''
                    if (experiment_duration < goodness_tresh):
                        bad_data_amount += 1
                        continue
                    else:
                        good_data_amount += 1
                    
                    log_output_interval = 100
                    if (good_data_amount % log_output_interval == 0):
                        print("Current progress..: ", good_data_amount)
                    if (bad_data_amount % log_output_interval == 0):
                        print("BAD!: ", bad_data_amount)

                    if with_averaging:
                        data_key = tuple(datadict['additional_info'].values())
                        cur_data_keys = list(datadict['additional_info'].keys())
                    else:
                        data_key = tuple(list(datadict['additional_info'].values()) + [exp_count])
                        cur_data_keys = list(datadict['additional_info'].keys()) + ["Number of experiment"]
                    if data_keys == []:
                        data_keys = cur_data_keys
                    elif len(data_keys) != len(cur_data_keys):
                        print("WARNING: 'additional_data' fields are not the same in files given!")
                    '''Create a defaultdict() for each experiment type (varies by data_key)'''
                    if data_key not in data:
                        data[data_key] = defaultdict(int)
                    data[data_key]['runs_count'] += 1
                    speed_array = np.array([sample['mean cwnd: '] for sample in datadict['content'][:(goodness_tresh - 1)]])
                    if 'speed_graph' in data[data_key].keys():
                        data[data_key]['speed_graph'] += speed_array
                    else:
                        data[data_key]['speed_graph'] = speed_array

    for key in data.keys():
        data[key]['speed_graph'] = data[key]['speed_graph'] / data[key]['runs_count']
    return data, data_keys

def get_data_columns(speed=False, formulas=False, onlyinput=False, speedisestimated=False, bbrparam=False):
    col = ["Congestion Controller", "Channel RTT (ms)", "Channel Loss (%)", "Channel BW (Kbit/s)",
           "Channel Jitter (ms)"]
    if (speed):
        col.append("Channel maximum speed (Kbit/s)")
    if (not onlyinput):
        col += ["Sender Speed (Kbit/s)", "Sender RTT (ms)", "Sender lost data to data inflight (%)",
                "Sender lost data to data sent (%)", "Sender Jitter (ms)", "Experiment duration (sec)"]
    if (speedisestimated):
        col += ["Optimal Speed (Kbit/s)", "Estimation iterations number"]
    if (formulas):
        col += ["F: Theoretical Square Root", "F: Practical Square Root on loss1", "F: Practical Square Root on loss2", 
                "F: Theoretical CUBIC BW Formula", "F: Practical CUBIC BW Formula on loss1", "F: Practical CUBIC BW Formula on loss2"]
    if (bbrparam):
        col += ["BBR Beta", "BBR Losstresh", "BBR Probe RTT Cwnd Gain", "BBR Probe RTT Duration"]

    return col


# %%
'''Put files needed to be composed into a dataframe in statfiles, then specify algos, that we want to analyse.'''
if __name__ == "__main__":

    if data_format_type in [1, 2]:
        data_lists, bad_data_fraction1 = get_data_yaml_dft1(paths_yaml, algos, bbrparam=bbrparam, minimum_goodness_tresh=30)
        data_lists_old, loss1_accuracy, loss2_accuracy, compound_loss, formulas_accuracy, bad_data_fraction2 = get_data_txt(paths_txt, algos, speed=False, formulas=False)
        data_lists += data_lists_old
        print('Bad data fractions: ', bad_data_fraction1, bad_data_fraction2)

        col = get_data_columns(onlyinput=onlyinput, speedisestimated=speedisestimated, bbrparam=bbrparam)
        df = pd.DataFrame(data_lists, columns=col)
    elif data_format_type in [3]:
        data_lists = get_data_yaml_dft3(paths_yaml)
    
        col = get_data_columns(onlyinput=onlyinput, speedisestimated=speedisestimated, bbrparam=bbrparam)
        df_t3 = pd.DataFrame(data_lists, columns=col)
    elif data_format_type in [4]:
        data_dict, data_dict_key_semantics = get_data_yaml_dict_times(paths_yaml, experiments_max=experiments_max, goodness_tresh=goodness_tresh, with_averaging=with_averaging)

# %%
'''If we want to group by something and calculate means'''
if __name__ == "__main__":
    '''Remove experiments, which don't have duplicated, otherwise deviation would be zero'''
    if data_format_type in [1, 2]:
        group_features = ["Congestion Controller", "Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)",
            "Channel Jitter (ms)"]
        if data_format_type == 2:
            group_features += ["BBR Beta", "BBR Losstresh", "BBR Probe RTT Cwnd Gain", "BBR Probe RTT Duration"]
        filtered_df = df.groupby(by=group_features, as_index=False).filter(lambda x: x['Sender Speed (Kbit/s)'].count() >= minimum_exp_count)
        '''Calculate mean value'''
        df2 = filtered_df.groupby(by=group_features, as_index=False).mean()
        '''Calculate min value'''
        df5 = filtered_df.groupby(by=group_features, as_index=False).min()
        '''Calculate max value'''
        df6 = filtered_df.groupby(by=group_features, as_index=False).max()
        '''Calculate Standard deviation'''
        df3 = filtered_df.groupby(by=group_features, as_index=False).std(ddof=1)
        df7 = filtered_df.groupby(by=group_features, as_index=False).count()

        df4 = df2.assign(MinimalSpeed=df5['Sender Speed (Kbit/s)']).assign(MaximalSpeed=df6['Sender Speed (Kbit/s)']).assign(Deviation=df3['Sender Speed (Kbit/s)']).assign(Runs_Count=df7['Sender Speed (Kbit/s)'])
        df4['Speed deviation percent (%)'] = df4['Deviation'] / df4['Sender Speed (Kbit/s)'] * 100

        df_featured = df4

        if data_format_type == 2:
            df_featured = df_featured.loc[:, ['Channel RTT (ms)', 'Channel Loss (%)', 'Channel BW (Kbit/s)', 'BBR Beta', 'BBR Losstresh', 'BBR Probe RTT Cwnd Gain', 'BBR Probe RTT Duration', 'Sender Speed (Kbit/s)', 'Experiment duration (sec)', 'Runs_Count']]
            df_featured = df_featured.set_index(['Channel RTT (ms)', 'Channel Loss (%)', 'Channel BW (Kbit/s)', 'BBR Losstresh']).sort_index()
    
        print(df_featured)    
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''BBR Parameters experiments: Explore data'''
def explore_bbr_params_in_one_dot(data, dot, channel_features, bbr_features, default_features, ymin, ymax):
    '''Ensure, that there is only one dot in data'''
    for feature in channel_features:
        assert data[feature].value_counts().shape[0] == 1

    fig, axes = plt.subplots(4, 4, figsize=(34, 34))
    fig.suptitle('Dependencies between BBRv2 features in dot: [' + str(dot) + '].')

    for i in range(len(bbr_features)):
        for j in range(len(bbr_features)):
            if i == j:
                continue
            axes[i, j].set_title(f'Speed by {bbr_features[i]}, with changing {bbr_features[j]}')
            data_x_ = data
            for k in range(len(bbr_features)):
                pass
            x_ = data_x_[bbr_features[i]]
            y_ = data_x_['Sender Speed (Kbit/s)']
            sns.lineplot(ax=axes[i, j], data=data_x_, x=bbr_features[i], y=f"Sender Speed (Kbit/s)", hue=bbr_features[j], ci=50)
            axes[i, j].set(ylim=(ymin, ymax))

if __name__ == "__main__":
    if data_format_type == 2:
        channel_features = ["Congestion Controller", "Channel RTT (ms)", 'Channel Loss (%)', "Channel BW (Kbit/s)", "Channel Jitter (ms)"]
        bbr_features = ["BBR Beta", "BBR Losstresh", "BBR Probe RTT Cwnd Gain", "BBR Probe RTT Duration"]
        default_features = {
            "BBR Beta": 0.7,
            "BBR Losstresh": 0.02,
            "BBR Probe RTT Cwnd Gain": 0.5,
            "BBR Probe RTT Duration": 200
        }
        dots = list(df[channel_features].value_counts().index)
        tmp = 0
        for dot in dots:
            df_dot = df
            tmp += 1
            for feature in channel_features:
                df_dot = df_dot.loc[(df_dot[feature] == dot[channel_features.index(feature)])]
            if tmp >= 0:
                print("Dot:", dot, "Shape:", df_dot.shape)
                # print(df_dot.groupby(by=bbr_features, as_index=False).mean())
                explore_bbr_params_in_one_dot(df_dot, dot, channel_features, bbr_features, default_features,
                    df_dot.groupby(by=bbr_features, as_index=False).mean()['Sender Speed (Kbit/s)'].min(),
                    df_dot.groupby(by=bbr_features, as_index=False).mean()['Sender Speed (Kbit/s)'].max(),
                )
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''See how speed changes depending on the time'''
def draw_speed_graphs(data_dict, data_dict_key_semantics, graphs_max=5):
    count = -1
    plt.figure(figsize=(25, 10))
    for key, value in data_dict.items():
        count += 1
        if count == graphs_max:
            break
        sns.lineplot(data=data_dict[key]['speed_graph'][:300], label=" ".join([f"{i[0]}: {i[1]}," for i in zip(data_dict_key_semantics, key)]) + f"    Runs_Count: {data_dict[key]['runs_count']}")

if __name__ == "__main__":
    if data_format_type == 4:
        draw_speed_graphs(data_dict, data_dict_key_semantics)
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''Inspect current mean deviation percent, for tactical purposes'''
if __name__ == "__main__":
    if data_format_type in [2]:
        print(df_featured['Speed deviation percent (%)'].mean())
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''Correlation examination 1'''
if __name__ == "__main__":
    if data_format_type in [1]:
        plt.figure(figsize=(25, 20))
        sns.heatmap(df4.corr(method='spearman'), annot=True)
        # df_featured[['Channel BW (Kbit/s)', 'Channel Loss (%)', 'Channel RTT (ms)', 'Deviation', 'Runs_Count']].corr(method='spearman')

# %%
'''Put results in excel'''
import openpyxl

if __name__ == "__main__":
    '''Save results in a xlsx'''
    if data_format_type in [1]:
        df_featured.to_excel(f"{path_savexlsxs}/{'_'.join(algos)}_mean_deviation_ML_v5.xlsx")
    elif data_format_type in [2]:
        df_featured.to_excel(f"{path_savexlsxs}/For_theoretics_v10_5min_loss1.xlsx")
    elif data_format_type in [3]:
        df_t3.to_excel(f"{path_savexlsxs}/{'_'.join(algos)}_Estimated_BBRParameters_v2.xlsx")
    elif data_format_type in [4]:
        data_dict_key_semantics_tmp = data_dict_key_semantics + ["Runs_count"] + [f"{i}-{i+1} second." for i in range(goodness_tresh - 1)]
        data_list_tmp = []
        for key, value in data_dict.items():
            data_list_tmp_sample = list(key) + [data_dict[key]['runs_count']] + list(data_dict[key]['speed_graph'][:(goodness_tresh - 1)])
            data_list_tmp.append(data_list_tmp_sample)
        df_tmp = pd.DataFrame(data_list_tmp, columns=data_dict_key_semantics_tmp)
        xlsx_name = "Speedchange_"
        if with_averaging:
            xlsx_name += "averaging_"
        else:
            xlsx_name += "noaveraging_"
        df_tmp.to_excel(f"{path_savexlsxs}/{'_'.join(algos)}_{xlsx_name}v1.xlsx")
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''DEPRECATED - We execute this cell if we want to convert dataframe to dict and put in json for technical purposes.'''
if __name__ == "__main__":
    # with open(path_savejson, 'w') as f:
    #     json.dump(df_featured.to_dict(), f, indent = 6)
    pass

# %%
'''DEPRECATED - In this cell we compare how two loss methods perform'''
if __name__ == "__main__":
    print("Loss has an error of ", loss1_accuracy, "\nLoss2 has an error of ", loss2_accuracy, "\nCompound error: ", compound_loss)
    print("")
    for ind, i in enumerate(formulas_accuracy):
        print(f"Formula {ind}, accuracy: {i}")
    print("-"*20, f"\nACCURACY: BEST: 0 <- WORST: {len(data_lists)}")

# %%
'''Inspect data distribution (Here we can change filters for df2)'''
if __name__ == "__main__":
    if data_format_type in [1]:
        A = df.loc[(df['Congestion Controller'] == 'BBR2')]
        A = A.loc[(A['Channel BW (Kbit/s)'] <= 302144) & (A['Channel BW (Kbit/s)'] >= 80992)]
        A = A.loc[(A['Channel RTT (ms)'] <= 60) & (A['Channel RTT (ms)'] >= 40)]
        A = A.loc[(A['Channel Loss (%)'] <= 1.5) & (A['Channel Loss (%)'] >= 1.5)]
        plt.figure(figsize=(15, 10))
        plt.scatter(A['Channel BW (Kbit/s)'], A['Sender Speed (Kbit/s)'], lw=2)
        plt.xlabel('Channel BW (Kbit/s)')
        plt.ylabel('Sender Speed (Kbit/s)')
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

# %%
'''NOTE: If both df and df_t3 initialized
Having estimated best BBR parameters, and speed, corresponding them, compare it with observed speed on BBR with standard parameters'''
if __name__ == "__main__":
    if data_format_type in [1, 2]:
        filters = ["Congestion Controller", 'Channel Loss (%)', "Channel BW (Kbit/s)", "Channel Jitter (ms)"]
        i1 = df_featured.set_index(filters).index
        i2 = df_t3.set_index(filters).index

        df_found_similar = df_t3[i2.isin(i1)]
        df_featured_similar = df_featured[i1.isin(i2)]
        df_found_similar = df_found_similar.sort_values(by=filters).reset_index(drop=True)
        df_featured_similar = df_featured_similar.sort_values(by=filters).reset_index(drop=True)
        df_found_similar['Sender Speed STANDARD PARAMS (Kbit/s)'] = df_featured_similar['Sender Speed (Kbit/s)']

        print(df_found_similar)
    else: 
        print(f"Current task/format is {data_format_type} => skipping")

