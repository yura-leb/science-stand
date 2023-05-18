'''
This utility is second version of speedlog.py, that we use to calculate statistics,
if ngtcp2 server uses --inopsy-log flag.
'''
import re
import sys
from collections import defaultdict
import argparse
import json
import yaml
    

def parse_stat(path, parti, args):
    '''Extract statistics from the log file'''
    with open(path, 'r') as f:
        lines = f.readlines()
    lastts = 0
    parti_high = parti
    data_dict = defaultdict(int)
    loss2_dict_lost = defaultdict(float)
    loss2_dict_delv = defaultdict(float)
    dicts = dict(defaultdict(int))
    dividends = ['cwnd', 'srtt', 'mrtt', 'lrtt', 'jitt', 'loss']
    numbers = defaultdict(int)
    firstelem = ''
    for elemtype in dividends:
        dicts[elemtype] = defaultdict(float)
    for line in lines:
        elems = line.split()
        if elems[0][0] == "!":
            "The statistical element has a structure of: '!{name}:{digit}'."
            firstelem = elems[0].split(":")
            parti_high_prev = parti_high - parti
            if firstelem[0] == "!TSMP_I":
                lastts = int(firstelem[1])
                if lastts > parti_high:
                    for elemtype in dividends:
                        if getattr(args, elemtype):
                            if numbers[elemtype] != 0:
                                dicts[elemtype][parti_high_prev] = round(dicts[elemtype][parti_high_prev] / numbers[elemtype], 6)
                            numbers[elemtype] = 0
                    if args.loss2:
                        if loss2_dict_delv[parti_high_prev] != 0:
                            loss2_dict_lost[parti_high_prev] = round(loss2_dict_lost[parti_high_prev] / loss2_dict_delv[parti_high_prev], 6)
                    parti_high += parti
            elif firstelem[0] == "!SentBytes" and args.direction.upper() == "SENT":
                if args.mode.upper() == "BYTE":
                    data_dict[parti_high_prev] += int(firstelem[1])
                elif args.mode.upper() == "PCKT":
                    data_dict[parti_high_prev] += 1
                else:
                    sys.exit("statlog.py:\nNo such mode")
            elif firstelem[0] == "!ReceivedBytes" and args.direction.upper() == "RCVD":
                if args.mode.upper() == "BYTE":
                    data_dict[parti_high_prev] += int(firstelem[1])
                elif args.mode.upper() == "PCKT":
                    data_dict[parti_high_prev] += 1
                else:
                    sys.exit("statlog.py:\nNo such mode")
            elif firstelem[0] == "!CcCwnd" and args.cwnd:
                dicts['cwnd'][parti_high_prev] += int(firstelem[1])
                numbers['cwnd'] += 1
            elif firstelem[0] == "!latest_rtt":
                instrct = ['lrtt', 'mrtt', 'srtt', 'jitt']
                for idx, elemtype in enumerate(instrct):
                    if getattr(args, elemtype):
                        dicts[elemtype][parti_high_prev] += int(elems[idx].split(":")[1])
                        numbers[elemtype] += 1
            elif firstelem[0] == "!loss2" and args.loss:
                dicts['loss'][parti_high_prev] += float(firstelem[1])
                numbers['loss'] += 1
            elif firstelem[0] == "!rs->lost" and args.loss2:
                '''Inspired by https://www.sciencedirect.com/topics/computer-science/packet-loss-rate'''
                loss2_dict_lost[parti_high_prev] += float(firstelem[1])
                loss2_dict_delv[parti_high_prev] += float(elems[2].split(":")[1])
            else:
                '''Wrong statistical elem starting with "!" found!'''
                pass
    if firstelem[0] != "!TSMP_I" or int(firstelem[1]) < parti_high:
        for elemtype in dividends:
            if getattr(args, elemtype):
                if numbers[elemtype] != 0:
                    dicts[elemtype][parti_high_prev] = round(dicts[elemtype][parti_high_prev] / numbers[elemtype], 6)
                numbers[elemtype] = 0
        if args.loss2:
            if loss2_dict_delv[parti_high_prev] != 0:
                loss2_dict_lost[parti_high_prev] = round(loss2_dict_lost[parti_high_prev] / loss2_dict_delv[parti_high_prev], 6)
    return [data_dict, dicts['srtt'], dicts['lrtt'], dicts['mrtt'], dicts['jitt'], dicts['loss'], loss2_dict_lost, dicts['cwnd']]


def indented(val: str, indent: int = 16):
    num_spaces = indent - len(val)
    if num_spaces <= 0:
        return val + " "
    return val + " " * num_spaces


arg_parser = argparse.ArgumentParser(prog='statlog (speedlog2)',
                                     description='Calculate amount of client data sent. ' 
                                     'Note, that total packets calculated can be more, than content delivered, '
                                     'because we calculate content size of packets, that include content. ')                                     
arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.1')
arg_parser.add_argument('--lrtt', action='store_true', 
                        help='Use this parameter if you want mean latest rtt to be calculated. '
                             'Latest rtt is rtt from each packet. '
                             'It is better to work with srtt, because we dont include ack delays there')
arg_parser.add_argument('--srtt', action='store_true', 
                        help='Use this parameter if you want mean smoothed rtt to be calculated. '
                             'By default srtt is calculated in agent with 7/8 of previous value, and 1/8 of new '
                             'depending on pckts recieved')
arg_parser.add_argument('--mrtt', action='store_true', 
                        help='Use this parameter if you want mean min rtt to be calculated'
                             'Min rtt is congestion controller estimate on what min_rtt is')
arg_parser.add_argument('--cwnd', action='store_true', 
                        help='Use this parameter if you want mean congestion window in bytes to be calculated. ')
arg_parser.add_argument('--jitt', action='store_true', 
                        help='Use this parameter if you want mean jitter to be calculated')
arg_parser.add_argument('--loss', action='store_true', 
                        help='Use this parameter if you want mean loss percent to be calculated. ')
arg_parser.add_argument('--loss2', action='store_true', 
                        help='Experimental version of loss percent calculation. ')
arg_parser.add_argument('--json', dest='json', type=str, default='', 
                        help='Save all the data in json format file with file path inserted (--json=PATH)')
arg_parser.add_argument('--yaml', dest='yaml', type=str, default='', 
                        help='Save all the data in yaml format file with file path inserted (--yaml=PATH)')

arg_parser.add_argument('path', metavar='PATH', nargs='?', type=str, default='',
                        help='Path of file with stderr of ngtcp2 client|server.')
arg_parser.add_argument('mode', metavar='MODE', nargs='?', type=str, default='',
                        choices=['BYTE', 'PCKT'], help='Specify mode of calculation. '
                        'Note, that these are UDP packets! ')
arg_parser.add_argument('direction', metavar='DIRECTION', nargs='?', type=str, default='',
                        choices=['SENT', 'RCVD'], help='Direction of packets calculated.')
arg_parser.add_argument('parti', metavar='PARTITION', nargs='?', type=str, default='',
                        help='Integer number, that provides time partition.')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    filepath = args.path
    try:
        parti = int(args.parti)
    except Exception as E:
        sys.exit(f"speedlog.py:\nPartition must be integer: {E}")
    print(f"Estimating speed in {args.mode.upper()} mode:")
    data = parse_stat(filepath, parti, args)
    ld = []
    fnd = "Sent" if args.direction.upper() == "SENT" else "Received"
    whatis = []
    '''The order of next values must be preserved to ease coexistence with other scripts'''
    instrct = [['bytes sent: ', 'speed'],
               ['mean s_rtt: ', 'srtt'],
               ['mean l_rtt: ', 'lrtt'],
               ['mean min_rtt: ', 'mrtt'],
               ['mean jitter: ', 'jitt'],
               ['mean loss: ', 'loss'],
               ['mean loss2: ', 'loss2'],
               ['mean cwnd: ', 'cwnd']]
    if args.mode.upper() == "PCKT":
        instrct[0][0] = "pckts sent: "
    for idx, elem in enumerate(instrct):
        if elem[1] == "speed" or getattr(args, elem[1]):
            ld.append(data[idx])
            whatis.append(elem[0])
    savelist = []
    for i in ld[0].keys():
        pri = f"Second {i/1000}-{(i+parti)/1000}:"
        print(pri, end=" "*(20 - len(pri)))
        pri = f"{whatis[0]}{ld[0][i]}"
        print(pri, end=" "*(24 - len(pri)))
        currd = dict()
        if args.json != '' or args.yaml != '':
            currd["time"] = f"{i/1000}-{(i+parti)/1000}"
            currd[whatis[0]] = ld[0][i]
        for g, j in enumerate(ld[1:]):
            pri = f"{whatis[g+1]}{j[i]}"
            print(end=indented(pri, 24))
            if args.json != '' or args.yaml != '':
                currd[whatis[g+1]] = j[i]
        if args.json != '' or args.yaml != '':
            savelist.append(currd)
        print("")
    ld_means = dict()
    ld_means["time"] = "global"
    if ld[0]:
        print(f"General mean values by each {parti/1000} second:")
        print(" "*20, end="")
        for ind, d in enumerate(ld):
            if len(d.values()) != 1 and whatis[ind].startswith("mean speed"):
                '''We count mean speed ignoring last sample, because we are not sure if last period of time completed'''
                mean = round(sum([*d.values()][:-1])/len([*d.values()][:-1]), 6)
            elif len(d.values()) != 1 and whatis[ind] == "mean loss: ":
                '''This is made intentionally, because we encountered bad results during first second'''
                mean = round(sum([*d.values()][1:])/len([*d.values()][1:]), 6)
            elif len(d.values()) != 1:
                mean = round(sum([*d.values()])/len([*d.values()]), 6)
            else:
                mean = round([*d.values()][0], 6)
            ld_means[whatis[ind]] = mean
            pri = f"{whatis[ind]}{mean}"
            print(end=indented(pri, 24))
    savelist.append(ld_means)
    if args.json != '':
        with open(args.json, 'w') as f:
            json.dump(savelist, f, indent = 6)
    if args.yaml != '':
        with open(args.yaml, 'w') as f:
            yaml.dump(savelist, f, default_flow_style=True)
    sys.exit("\nSuccess")