import sys
from collections import defaultdict
import argparse
import json
import yaml


def checkattr(timedata, args, key, argkey, islower):
    if key in timedata.keys():
        if islower * (timedata[key] - getattr(args, argkey)) >= 0:
            return("OK")
        else:
            return("BAD!")
    else:
        return("~~")

def indented(val: str, indent: int = 16):
    num_spaces = indent - len(val)
    if num_spaces <= 0:
        return val + " "
    return val + " " * num_spaces

arg_parser = argparse.ArgumentParser(prog='checksla',
                                     description='Check if connection statistics satisfies SLA. '
                                     'Note, that whether --jsonin or --yamlin parameter must be set. '
                                     'Depending on semantics SLA is satisfied if parameter is less/more than defined value')
arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.1')
arg_parser.add_argument('--speed', type=float,
                        help='Use this parameter if you want mean speed SLA to be checked. '
                             'IF (real speed > --speed): SLA is OKAY')
arg_parser.add_argument('--lrtt', type=float,
                        help='Use this parameter if you want mean latest rtt SLA to be checked. '
                             'Latest rtt is rtt from each packet. '
                             'IF (real lrtt < --lrtt): SLA is OKAY')
arg_parser.add_argument('--srtt', type=float,
                        help='Use this parameter if you want mean smoothed rtt SLA to be checked. '
                             'By default srtt is calculated in agent with 7/8 of previous value, and 1/8 of new '
                             'depending on pckts recieved. '
                             'IF (real srtt < --srtt): SLA is OKAY')
arg_parser.add_argument('--mrtt', type=float,
                        help='Use this parameter if you want mean min rtt SLA to be checked'
                             'Min rtt is congestion controller estimate on what min_rtt is. '
                             'IF (real minrtt < --mrtt): SLA is OKAY')
arg_parser.add_argument('--loss', type=float,
                        help='Use this parameter if you want mean loss percent SLA to be checked. '
                             'IF (real loss < --loss): SLA is OKAY')
arg_parser.add_argument('--jitt', type=float,
                        help='Use this parameter if you want mean jitter SLA to be checked'
                             'IF (real jitt < --jitt): SLA is OKAY')

input_group = arg_parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--jsonin', dest='jsonin', type=str, default='',
                         help='Import data from json format file with file path inserted (--jsonin=PATH)')
input_group.add_argument('--yamlin', dest='yamlin', type=str, default='',
                         help='Import data from yaml format file with file path inserted (--yamlin=PATH)')

arg_parser.add_argument('--jsonout', dest='jsonout', type=str, default='',
                        help='Save all SLA data in json format file with file path inserted (--jsonout=PATH)')
arg_parser.add_argument('--yamlout', dest='yamlout', type=str, default='',
                        help='Save all SLA data in yaml format file with file path inserted (--yamlout=PATH)')

if __name__ == '__main__':
    indent = 24
    args = arg_parser.parse_args()
    datalist = []
    if args.jsonin != '':
        with open(args.jsonin, 'r') as f:
            datalist = json.load(f)
    elif args.yamlin != '':
        with open(args.yamlin, "r") as f:
            datalist = yaml.safe_load(f)
    else:
        sys.exit("Whether --jsonin or --yamlin parameter must be set!")
    slascore = defaultdict(int)
    instrct = [['bytes sent: ', 'speed', 1],
               ['mean s_rtt: ', 'srtt', -1],
               ['mean l_rtt: ', 'lrtt', -1],
               ['mean min_rtt: ', 'mrtt', -1],
               ['mean jitter: ', 'jitt', -1],
               ['mean loss: ', 'loss', -1]]
    '''This stucture eases parameter iteration in multiple cases.
    1) is name of value in output;
    2) is value name in input;
    3) is how we compare SLA values to actual values.
    '''    
    instrct = [tup for tup in instrct if getattr(args, tup[1])]
    glmeanspri = "No GLOBAL MEANS found"
    samplesnum = []
    savedata = dict()
    savedata['total_sla_okay'] = dict()
    savedata['total_samples'] = dict()
    savedata['global_means'] = dict()
    print(" "*indent, end="")
    for ch in instrct:
        samplesnum.append(0)
        print(end=indented(ch[0]))
    print("")
    for timedata in datalist:
        if timedata['time'] == "global":
            pri = "GLOBAL MEANS:"
            pri += " "*(indent-len(pri))
            for k in instrct:
                r = checkattr(timedata, args, k[0], k[1], k[2])
                savedata['global_means'][k[0]] = r
                pri += r + " "*(indent - len(r) - 8)
            glmeanspri = pri
        else:
            pri = f"Second {timedata['time']}:"
            print(pri, end=" "*(indent - len(pri)))
            for indx, k in enumerate(instrct):
                r = checkattr(timedata, args, k[0], k[1], k[2])
                slascore[k[0]] += int(r == "OK")
                samplesnum[indx] += int((r == "OK") or (r == "BAD!"))
                print(end=indented(r))
            print("")
    print(end=indented("TOTAL:", 24))
    for ind, val in enumerate(slascore.values()):
        savedata['total_sla_okay'][instrct[ind][0]] = val
        savedata['total_samples'][instrct[ind][0]] = samplesnum[ind]
        pri = str(val)+" / "+str(samplesnum[ind])
        print(end=indented(pri))
    print("\n")
    print(glmeanspri)
    if args.jsonout != "":
        with open(args.jsonout, 'w') as f:
            json.dump(savedata, f, indent = 6)
    if args.yamlout != "":
        with open(args.yamlout, 'w') as f:
            yaml.dump(savedata, f, default_flow_style=True)
    print("Success")