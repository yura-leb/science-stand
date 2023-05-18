import sys
from collections import defaultdict
import argparse
import json
import yaml
import numpy as np


def make_samples(args):
    '''Return samples depending on distribution'''
    if args.distr == "NORM":
        samples_list = np.random.normal(args.loc, args.scale, args.size).tolist()
    elif args.distr == "UNIFORM":
        samples_list = np.random.unfiorm(args.low, args.high, args.size).tolist()
    else:
        sys.exit("Wrong distribution name, use help!")
    '''Time periods should be greater than zero'''
    samples_list_greater_than_zero = [i if i > 0 else 0 for i in samples_list]
    return samples_list_greater_than_zero

arg_parser = argparse.ArgumentParser(prog='distribute_on_off',
                                     description='Create time periods to control sending and waiting periods ' 
                                     'of on-off application. This tool creates DICT with keys "wait", "send" '
                                     'Each key can be used to access array with data samples. ')
arg_parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.1')
input_group = arg_parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--json', dest='json', type=str, default='', 
                        help='Save all the data in json format file with file path inserted (--json=PATH)')
input_group.add_argument('--yaml', dest='yaml', type=str, default='', 
                        help='Save all the data in yaml format file with file path inserted (--yaml=PATH)')
                    
arg_parser.add_argument('size', metavar='SIZE', type=int,
                        help='Number of data samples in each array.')

subparsers = arg_parser.add_subparsers(help='Sub-distribution help')

parser_norm = subparsers.add_parser('NORM')
parser_norm.set_defaults(distr='NORM')
parser_norm.add_argument('loc', metavar='PARAM_1', type=float,
                         help='Mean (“centre”) of the distribution.')
parser_norm.add_argument('scale', metavar='PARAM_2', type=float,
                         help='Standard deviation (spread or “width”) of the distribution. Must be non-negative.')

parser_uniform = subparsers.add_parser('UNIFORM')
parser_uniform.set_defaults(distr='UNIFORM')
parser_uniform.add_argument('low', metavar='PARAM_1', type=float,
                            help='Lower boundary of the output interval. All values generated will be greater than '
                            'or equal to low. The default value is 0.')
parser_uniform.add_argument('high', metavar='PARAM_2', type=float,
                            help='Upper boundary of the output interval. All values generated will be less than or '
                            'equal to high.')


if __name__ == '__main__':
    args = arg_parser.parse_args()
    savelist = dict()
    savelist["wait"] = make_samples(args)
    savelist["send"] = make_samples(args)
    if args.json != '':
        with open(args.json, 'w') as f:
            json.dump(savelist, f, indent = 6)
    elif args.yaml != '':
        with open(args.yaml, 'w') as f:
            yaml.dump(savelist, f, default_flow_style=True)
    else:
        sys.exit("Whether --json or --yaml parameter must be set!")
    sys.exit("\nSuccess")