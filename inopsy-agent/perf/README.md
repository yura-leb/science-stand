# This folder accumulates applications for perfomance analysis

Workflow for statistical analysis:
 - Configure parameters in **inopsy-stand/RunAllExperiments.py**, run it;
 - Use **displayresults.py** to accumulate data into some dataframes from a directory **yaml** files;
 - Use **decision_tree_and_reg.py** to build ML algorithms for a variety of purposes in network design (for example: restore (predict) the speed with channel SLA variables status given)

Alternative way to collect data:
 - Run ngtcp2 **server** and **client** (examples folder), and put their stderr into two different files;
 - Run **statlog.py** to collect needed data about the experiment with **yaml** parameter, save all **yaml** files into one directory;
 - Configure these yaml parameters following way:
```
 {
    additional_info: {some parameters to parse further},
    content: [statlog.py output list]
 }
```
 - After several experiments use **displayresults.py** to acuumulate data into some dataframes from a directory **yaml** files;
 - Use **decision_tree_and_reg.py** to build ML algorithms for a variety of purposes in network design (for example: restore (predict) the speed with channel SLA variables status given).

# Utilities:
## decision_tree_and_reg.py
A notebook utility that aims to import data used for machine learning, preprocess it, and then use ML approaches themselves to train an algorithm capable of "predictions" such as restoration of speed function.

A brief description of each cell is given in the beginning of it.

## displayresults.py
A notebook utility that aims to process files with multiple statlog.py runs outputs into data, that can be put into xlsx file.

A brief description of each cell is given in the beginning of it.

## checksla.py
Check if connection statistics satisfies SLA. Note, that whether --jsonin or --yamlin parameter must be set. Depending on semantics SLA is satisfied if parameter is less/more than defined value

## speedlog.py (depricated, use statlog.py)
Calculate amount of client data sent. Note, that total packets calculated can be more, than content delivered, because we calculate content size of packets, that include content.

## statlog.py
Calculate the net features status through a period of time based on the agent output. Note, that agent (../examples/server, ../examples/client OR ../examples/quicin, ../examples/quicout) must be ran with an option --inopsy-log in order to generate logs suitable for statlog.py analysis. 

## distribute_on_off.py
Create time periods to control sending and waiting periods of on-off application. This tool creates DICT with keys "wait", "send" Each key can be used to access array with data samples.
