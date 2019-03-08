# -*- coding: utf-8 -*-
"""
Collect scores from experiments and output a nice latex table

usage: 
    python collect_scores.py file_mask


eg: 
  - to get all dev scores on the split.tok config
    python collect_scores.py 'Results_split.tok/*/*dev.scores'
  - to get all methods on one dataset on dev
    python collect_scores.py 'Results_*/*eng.rst.rstdt/*dev.scores'

"""
import sys, os
from glob import glob
import pandas as pds
# where are they
# eg ../results/conll/
data_dir = sys.argv[1]

# remove info about the config used tok/conll/split.tok
DROP_CONFIG = True    


def collect_scores(data_mask):
    all_tasks = glob(data_mask) 


    default_data = [("Precision",0),('Recall', 0),('F-Score', 0)]

    res = []
    for task in sorted(all_tasks):
        task_name = task.split(os.sep)[-1]
        task_name = task_name.split("_")[0]
        method_name = task.split(os.sep)[-2].split("_")[-1]
        config_name = task.split("Results_")[1].split("/")[0]
        print("counting:",task_name)
        with open(task) as f:
            data = [x.strip().split()[1:] for x in f.readlines()[-3:]]
            data = [(x[:-1],float(y)) for (x,y) in data]
            if data == []:
                data = default_data
            #res[task_name,method_name]=dict(data)
            data = [("task",task_name),("method",method_name),("config",config_name)] + data
            res.append(dict(data))

    table = pds.DataFrame(res)
    return table

table = collect_scores(data_dir)
if DROP_CONFIG: 
    table = table.drop("config",axis=1)
    indexing = ["task","method"]
else:
    indexing = ["task","method","config"]
table = table.set_index(indexing)
table.sort_index(inplace=True)

pds.options.display.float_format = '{:.2%}'.format

print(table.to_latex().replace("\\%",""))
