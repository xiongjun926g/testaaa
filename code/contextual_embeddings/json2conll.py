"""
reexports allennlp predictions from json to 
conll format
"""

import json
import sys

filepath = sys.argv[1]
config = sys.argv[2]
# conll ou tok

map = {"O":"_",
       "B-S":"BeginSeg=Yes",
       "U-S":"BeginSeg=Yes",
       "U-Conn":"Seg=B-Conn",
       "L-Conn":"Seg=I-Conn",
       "I-Conn":"Seg=I-Conn",
       "B-Conn":"Seg=B-Conn",
       "B-E":"_",
       "U-E":"_",
       }

data = [] 
for line in open(filepath, 'r'):
    data.append(json.loads(line))

for doc in data:
    tokens = zip(doc["words"],doc["tags"])
    out = "\n".join(("%s\t%s\t%s%s"%(i+1,word,"_\t"*7,map.get(tag,tag)) for (i,(word,tag)) in enumerate(tokens)))
    if config=="tok":
        print("# blabla")
    print(out)
    print()
