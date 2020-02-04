"""24brièvement________

1ok_______BeginSeg=Yes
2bonjour________
3tout________
4le________
5monde________
6je________
7suis________
8Ilyes________
9Rebai________
10euh________
11je________
12suis________
13un________
14ingénieur________
15de________
16recherche________
17chez________
18Linagora________

1bonjour_______BeginSeg=Yes
"""


import sys
import codecs

input =  open(sys.argv[1],encoding="utf8").readlines()

start = True

for line in input: 
    if line.strip()=="":
        print("]")
        print()
        start = True
    else:
        n, word, *junk, tag = line.split()
        if tag=="BeginSeg=Yes":
            if not(start):
                print("]",end=" ")
            print("[",word,end=" ")
        else:
            print(word,end=" ")
        start = False
print("]")
        
