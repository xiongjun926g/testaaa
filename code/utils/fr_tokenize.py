"""take a French document and 
 
 1) use spacy to tokenize it
 2) format it as disrpt input 

TODO: conll option to output spacy analysis 
"""

import sys
import codecs
import spacy

fr = spacy.load('fr_core_news_sm')
extra_fields = 8
# WIP ...
CONLL = False

input_file = sys.argv[1]
input = codecs.open(input_file,encoding="utf8").read()



# watch the max length because of Bert restrictions to 512 subword units
current = max_length = 0
#cutoff = 200


# build conll stuff. bits from spacy_conll, adapted
# WIP
tagmap = fr.Defaults.tag_map

def get_morphology(self, tag):
    if not self.tagmap or tag not in self.tagmap:
        return '_'
    else:
        feats = [f'{prop}={val}' for prop, val in self.tagmap[tag].items() if not Spacy2ConllParser._is_number(prop)]
        if feats:
            return '|'.join(feats)
        else:
            return '_'

def head_idx(idx,word):
    if word.dep_.lower().strip() == 'root':
        head_idx = 0
    else:
        head_idx = word.head.i + 1 - sent[0].i
    return head_idx 

def word_tuple(idx,word):
    return (idx,
        word.text,
        word.lemma_,
        word.pos_,
        word.tag_,
        get_morphology(word.tag_),
        head_idx(idx,word),
        word.dep_,
        '_',
        '_'
    )
######################

doc = fr(input)



# raw doc
for i,token in enumerate(doc):
    if token.text.strip()!="":
        line = [str(i),token.text]+["_"]*extra_fields
        print("\t".join(line))
        current = current + 1
        #if current>cutoff:
        #    print()
    else:
        print()
        max_length = max(max_length,current)
        current = 0 

print("max length sequence = %s"%max_length,file=sys.stderr)
