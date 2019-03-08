import argparse, os, sys, subprocess, shutil, codecs

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json

from sklearn import metrics

from LSTMTagger import *
from utils_data import *
from utils_scores import *

torch.manual_seed(1)
torch.set_num_threads(5) #also in train fct, where should it be?

def main( ):
    parser = argparse.ArgumentParser(
            description='Discourse segmenter based on a bi-LSTM model.')
    parser.add_argument('--data_dir',
            dest='data_dir',
            action='store',
            default="../../data/",
            help='Train file (Default:../../data/)')
    parser.add_argument('--dataset',
            dest='dataset',
            action='store',
            default="deu.rst.pcc",
            help='Dataset name (Default: deu.rst.pcc)')
    parser.add_argument('--config',
            dest='config',
            action='store',
            default='config/config1.json',
            help='config file')
    parser.add_argument('--outpath',
            dest='outpath',
            action='store',
            help='outpath dir')
    parser.add_argument('--sentence',
            dest='sentence',
            action='store_true',
            default=False,
            help='Read either documents (False) or sentences (True). Set to True if conll is True. (Default: False)')
    parser.add_argument('--conll',
            dest='conll',
            action='store_true',
            default=False,
            help='Read conll format. (Default: False)')
 
    args = parser.parse_args()

    if not os.path.isdir( args.outpath ):
    	os.mkdir(args.outpath)

    DATASETS=['zho.rst.sctb', 'eng.pdtb.pdtb', 'eng.rst.rstdt', 'spa.rst.rststb', 
    'fra.sdrt.annodis', 'rus.rst.rrt', 'eng.sdrt.stac', 'zho.pdtb.cdtb', 'nld.rst.nldt', 
    'deu.rst.pcc', 'por.rst.cstn', 'spa.rst.sctb', 'eus.rst.ert', 'eng.rst.gum']
    
    sentence=args.sentence
    if args.conll:
        sentence=True
    data = Dataset( args.data_dir, args.dataset, sentence=sentence, conll=args.conll )

    # Train and save models
    segmenter = LSTMTagger( data.vocab_size, data.tagset_size, args.config, data.name, target_vocab=data.word_to_ix )
    # --> need a path to save models, predictions, scores...
    segmenter.train( data, output_dir=args.outpath  )


if __name__ == '__main__':
    main()
