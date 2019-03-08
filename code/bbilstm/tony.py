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
            description='Discourse corpus (for now .dis or .rs3).')
    parser.add_argument('--data_dir',
            dest='data_dir',
            action='store',
            default="../../data/",
            help='Train file')

    parser.add_argument('--dataset',
            dest='dataset',
            action='store',
            default="eng.rst.gum",
            help='Dataset name')

    parser.add_argument('--train',
            dest='train',
            action='store',
            default="../../eng.rst.gum/eng.rst.gum_dev.tok",
            help='Train file')
    parser.add_argument('--dev',
            dest='dev',
            action='store',
            default="../../eng.rst.gum/eng.rst.gum_dev.tok",
            help='dev file')
    parser.add_argument('--config',
            dest='config',
            action='store',
            default='config.json',
            help='config file')
    parser.add_argument('--outpath',
            dest='outpath',
            action='store',
            default='../../../../../sandbox/expe_tony/', #'/media/chloe/miya/expe_tony/',
            help='outpath dir')
    parser.add_argument('--sentence',
            dest='sentence',
            action='store_true',
            default=False,
            help='Read either documents (False) or sentences (True). Set to True if conll is True.')
    parser.add_argument('--conll',
            dest='conll',
            action='store_true',
            default=False,
            help='Read conll format')
    parser.add_argument('--model',
            dest='model',
            action='store',
            default=None,
            help='Model for eval')
    parser.add_argument('--confign',
            dest='confign',
            action='store',
            default=None,
            help='Config name (because config name in the nam of th f** model')

    args = parser.parse_args()

    if not os.path.isdir( args.outpath ):
    	os.mkdir(args.outpath)

    DATASETS=['zho.rst.sctb', 'eng.pdtb.pdtb', 'eng.rst.rstdt', 'spa.rst.rststb', 
    'fra.sdrt.annodis', 'rus.rst.rrt', 'eng.sdrt.stac', 'zho.pdtb.cdtb', 'nld.rst.nldt', 
    'deu.rst.pcc', 'por.rst.cstn', 'spa.rst.sctb', 'eus.rst.ert', 'eng.rst.gum']
    
    #for d in read_datasets( args.data_dir ):
    #	print(d.name)

    #data = Dataset( args.data_dir, 'eng.rst.gum' )
    if args.model == None:
        sentence=args.sentence
        if args.conll:
            sentence=True
        data = Dataset( args.data_dir, args.dataset, sentence=sentence, conll=args.conll )
        print( data.tag_to_ix )

        # Train and save models
        segmenter = LSTMTagger( data.vocab_size, data.tagset_size, args.config, data.name, target_vocab=data.word_to_ix )
        # --> need a path to save models, predictions, scores...
        segmenter.train( data, output_dir=args.outpath  )

    # Evaluate: return scores and predictions
    else:
        sentence=args.sentence
        if args.conll:
            sentence=True
        ite = 9
        model_path = args.model 
        for dname in DATASETS:
            print( '\n', dname )
            data = Dataset( args.data_dir, dname, sentence=sentence, conll=args.conll )
            tag_to_ix, word_to_ix = data.tag_to_ix, data.word_to_ix
            #config_file = os.path.join( args.config, dname+'_'+args.confign, dname+'_expe.json' )
            #print( 'Config:', config_file )
            #tag_to_ix, word_to_ix = read_config(config_file )
            model_file = os.path.join( model_path, dname+'_'+args.confign, 'model_'+str(ite)+'.pth' )
            print( 'Model:', model_file )

            model = LSTMTagger( data.vocab_size, data.tagset_size, args.config, data.name, target_vocab=data.word_to_ix )
            model.load_state_dict(torch.load(model_file))
            model.eval()

            y_gold, y_pred = _predict( model, data.test_dataset, word_to_ix, tag_to_ix )
            print( dname, f1_score(y_true, y_pred) )
            
def _predict( model, data, word_to_ix, tag_to_ix ):
    doc2count = {}
    gold, pred = [],[]
    with torch.no_grad():
        for i,(sentence, tags) in enumerate(data):
            inputs = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(inputs)
            y_pred = torch.argmax( tag_scores, dim=1 )      
            gold.append( targets )
            pred.append( y_pred )
            doc2count[i] = len(inputs)
    ##print( "# of tokens per doc in val set:", doc2count )
    f_pred = np.concatenate(tuple(pred))
    f_gold = np.concatenate(tuple([[t for t in y] for y in gold])).reshape(f_pred.shape) 
    return f_gold, f_pred

def read_config( fjson, num_epoch=9 ):
    dict_ex=json.load(open(fjson))
    for e in dict_ex["Experiment"]:
        if dict_ex["Experiment"][e]["epoch"] == num_epoch:
            print( dict_ex["Experiment"][e].keys())
            print( dict_ex["Experiment"].keys())
            print( dict_ex.keys())
            return dict_ex["Experiment"][e]["tag_to_ix"], dict_ex["Experiment"][e]["word_to_ix"]

if __name__ == '__main__':
    main()
