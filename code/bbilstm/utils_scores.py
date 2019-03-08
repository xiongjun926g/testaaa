from utils_data import *
import numpy as np
from sklearn import metrics
import torch
import joblib
import json
from utils_data import *


class Experiment:
    def __init__( self, model, data, y_true=None, y_pred=None, train_loss=None, num_epoch=None, config=None, ide=0 ):
        self.model = model
        self.data = data
        self.y_pred = np.concatenate(tuple(y_pred))
        self.y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(self.y_pred.shape)
        self.train_loss = train_loss
        self.num_epoch = num_epoch
        self.config = config
        self.id_exp = ide
        
        self.train_acc, self.train_p, self.train_r, self.train_f1, self.train_s = compute_scores(self.y_true, self.y_pred)
        self.train_support = len( self.y_true==1 )

        self.dev_scores_ml = {} # for multilingual experiments
        self.test_scores_ml = {}

    def add_devScores( self, y_true, y_pred ):
        self.dev_y_true, self.dev_y_pred = y_true, y_pred 
        (self.dev_acc, self.dev_p, self.dev_r, self.dev_f1, self.dev_s) = compute_scores(self.dev_y_true, self.dev_y_pred)
        self.dev_support = len( y_true==1 )

    def add_testScores( self, y_true, y_pred ):
        self.test_y_true, self.test_y_pred = y_true, y_pred 
        (self.test_acc, self.test_p, self.test_r, self.test_f1, self.test_s) = compute_scores(self.test_y_true, self.test_y_pred)
        self.test_support = len( y_true==1 )

    def printScores( self, set='train', fname=None ):
        if set=='train':
            a,p,r,f,s = self.train_acc, self.train_p, self.train_r, self.train_f1, self.train_support
        elif set=='dev':
            if fname == None:
                a,p,r,f,s = self.dev_acc, self.dev_p, self.dev_r, self.dev_f1, self.dev_support
            else:
                _,_,a,p,r,f,s = self.dev_scores_ml[fname]
        elif set=='test':
            if fname == None:
                a,p,r,f,s = self.test_acc, self.test_p, self.test_r, self.test_f1, self.test_support
            else:
                _,_,a,p,r,f,s = self.test_scores_ml[fname]
        scores = {k:round(v*100,2) for (k,v) in {'a':a, 'p':p, 'r':r, 'f':f}.items()}
        scores['s'] = s
        return "Acc:{a}, Prec:{p}, Rec:{r}, F1:{f}, Sup:{s}".format_map( scores )

    def todict( self, outpath ):
        self.dict_ex = {}
        self.dict_ex["data"] = (self.data.inpath, self.data.name, self.data.vocab_size, self.data.tagset_size)
        self.dict_ex["train_scores"] = (self.train_acc, self.train_p, self.train_r, self.train_f1, self.train_s)
        
        self.dict_ex["dev_scores"] = (self.dev_acc, self.dev_p, self.dev_r, self.dev_f1, self.dev_s)
        self.dict_ex["dev_y_pred_file"] = os.path.join( outpath, "dev_preds_"+str(self.id_exp)+'.gz')
        if outpath != None:
            joblib.dump( self.dev_y_pred, self.dict_ex["dev_y_pred_file"])

        self.dict_ex["test_scores"] = (self.test_acc, self.test_p, self.test_r, self.test_f1, self.test_s)
        self.dict_ex["test_y_pred_file"] = os.path.join( outpath, "test_preds_"+str(self.id_exp)+'.gz')
        if outpath != None:
            joblib.dump( self.test_y_pred, self.dict_ex["test_y_pred_file"])
            
        self.dict_ex["train_loss"] = float( self.train_loss )
        self.dict_ex["epoch"] = self.num_epoch
        
        self.dict_ex["tag_to_ix"] = self.data.tag_to_ix
        self.dict_ex["word_to_ix"] = self.data.word_to_ix
        
        self.dict_ex["model_state_dict"] = os.path.join( outpath, "model_"+str(self.id_exp)+'.pth')
        if outpath != None:
            torch.save(self.model.state_dict(), self.dict_ex["model_state_dict"])
        self.dict_ex["config_file"] = self.model.config_file

def save_experiments( experiments, outpath, model ):
    saved = model.config #Retrieve all info from the original config file
    saved["Experiment"] = {}
    for e in experiments:
        saved["Experiment"][e.id_exp] = e.dict_ex

    with open(outpath, 'w') as outfile:
        json.dump( saved, outfile )


def load_experiment( inpath ):
    f = json.load(open(inpath))
    data_file, data_name, vocab_size, tagset_size = f["Experiment"]["data"]
    data = Dataset( data_file, data_name )
    model = LSTMTagger( vocab_size, tagset_size, f["Experiment"]["config_file"] )
    model.load_state_dict( torch.load( f["Experiment"]["model_state_dict"] ) )
    model.eval()
    e = Experiment( model, data )

def compute_scores( y_true, y_pred ):
    '''Return (Acc, P, R, F, support)'''
    acc = accuracy_score(y_true, y_pred)
    (p,r,f,s) = prf_scores( y_true, y_pred )
    return ( acc, p, r, f, s)

# -- Fct for computing scores
# TODO use scikit? issue when zero good pred
def accuracy_score( y_true, y_pred):
    # Overall
    return (y_true == y_pred).sum().item() / float(len(y_true))

def accuracy_score_s( y_true, y_pred):
    return metrics.accuracy_score( y_true, y_pred )

def prf_scores(y_true, y_pred):
    '''Return (prec, rec, f1, support)'''
    if len( np.unique(y_true) ) ==2:
        return metrics.precision_recall_fscore_support( y_true, y_pred, pos_label=1, average='binary' )
    else:
        return metrics.precision_recall_fscore_support( y_true, y_pred, average='micro' )

def f1_score(y_true, y_pred):
    return _f1_score( y_true, y_pred )

def _f1_score( y_true, y_pred ):
    if len( np.unique(y_true) ) ==2:
        return metrics.f1_score( y_true, y_pred, pos_label=1, average='binary' )
    else:
        #print( "f1score() --> multiclass computation (average = micro)" )
        return metrics.f1_score( y_true, y_pred, average='micro' )
  



'''      
def predict( model, data ):
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
    acc = _accuracy_score( f_gold, f_pred)
    f1 = _f1_score( f_gold, f_pred)
    return acc, f1, f_gold, f_pred
'''

# TODO rewrite
def write_pred( infile, outfile, preds ):
    lines = open(infile).readlines()
    # TODO what s the elegant way of doing this? just tolist()? flatten? squeeze?
    #https://stackoverflow.com/questions/53903373/convert-pytorch-tensor-to-python-list
    doc_id = -1
    with open( outfile, 'w' ) as o:
        line_count = 0
        for line in lines: 
            line = line.strip()
            if line.startswith("#"):
                ##print( "LINE #:", line, doc_id, line_count )
                doc_id += 1
                ##print( "Writing preds for doc:", doc_id)
                o.write( line.strip()+'\n')
            elif line.strip()!="":
                tag = preds[line_count]
                stag = 'BeginSeg=Yes' if tag == 1 else '_'
                ##print( "LINE:", line, doc_id, line_count, tag, stag)
                o.write( line.strip()+'\t'+stag+'\n' )
                line_count += 1
            else:
                ##print( "LINE EMPTY:", line, doc_id, line_count)
                o.write( line.strip()+'\n')
