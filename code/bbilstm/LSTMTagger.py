import os, sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import datetime
import io

from utils_data import *
from utils_scores import *

class LSTMTagger(nn.Module):

    def __init__(self, vocab_size, tagset_size, config_file, task, target_vocab=None):
        super(LSTMTagger, self).__init__()
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.task = task
        self.config_file = config_file
        self.read_config()

        # -- Build word embeddings
        if self.embedding_file=="None":
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        else:
            word_to_vectors, word_vector_size, unk_vector = load_vectors( self.embedding_file )
            print( "Embeddings loaded from:", self.embedding_file, ", dim =", word_vector_size)
            if self.embedding_dim != word_vector_size:
                sys.exit("Embeddings loaded size %s do not match expected dim %s" % (word_vector_size, self.embedding_dim))
            weights_matrix = np.zeros((len(target_vocab),word_vector_size))
            for i,w in enumerate(target_vocab):
                try:
                    weights_matrix[i] = word_to_vectors[w]
                except KeyError:
                    weights_matrix[i] = unk_vector
                    #weights_matrix[i] = np.random.normal(scale=0.6, size=(word_vector_size, ))
            self.word_embeddings = nn.Embedding.from_pretrained( torch.FloatTensor(weights_matrix) )

        if self.embedding_trainable == False:
            self.word_embeddings.weight.requires_grad = False

        self.num_directions = 1
        if self.bidirectional:
        	self.num_directions = 2
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        if self.num_layers > 1:
            self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, 
            	dropout=self.dropout, bidirectional=self.bidirectional)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, 
            	bidirectional=self.bidirectional )
        # The linear layer that maps from hidden state space to tag space
        if self.bidirectional:
            self.hidden2tag = nn.Linear(self.hidden_dim*2, self.tagset_size)  # 2 for bidirection
        else:
        	self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.hidden = self.init_hidden()
        
    def read_config(self):
        self.config=json.load(open(self.config_file))

        self.embedding_file = self.config["embeddings"]["tokens"]["pretrained_file"]
        #print("EMBEDDING:", self.embedding_file, os.path.isdir( self.embedding_file ))
        if os.path.isdir( self.embedding_file ):
            self.embedding_file = retrieve_embed_file(self.embedding_file, self.task)
            #print( self.embedding_file )
        self.embedding_dim = self.config["embeddings"]["tokens"]["embedding_dim"]
        self.embedding_trainable = self.config["embeddings"]["tokens"]["trainable"]

        self.hidden_dim = self.config["encoder"]["hidden_size"]
        self.num_layers = self.config["encoder"]["num_layers"]
        self.dropout = self.config["encoder"]["dropout"]
        self.bidirectional = self.config["encoder"]["bidirectional"]

        self.optimizer = self.config["trainer"]["optimizer"]["type"]
        self.learning_rate = self.config["trainer"]["optimizer"]["lr"]

        self.num_epochs = self.config["num_epochs"]

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim),
                torch.zeros(self.num_layers*self.num_directions, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        
        x = embeds.view(len(sentence), 1, -1)
        # Forward propagate LSTM
        lstm_out, self.hidden = self.lstm(x, self.hidden) # embeds: 1 line per words, 1 vector # embed per line
        # Decode the hidden state of the last time step
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) # 1 line per word, 1 score per tag (i.e. 3 here)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def predict( self, data, word_to_ix, tag_to_ix ):
    	doc2count = {}
    	gold, pred = [],[]
    	with torch.no_grad():
        	for i,(sentence, tags) in enumerate(data):
        		inputs = prepare_sequence(sentence, word_to_ix)
        		targets = prepare_sequence(tags, tag_to_ix)
        		tag_scores = self(inputs)
        		y_pred = torch.argmax( tag_scores, dim=1 )      
        		gold.append( targets )
        		pred.append( y_pred )
        		doc2count[i] = len(inputs)
    	##print( "# of tokens per doc in val set:", doc2count )
    	f_pred = np.concatenate(tuple(pred))
    	f_gold = np.concatenate(tuple([[t for t in y] for y in gold])).reshape(f_pred.shape) 
    	return f_gold, f_pred
    
    def train( self, data, output_dir=None, loss='nll' ):
        torch.set_num_threads(5)
        beg = datetime.datetime.now()
        print( "Training starts at:", beg.strftime("%Y-%m-%d--%H:%M") )

        loss_function = self.get_loss( loss )
        optimizer = self.get_optimizer( )
        losses, experiments = [],[]
        for epoch in range( self.num_epochs ): 
            print( "\nTraining epoch:", epoch )
            total_loss = torch.Tensor([0])
            doc_targets, doc_preds = [], []
            for i, (document, tags) in enumerate(data.train_dataset):
                self.zero_grad() # Step 1. Clear the gradients out before each instance
                self.hidden = self.init_hidden() # Clear out the hidden state of the LSTM,
                sentence_in = prepare_sequence(document, data.word_to_ix) # Step 2. Turn the inputs into Tensors of word indices.
                targets = prepare_sequence(tags, data.tag_to_ix)
                tag_scores = self(sentence_in) # Step 3. Run our forward pass.
                loss = loss_function(tag_scores, targets) # Step 4. Compute the loss, gradients, and update the parameters
                loss.backward()
                optimizer.step()
                total_loss += loss.data
                # Save predictions
                y_pred = torch.argmax( tag_scores, dim=1 )
                #print( targets, '\n',y_pred )
                doc_targets.append( targets )
                doc_preds.append( y_pred )
            
            expe = Experiment( self, data, y_true=doc_targets, y_pred=doc_preds, 
            	train_loss=total_loss, num_epoch=epoch, config=self.config, ide=len(experiments) )
            # Predict on Dev
            if data.name == 'merged':
                print( 'Ite', epoch, 'Loss:', total_loss.item(), 
                        '\n\tTrain:', expe.printScores() )
                for i, (dset,dtoix) in enumerate( data.dev_datasets_doc_to_ix ):
                    fname = data.dev_fnames[i]
                    dev_y_true, dev_y_pred = self.predict( dset, data.word_to_ix, data.tag_to_ix )
                    expe.add_devScores_ml(dev_y_true, dev_y_pred, data.dev_fnames[i])
                    print( '\tDev', fname,':', expe.printScores(set='dev', fname=data.dev_fnames[i]) )

                for i, (dset,dtoix) in enumerate( data.test_datasets_doc_to_ix ):
                    fname = data.test_fnames[i]
                    print( fname )
                    test_y_true, test_y_pred = self.predict( dset, data.word_to_ix, data.tag_to_ix )
                    expe.add_testScores_ml(test_y_true, test_y_pred, data.test_fnames[i])
                    print( '\tTest', fname,':', expe.printScores(set='test', fname=data.test_fnames[i]) )
            else:
                dev_y_true, dev_y_pred = self.predict( data.dev_dataset, data.word_to_ix, data.tag_to_ix )
                expe.add_devScores(dev_y_true, dev_y_pred)
                print( 'Ite', epoch, 'Loss:', total_loss.item(), 
                    '\n\tTrain:', expe.printScores(), '\n\tDev:', expe.printScores(set='dev') )

                test_y_true, test_y_pred = self.predict( data.test_dataset, data.word_to_ix, data.tag_to_ix )
                expe.add_testScores(test_y_true, test_y_pred)
                print( '\tTest:', expe.printScores(set='test') )
            
            
            expe.todict( output_dir ) #dump models and predictions
            experiments.append( expe )
            
            losses.append(float(total_loss))
    
            
        # Save all the experiments (from utils_score.py)
        end = datetime.datetime.now()
        date_hour = end.strftime("%Y-%m-%d--%H:%M")
        print( "\nTraining ends at:", date_hour )
        
        save_experiments( experiments, os.path.join( output_dir, data.name+'_expe.json'), self )
        #print( losses )
        
    # Sortir de la classe
    def get_loss( self, loss_type ):
        if loss_type == 'nll':
            return nn.NLLLoss()
        # Default
        return nn.NLLLoss()
    
    def get_optimizer( self ):
        if self.optimizer == 'adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        return optim.Adam(self.parameters(), lr=self.learning_rate)


def load_word_vectors(fname):
    word_to_index, word_vectors, word_vector_size = {},[],-1
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
        word_to_index[tokens[0]] = len(word_to_index)
        word_vectors.append(map(float, tokens[1:]))
        if word_vector_size < 0:
            word_vector_size = len(tokens[1:])
    #print(word_to_index, '\n', word_vectors,'\n', word_vector_size)
    return word_to_index, word_vectors, word_vector_size

def load_vectors(fname):
    word_vector_size = -1
    unk_vector = np.zeros(0) 
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    if not 'glove' in fname:
        n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        #data[tokens[0]] = map(float, tokens[1:]) 
        data[tokens[0]] = [float(v) for v in tokens[1:]] 
        if unk_vector.shape[0] == 0:
            unk_vector = np.asarray( data[tokens[0]] )
        else:
            unk_vector += np.asarray( data[tokens[0]] )
        if word_vector_size < 0:
            word_vector_size = len(tokens[1:])
    unk_vector = np.asarray(unk_vector) / len(data)
    return data, word_vector_size, list(unk_vector)



def retrieve_embed_file( inpath, task, name=None ):
    #if inpath.endswith('multi_fastText'):
    task2embed = {'zho.rst.sctb':'zh', 'eng.pdtb.pdtb':'en', 'eng.rst.rstdt':'en', 'spa.rst.rststb':'es', 
    'fra.sdrt.annodis':'fr', 'rus.rst.rrt':'ru', 'eng.sdrt.stac':'en', 'zho.pdtb.cdtb':'zh', 'nld.rst.nldt':'nl', 
    'deu.rst.pcc':'de', 'por.rst.cstn':'pt', 'spa.rst.sctb':'es', 'eus.rst.ert':'eu', 'eng.rst.gum':'en',
    'tur.pdtb.tdb':'tr', 'merged':'merged'}
    #print( inpath, task, task2embed[task] )
    fasttext = os.path.join( inpath, 'cc.'+task2embed[task]+'.300.vec' )
    if os.path.isfile( fasttext ):
        return fasttext
    else:
        muse = os.path.join( inpath, 'wiki.multi.'+task2embed[task]+'.vec' )
        if os.path.isfile(muse):
            return muse
        else:
            sys.exit("Unk embeddings", inpath, task)