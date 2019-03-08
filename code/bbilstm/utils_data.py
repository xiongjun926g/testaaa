import os, sys, random
import torch


class Dataset:

    def __init__(self, inpath, name, read=True, sentence=False, conll=False):
        self.inpath=inpath
        self.name=name

        self.train_tok = os.path.join( self.inpath, self.name, self.name+'_train.tok' )
        self.dev_tok = os.path.join( self.inpath, self.name, self.name+'_dev.tok' )
        self.test_tok = os.path.join( self.inpath, self.name, self.name+'_test.tok' )

        if conll:
            sentence=True
            self.train_tok = os.path.join( self.inpath, self.name, self.name+'_train.conll' )
            self.dev_tok = os.path.join( self.inpath, self.name, self.name+'_dev.conll' )
            self.test_tok = os.path.join( self.inpath, self.name, self.name+'_test.conll' )


        if not os.path.isfile( self.train_tok ) or not os.path.isfile( self.dev_tok ):
            print( "No train or dev file for", inpath, file=sys.stderr )

        if read:
            self.train_dataset, self.train_doc_to_ix = read_data( self.train_tok, sentence=sentence )
            self.dev_dataset, self.dev_doc_to_ix = read_data( self.dev_tok, sentence=sentence )
            #if os.path.isfile( self.test_tok ):
            self.test_dataset, self.test_doc_to_ix = read_data( self.test_tok, sentence=sentence )
            # Populate vocabulary and tagset
            self.word_to_ix, self.tag_to_ix = get_vocab_tagset( self.train_dataset )
            if 'BeginSeg=Yes' in self.tag_to_ix:
                self.tag_to_ix = {'BeginSeg=Yes': 1, '_': 0}
            if 'pdtb' in self.name.lower():
                self.tag_to_ix = {'_':0, 'Seg=B-Conn':1, 'Seg=I-Conn':2}


            self.vocab_size = len( self.word_to_ix )
            self.tagset_size = len( self.tag_to_ix )

            print( 'Data loaded:', self.name, 'Vocab:', self.vocab_size, 'Tagset:', self.tagset_size, 'labels', self.tag_to_ix )

class MLDataset:

    def __init__( self, inpath, name, datasets_names, tag_to_ix ):
        self.inpath=inpath
        self.name='merged'
        self.tag_to_ix=tag_to_ix
        self.train_tok = [ os.path.join( self.inpath, f, f+'_train.tok' ) for f in datasets_names ]
        self.dev_tok = [ os.path.join( self.inpath, f, f+'_dev.tok' ) for f in datasets_names ]
        self.test_tok = [ os.path.join( self.inpath, f, f+'_test.tok' ) for f in datasets_names ]
        #self.dev_tok = os.path.join( self.inpath, self.dev_name, self.dev_name+'_dev.tok' )
        #self.test_tok = os.path.join( self.inpath, self.dev_name, self.dev_name+'_test.tok' )

        self.train_dataset, self.train_doc_to_ix = read_data_ml( self.train_tok )

        self.dev_fnames = datasets_names
        self.dev_datasets_doc_to_ix = [ read_data( dtok ) for dtok in self.dev_tok]

        self.test_fnames = datasets_names
        self.test_datasets_doc_to_ix = [ read_data( dtok ) for dtok in self.test_tok]
        #self.dev_datasets, self.dev_doc_to_ix = [read_data( dtok ) for dtok in self.dev_tok] 
        #if os.path.isfile( self.test_tok ):
        #    self.test_dataset, self.test_doc_to_ix = read_data( self.test_tok )
        # Populate vocabulary and tagset
        self.word_to_ix, _ = get_vocab_tagset( self.train_dataset )

        self.vocab_size = len( self.word_to_ix )
        self.tagset_size = len( self.tag_to_ix )

        print( self.name, 'Vocab:', self.vocab_size, 'Tagset:', self.tagset_size )


def read_datasets( inpath ):
    for subdir in [d for d in os.listdir( inpath ) if not d.startswith('.')]:
        yield Dataset( inpath, subdir )

def read_data( input_file, sentence=False ):
    '''Return: list of (tokens, tags), dict: doc_name:id'''
    dataset = []
    doc_to_ix = {}
    i = 0
    if sentence == True:
    	for toks, tags, doc_id in _read_sentence(input_file):
    	    dataset.append( (toks, tags))
    	    doc_to_ix[i] = doc_id
    	    i += 1
    else:
	    for toks, tags, doc_id in _read(input_file):
	    	dataset.append( (toks, tags))
	    	doc_to_ix[i] = doc_id
	    	i += 1
    return dataset, doc_to_ix

def read_data_ml( input_files ):
    '''Return: list of (tokens, tags), dict: doc_name:id'''
    dataset = []
    _doc_to_ix, doc_to_ix = [], {}
    i = 0
    for input_file in input_files:
        for toks, tags, doc_id in _read(input_file):
            dataset.append( (toks, tags))
            _doc_to_ix.append( doc_id )
            i += 1
    seed = 123456
    random.seed = seed 
    combined = list(zip(dataset, _doc_to_ix))
    random.shuffle(combined)
    dataset[:], _doc_to_ix[:] = zip(*combined)
    for i in range( len(_doc_to_ix)):
        doc_to_ix[i] = _doc_to_ix[i]
    return dataset, doc_to_ix



def _read(file_path: str):
    doc_id = 0
    idx = {}
    with open(file_path) as f:
        current_doc_tok = []
        current_doc_tags = []
        for line in f: 
            if line.startswith("#"):
                if doc_id!=0:
                    # end of previous document detected
                    if len( current_doc_tok ) == 0:
                        print( 'Empty document:', doc_name )
                    else:
                        yield [word for word in current_doc_tok], current_doc_tags, doc_name
                    current_doc_tok = []
                    current_doc_tags = []
                # go on as normal
                doc_id += 1
                doc_name = line.strip().split()[-1]
                idx[int(doc_id)] = doc_name
            elif line.strip()!="":
                id, token, *middle, tag = line.strip().split()
                current_doc_tok.append(token)
                current_doc_tags.append(tag)
    # final document emission  
    yield [word for word in current_doc_tok], current_doc_tags,doc_name

def _read_sentence(file_path: str):
    print("MODE SENTENCE")
    sent_id, doc_id, doc_name = 0, 0, None
    idx = {}
    doc2ctsent = {}
    with open(file_path) as f:
        current_sent_tok = []
        current_sent_tags = []
        for line in f: 
        	if line.strip() =="":
        		if len(current_sent_tok) != 0:
        			#print( "#Sentences for doc:", doc_name, "=", sent_id)
        			# end of previous sentence
        			yield [word for word in current_sent_tok], current_sent_tags, doc_name+'_'+str(sent_id)
        			current_sent_tags, current_sent_tok = [],[]
        		sent_id += 1
        	elif line.startswith("#"):
        		if doc_name != None:
        			doc2ctsent[doc_name] = sent_id
        		doc_id += 1
        		sent_id = 0
        		doc_name = line.strip().split()[-1]
        		idx[int(doc_id)] = doc_name
        	elif line.strip()!="":
        		id, token, *middle, tag = line.strip().split()
        		current_sent_tok.append(token)
        		current_sent_tags.append(tag)
    # final document emission  
    doc2ctsent[doc_name] = sent_id
    if len(current_sent_tok) != 0:
    	yield [word for word in current_sent_tok], current_sent_tags,doc_name
    #print(doc2ctsent)


def get_datasets( dataset, size_dev=10, size_test=10 ):
    '''
    dataset: list of (tokens, tags)
    Return: dev, test, train
    '''
    return dataset[:size_dev], dataset[:size_test], dataset[size_dev+size_test:] 
    
# NOTE: here we use a specific index (len(to_ix)-1) for UNKNOWN words
def prepare_sequence(seq, to_ix):
    '''Return a Tensor containing the id of the elements in the seq according to to_ix'''
    idxs = [to_ix[w] if w in to_ix else len(to_ix)-1 for w in seq] #specific index for unk word?
    return torch.tensor(idxs, dtype=torch.long)

def get_vocabulary( train ):
    '''Return a dict: word:id'''
    word_to_ix = {}
    for sent, tags in train:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # Add an index for UNK words
    word_to_ix['__unk__'] = len(word_to_ix)
    return word_to_ix

def get_tagset( train ):
    tag_to_ix = {}
    for sent, tags in train:
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)
    return tag_to_ix

def get_vocab_tagset( train ):
    '''Return a dict: word:id and a dict tag:id'''
    word_to_ix, tag_to_ix = {},{}
    for sent, tags in train:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)
    # Add an index for UNK words
    word_to_ix['__unk__'] = len(word_to_ix)
    return word_to_ix, tag_to_ix


def stats( train ):
    tok2begin = {}
    tok2count = {}
    for i,(tokens, tags) in enumerate(train):
        for j,w in enumerate( tokens ):
            if w in tok2count:
                tok2count[w] += 1
            else:
                tok2count[w] = 1
            if tags[j] == "BeginSeg=Yes":
                if w in tok2begin:
                    tok2begin[w] += 1
                else:
                    tok2begin[w] = 1
    return tok2count, tok2begin
