# ToNy

Systems for discourse segmentation submitted  to the 2019 DisRPT shared task:

*ToNy: Contextual embeddings for accurate multilingual discourse segmentation of full documents*

#### Code for the sentence segmentation preprocessing
*code/ssplit/* contains the code for the sentence splitter.
It assumes that each corpus is a folder under *data/* (ex: *data/deu.rst.pcc*).

Requirements:

* StanfordNLP: tested with v. 0.1.1
* Word embeddings: see Utils

Usage:
```
# call separately for each corpus in turn
python code/ssplit/parse_corpus.py deu.rst.pcc
# follow the instruction on the prompt to download the model for the language
# (the first time) ; typical models are 1--2 GB large
```

#### Code for the baseline heuristic segmenter
*code/baseline/* contains the code for the baseline heuristic, sentence-initial, segmenter.

Usage:
```
# call for all corpora in one go (very fast) ;
# processes both the original .conll files and the ssplit .tok files produced earlier
python code/baseline/baseline_seg.py
```

#### Code for the baseline systems using a bi-LSTM

*code/bbilstm/* contains the code for the baseline bi-LSTM systems

Requirements:

* pytorch (cuda not required): tested with v.1.0.1

##### Config

*config/* dir contains json config files for each setting (e.g. set of hyper-parameters and embedding file). **The path to the embedding file for GloVe and the directory for FastText (one file per language) should be modified in config files 3 and 5**

* config1: rand embeddings, dim 50
* config2: rand embeddings, dim 300
* config3: multi FastText, dim 300
* config5: GloVe (English), dim 50

##### Scripts

 *scripts/* dir contains bash scripts to run the systems:
 
 * *expes_tony_doc.sh* runs a system taking a whole document as input (i.e. .tok files in data dir) for each dataset. It takes two arguments: the config file and the path to the output directory (where the models, predictions and scores will be saved)

		cd tony/code/bbilstm/scripts
		bash expes_tony_doc.sh ../config/config1.json ../../../expes/bbilstm/doc/

* *expes_tony_sent.sh* runs a system taking a sntence split corpus as input (i.e. *SEE SECTION ON PRE PROCESSING*). It takes three arguments: the config file, the path to the output directory and either gold or pred (gold to use the sentence split given in the .conll files, and pred fot the sentence split predicted by the StanfordNLP pipeline)

For example, to run the system using the .conll files:

		cd tony/code/bbilstm/scripts
		bash expes_tony_sent.sh ../config/config1.json ../../../expes/bbilstm/gsent/ gold


#### Utils

*code/utils/* contains:
	* *seg_eval.py*: script for evaluation
	* *dl_embeddings.sh*: script for downloading the word embeddings (FastText and GloVe)

#### Other directories:

* *data/* contains the original data (.tok and .conll)
* *data_ssplit/* contains the pre-processed data using the StanfordNLP pipeline
* *expes/* contains examples of the output of our scripts
	* Examples of the output of the baseline bi-LSTM can be found in *doc/*, *gsent/* and *psent/*
