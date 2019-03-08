# ToNy

Systems for discourse segmentation submitted  to the 2019 DisRPT shared task:

*ToNy: Contextual embeddings for accurate multilingual discourse segmentation of full documents*

#### Code for the baseline systems using a bi-LSTM

*code/bbilstm/* contains the code for the baseline bi-LSTM systems

Requirements:

* pytorch (cuda not required): tested with v.1.0.1

##### Config

*config/* dir contains json config files for each setting (e.g. set of hyper-parameters and embedding file). **The path to the embedding file should be modified in config files 3 and 5**

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


#### Other directories:

* *data/* contains samples of the original data (for the corpus deu.rst.pcc)
	* Examples of the data pre-processed using the StanfordNLP pipeline are in *data/deu.rst.pcc/stanfordnlp/*
* *expes/* contains examples of the output of our scripts
	* Examples of the output of the baseline bi-LSTM can be found in *doc/*, *gsent/* and *psent/*
