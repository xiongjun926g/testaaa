This is the contextual-embeddings segmentation experiments base directory

* Requirements:
------------------

 - allennlp library (and pytorch)
 - Glove 50d English embedding for the elmo English models, which needs to be pointed to with the GLOVE_PATH environment variable

Other resources (Bert, Elmo) are automatically downloaded

The experiment is setup to run on the GPU, it is much slower otherwise, but you can try by changing the cuda_device variable to -1 in configs/bert.jsonnet

* How to train/test the models
--------------------------------

You can train all bert-based  models at once with (takes a while); it will also predict on the dev sets 

> sh train_all.sh "deu.rst.pcc  eng.rst.gum  eng.sdrt.stac  fra.sdrt.annodis  por.rst.cstn  spa.rst.rststb  tur.pdtb.tdb  zho.rst.sctb  eng.pdtb.pdtb  eng.rst.rstdt  eus.rst.ert  nld.rst.nldt  rus.rst.rrt  spa.rst.sctb zho.pdtb.cdtb" "split.tok conll" bertM

You can predict on all tests with: 

> sh test_all.sh

Then collect all scores with ../utils/collect_scores.py on dev or test, i.e.

> python ../utils/collect_scores.py 'Results_split.tok/*/*test.scores'
> python ../utils/collect_scores.py 'Results_conll/*eng.rst.rstdt/*test.scores'


* Notes about preprocessing:
the directory data_converted contains preprocessed input, that is automatically computed if absent.

there are catches with russian and turkish though, because turkish is automatically split when going beyond a certain word limit, and
russian has been cleaned up for urls and some special symbols.  

