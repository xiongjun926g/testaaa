#corpora = "deu.rst.pcc  eng.rst.gum  eng.sdrt.stac  fra.sdrt.annodis  por.rst.cstn  spa.rst.rststb  tur.pdtb.tdb  zho.rst.sctb  eng.pdtb.pdtb  eng.rst.rstdt  eus.rst.ert  nld.rst.nldt  rus.rst.rrt  spa.rst.sctb "
#configs = "tok conll"
#models = "bert elmo"


# 
for DATASET in deu.rst.pcc  eng.rst.gum  eng.sdrt.stac  fra.sdrt.annodis  por.rst.cstn  spa.rst.rststb  tur.pdtb.tdb  zho.rst.sctb  eng.pdtb.pdtb  eng.rst.rstdt  eus.rst.ert  nld.rst.nldt  rus.rst.rrt  spa.rst.sctb; 

# specific for english
#for DATASET in eng.rst.gum  eng.sdrt.stac eng.pdtb.pdtb  eng.rst.rstdt;

do
    for CONFIG in split.tok conll; do
	#specific english would be / 
	#for MODEL in elmo bertM; do
	    export MODEL=bertM
	    export OUTPUT=${DATASET}"_"${MODEL}
	    if [ ! -f Results_${CONFIG}/results_${OUTPUT}/${DATASET}_test.predictions.json ] ; then
		echo predicting $DATASET $CONFIG with $MODEL
		sh test_eval.sh $DATASET $CONFIG $MODEL
	    fi
	#done;
    done;
done;
