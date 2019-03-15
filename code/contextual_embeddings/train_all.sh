export corpora=$1 
#"deu.rst.pcc  eng.rst.gum  eng.sdrt.stac  fra.sdrt.annodis  por.rst.cstn  spa.rst.rststb  tur.pdtb.tdb  zho.rst.sctb  eng.pdtb.pdtb  eng.rst.rstdt  eus.rst.ert  nld.rst.nldt  rus.rst.rrt  spa.rst.sctb zho.pdtb.cdtb"
export configs=$2
#"tok conll"
export models=$3
# bertM bert elmo


for d in $corpora; do
    for c in $configs; do
	for m in $models; do
	    echo "doing" $d $c $m
	    sh -x expes.sh $d $c $m
	done;
    done;
done;
