#Config1: rand embeddings, dim 50
#Config2: rand embeddings, dim 300
#Config3: multi FastText, dim 300
#Config5: GloVe (English), dim 50

#"zho.pdtb.cdtb"
datasets=( "spa.rst.sctb" "spa.rst.rststb" "zho.rst.sctb" "eng.rst.rstdt" "fra.sdrt.annodis" "rus.rst.rrt" "eng.sdrt.stac" "nld.rst.nldt" "deu.rst.pcc" "por.rst.cstn" "eus.rst.ert" "eng.rst.gum" "eng.pdtb.pdtb" )
data="../../../data/"
#data="../../../data/_ssplit" # For experiments with predicted sentence split
config=$1
expedir=$2
src="../../../code/"
stype=$3 # gold (ie from .conll files) or pred (ie from ssplit files)

outpred=${expedir}/preds/
mkdir -p ${outpred}

for i in "${datasets[@]}"
do
   : 
   echo
   expedir_d=${expedir}/${i}/
   mkdir -p ${expedir_d}

   # (1) Train the system and save the settings, the scores and the predictions into a json file
   # dataset name, dat dir, config file, outpath dir, code dir, stype
   echo Sentence split from ${stype}
   bash ${src}/bbilstm/scripts/tony_sent.sh $i ${data} ${config} ${expedir_d} ${src} ${stype}

   echo Scores in ${expedir_d}/${i}_expe.json 
   # (2) Read the json and output pred file (for epoch 9)
   python ${src}/bbilstm/eval_tony.py --fjson ${expedir_d}/${i}_expe.json --outpath ${outpred} --epoch 0
done

# (3) Compute the scores by comparing gold and pred file
echo DEV 
for i in "${datasets[@]}"
do
   pred=${outpred}/${i}_dev.scores #TODO: rename, this is a pred file
   gold=${data}/${i}/${i}_dev.tok 
   python ${src}/utils/seg_eval.py ${gold} ${pred} 
done 

echo TEST 
for i in "${datasets[@]}"
do
   pred=${outpred}/${i}_test.scores
   gold=${data}/${i}/${i}_test.tok 
   python ${src}/utils/seg_eval.py ${gold} ${pred} 
done 