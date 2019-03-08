# usage
#  sh expes.sh dataset config model

echo "data=$1, config=$2, model=$3"
   
export DATASET=${1}
# eg "eng.rst.gum"

export CONFIG=${2}
# options: conll tok split.tok wend.tok
#
export MODEL=${3}
# options: bert elmo bertM
# 
if [ "$MODEL"="bertM" ] ; 
then 
    export BERT_VOCAB="bert-base-multilingual-cased"
    export BERT_WEIGHTS="bert-base-multilingual-cased"
else
    # english models
    export BERT_VOCAB="bert-base-cased"
    export BERT_WEIGHTS="bert-base-cased"                                                                                                   
fi
# dev ou test
export EVAL=dev

export GOLD_BASE="../../data/"
export CONV="data_converted/"
export TRAIN_DATA_PATH=${CONV}${DATASET}"_train.ner."${CONFIG}
export TEST_A_PATH=${CONV}${DATASET}"_"${EVAL}".ner."${CONFIG}
export OUTPUT=${DATASET}"_"${MODEL}
export GOLD=${GOLD_BASE}${DATASET}"/"${DATASET}"_"${EVAL}"."${CONFIG}

# conversion des datasets au format NER/BIO  en testant d'abord existence fichiers pour pas le refaire à chaque fois
if [ ! -f ${CONV}${DATASET}"_train.ner."${CONFIG} ]; then
    echo "converting to ner format -> in data_converted ..."
    python conv2ner.py "../data/"${DATASET}"/"${DATASET}"_train."${CONFIG} > ${CONV}/${DATASET}"_train.ner."${CONFIG}
    python conv2ner.py "../data/"${DATASET}"/"${DATASET}"_"${EVAL}"."${CONFIG} > ${CONV}/${DATASET}"_"${EVAL}".ner."${CONFIG}
fi
# train with config in ner_elmo ou ner_bert.jsonnet; the config references explicitely variables TRAIN_DATA_PATH and TEST_A_PATH
allennlp train -s Results_${CONFIG}/results_${OUTPUT} ner_${MODEL}.jsonnet
# predict with model -> outputs json
allennlp predict --use-dataset-reader --output-file Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.json Results_${CONFIG}/results_${OUTPUT}/model.tar.gz ${TEST_A_PATH}
# convert to disrpt format 
python json2conll.py Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.json ${CONFIG} > Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.${CONFIG}
# eval with disrpt script
python ../utils/seg_eval.py $GOLD Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.${CONFIG} >> Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.scores
