# usage
#  sh expes.sh dataset config model

echo "data=$1, config=$2, model=$3"
   
export DATASET=${1}
# eg "eng.rst.gum"

export CONFIG=${2}
# options: conll tok split.tok
#
export MODEL=${3}
# options: bert elmo bertM
# 
export EVAL=test

export GOLD_BASE="../../data/"
export CONV="data_converted/"

export TRAIN_DATA_PATH=${CONV}${DATASET}"_train.ner."${CONFIG}
export TEST_A_PATH=${CONV}${DATASET}"_"${EVAL}".ner."${CONFIG}

export OUTPUT=${DATASET}"_"${MODEL}
export GOLD=${GOLD_BASE}${DATASET}"/"${DATASET}"_"${EVAL}"."${CONFIG}

# conversion of datasets to format NER/BIO  
if [ ! -f ${CONV}${DATASET}"_train.ner."${CONFIG} ]; then
    echo "converting train set to ner format -> in data_converted ..."
    python conv2ner.py "../data/"${DATASET}"/"${DATASET}"_train."${CONFIG} > ${CONV}/${DATASET}"_train.ner."${CONFIG}
fi
if [ ! -f ${CONV}${DATASET}"_"${EVAL}".ner."${CONFIG} ]; then
    echo "converting eval set to ner format -> in data_converted ..."
    python conv2ner.py "../data/"${DATASET}"/"${DATASET}"_"${EVAL}"."${CONFIG} > ${CONV}/${DATASET}"_"${EVAL}".ner."${CONFIG}
fi

# predict with model -> outputs json
allennlp predict --use-dataset-reader --output-file Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.json Results_${CONFIG}/results_${OUTPUT}/model.tar.gz ${TEST_A_PATH}
# convert to disrpt format 
python json2conll.py Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.json ${CONFIG} > Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.${CONFIG}
# eval with disrpt script
python ../utils/seg_eval.py $GOLD Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.predictions.${CONFIG} >> Results_${CONFIG}/results_${OUTPUT}/${DATASET}_${EVAL}.scores
