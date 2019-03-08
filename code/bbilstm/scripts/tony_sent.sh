# usage
#  sh expes.sh dataset config model

echo "data=$1, config=$3, expedir=$4"

export DATASET=${1}
# eg "eng.rst.gum"
export DATADIR=${2}
# path_to/data/
export CONFIG=${3}
# eg config.json
export EXPEDIR=${4}
# output dir to save models
export SRC=${5}
# path_to/code/tony-pytorch/
export STYPE=${6}

# Training
if [ "${STYPE}" == "gold" ]
   	then
   		echo Read from .conll files
   		python ${SRC}/bbilstm/tony.py --dataset ${DATASET} --data_dir ${DATADIR} --config ${CONFIG} --outpath ${EXPEDIR} --sentence --conll 
	
	else
		echo read from .ssplit files
		python ${SRC}/bbilstm/tony.py --dataset ${DATASET} --data_dir ${DATADIR}/stanfordnlp/ --config ${CONFIG} --outpath ${EXPEDIR} --sentence
fi
