# from raw (french only for now) text to segmented text with 
# two outputs: bracket notation and conll with "disrpt 2019 shared task"  tags 
# TODO: language specific calls


# home of the segmenter
export SEG_DIR=/home/muller/Devel/Disrpt/tony/

# first argument is file to segment
export RAW=$1
export FILE=$(basename "$RAW")
export BASE=$(dirname "$RAW")

echo "handling" $FILE "in" $BASE


# should be an argument of the script two
export MODEL=${SEG_DIR}/models/french_tokens.tar.gz




export RUNTIME=$SEG_DIR/code/contextual_embeddings/
# optional argument ?
export RESULT_DIR=Results_split.tok



# tokenize script should be made more generic
# or be optional if one wants to use separate tokenizer
# french one uses spacy french model
python ${SEG_DIR}/code/utils/fr_tokenize.py $RAW > ${RAW}.tok 

python $RUNTIME/conv2ner.py ${RAW}.tok > ${RAW}.ner.tok

allennlp predict --use-dataset-reader --output-file ${RESULT_DIR}/${FILE}.json ${MODEL} ${RAW}.ner.tok

python $RUNTIME/json2conll.py ${RESULT_DIR}/${FILE}.json split.tok > ${RESULT_DIR}/${FILE}.split.tok

python $RUNTIME/conll2bracket.py ${RESULT_DIR}/${FILE}.split.tok >  ${RESULT_DIR}/${FILE}.split.tok.bracket
