FILE=$1
BASE=${FILE##*/}
BASE=${BASE%.*}
# echo $BASE
PARAM_FILE=params_$BASE
PRNN=biGRU_jump
ARNN=biGRU_big
JOINT=_
echo $PARAM_FILE

python decompressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -input $BASE -output decom$JOINT$BASE

