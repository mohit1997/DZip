FILE=$1
BASE=${FILE##*/}
BASE=${BASE%.*}
# echo $BASE
PARAM_FILE=params_$BASE
PRNN=biGRU_jump
ARNN=biGRU_big
JOINT=_
mode=$2
echo $PARAM_FILE

python run.py --file_name $FILE

python train_bootstrap.py --file_name $BASE --model $PRNN --epochs 10
echo "$BASE$JOINT$PRNN"
if [ "$mode" = com ] ; then
	python compressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -data $BASE -data_params $PARAM_FILE -output $BASE
elif [ "$mode" = bs ] ; then
	python compressor_bs.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -data $BASE -data_params $PARAM_FILE -output $BASE
fi

# python decompressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -input $BASE -output decom$JOINT$BASE


