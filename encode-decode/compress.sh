FILE=$1
BASE=${FILE##*/}
BASE=${BASE%.*}
# echo $BASE
PARAM_FILE=params_$BASE
PRNN=biGRU_jump
ARNN=biGRU_big
JOINT=_
EXT=.bitstream
mode=$3
OUTPUT=$2
echo $PARAM_FILE

python run.py --file_name $FILE

python train_bootstrap.py --file_name $BASE --model $PRNN --epochs 10
echo "$BASE$JOINT$PRNN"
if [ "$mode" = com ] ; then
	python compressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -data $BASE -data_params $PARAM_FILE -output $BASE$EXT
elif [ "$mode" = bs ] ; then
	python compressor_bs.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -data $BASE -data_params $PARAM_FILE -output $BASE$EXT
fi

for var in BASE PRNN ARNN mode; do
    declare -p $var | cut -d ' ' -f 3- >> vars.dzip
done

./bsc e $BASE$JOINT$PRNN $BASE$JOINT$PRNN.bsc
tar -cf $OUTPUT vars.dzip $PARAM_FILE $BASE$JOINT$PRNN.bsc $BASE$EXT.dzip
rm vars.dzip $PARAM_FILE $BASE$JOINT$PRNN $BASE$EXT.dzip $BASE$JOINT$PRNN.bsc

# python decompressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -input $BASE -output decom$JOINT$BASE


