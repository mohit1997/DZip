FILE=$1
tar -xf $1
source vars.dzip
# BASE=${FILE##*/}
# BASE=${BASE%.*}
# echo $BASE
PARAM_FILE=params_$BASE
# PRNN=biGRU_jump
# ARNN=biGRU_big
JOINT=_
# mode=$2
OUTPUT=$2
echo $PARAM_FILE

if [ "$mode" = com ] ; then
	python decompressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -input $BASE -output $2
elif [ "$mode" = bs ] ; then
	python decompressor_bs.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 128 -input $BASE -output $2
fi

rm vars.dzip $PARAM_FILE $BASE$JOINT$PRNN $BASE.dzip

