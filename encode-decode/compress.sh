FILE="files_to_be_compressed/xor10.txt"
BASE=${FILE##*/}
BASE=${BASE%.*}
# echo $BASE
PARAM_FILE=params_$BASE
PRNN=biGRU_jump
ARNN=biGRU_big
JOINT=_
echo $PARAM_FILE

python run.py --file_name $FILE
# python train_PRNN.py --file_name $BASE --model $PRNN
echo "$BASE$JOINT$PRNN"
python compressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 64 -data $BASE -data_params $PARAM_FILE -output $BASE
python decompressor.py -model $BASE$JOINT$PRNN -model_name $ARNN -batch_size 64 -input $BASE -output decom$JOINT$BASE


