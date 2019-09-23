FILE="first_400000"
BASE=${FILE##*/}
BASE=${BASE%.*}
# echo $BASE
PARAM_FILE=params_$BASE
PRNN=biGRU_jump
ARNN=biGRU_big
JOINT=_
echo $PARAM_FILE

# python run.py --file_name $FILE
# python train_PRNN.py --file_name $BASE --model $PRNN
echo "$BASE$JOINT$PRNN"
# python train_PRNN.py --file_name $BASE --epochs 10 --gpu 0
python train_ARNN.py --file_name $BASE --ARNN $ARNN --PRNN $PRNN --gpu 0
