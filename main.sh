GPUS=$1

# Random MASTER_PORT
MIN=11111
MAX=99999
DIFF=$(($MAX-$MIN+1))
PORT=$(($(($RANDOM%$DIFF))+$MIN))

MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 torchrun --master_port=$PORT --nproc_per_node=$GPUS main.py ${@:2}
