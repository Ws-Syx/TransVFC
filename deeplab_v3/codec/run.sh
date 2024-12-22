ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
mkdir $ROOT/snapshot
CUDA_VISIBLE_DEVICES=0,1 python -u $ROOT/subnet/main.py --log log.txt --config $ROOT/config2048.json \
#  --pretrain $ROOT/snapshot/iter1292240.model

