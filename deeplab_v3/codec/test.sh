ROOT=./
export PYTHONPATH=$PYTHONPATH:$ROOT
#mkdir snapshot_FVC
CUDA_VISIBLE_DEVICES=0  python -u $ROOT/subnet/main.py --log loguvg.txt --testuvg --config config1024.json \
--pretrain /opt/data/private/lyx/DCVC-main_new/snapshot_6.9/iter80765.model




#--pretrain $ROOT/checkpoints/model_dcvc_quality_3_psnr.pth

# nohup sh test.sh >> nohup_test.txt &