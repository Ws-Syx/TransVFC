# cd demo/

python demo.py \
--config-file "/opt/data/private/syx/FastRCNN-envi/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml" \
--input "/opt/data/private/syx/dataset/waymo/coco-final-train/89454214745557131_3160_000_3180_000/1/FRONT.jpg" \
--output "/opt/data/private/syx/FastRCNN-envi/detectron2/demo" \
--opts MODEL.WEIGHTS "/opt/data/private/syx/FastRCNN-envi/detectron2/ckpt/faster-rcnn-r50-1x.pkl" 
# [--other-options] \