python molt_iou.py -i ../goat_voc/PredictIOU/Train/JPEGImages/ -t ../goat_voc/PredictIOU/Train/SegmentationClass/ -o Prediction/Train/ | tee train_iou.txt
python molt_iou.py -i ../goat_voc/PredictIOU/Val/JPEGImages/ -t ../goat_voc/PredictIOU/Val/SegmentationClass/ -o Prediction/Val | tee val_iou.txt
python molt_iou.py -i ../goat_voc/PredictIOU/Untested/JPEGImages/ -t ../goat_voc/PredictIOU/Untested/SegmentationClass/ -o Prediction/Untested/ | tee untested_iou.txt

