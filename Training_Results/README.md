These are the graphs and logs that YOLOv8 automatically saves after training.

Training was done on my MacBook, took around 15+ hours.
Dataset: custom built for Indian road conditions, 12,198 images after augmentation.
This run is the final fine-tuning phase on my personal deployment-angle images
(photos taken from the actual spot where the camera would be mounted on the bike).

Files:
- results.png — shows how loss went down and precision/recall/mAP improved over 15 epochs
- confusion_matrix.png and confusion_matrix_normalized.png — how well helmet vs no_helmet 
  is being separated
- BoxF1_curve.png — F1 score at different confidence thresholds
- BoxPR_curve.png — Precision vs Recall tradeoff
- BoxP_curve.png and BoxR_curve.png — individual precision and recall curves
- labels.jpg — class distribution in the dataset
- train_batch0.jpg, train_batch1.jpg — sample training images with bounding boxes
- val_batch0_pred.jpg, val_batch1_pred.jpg — what the model predicted on validation images
- results.csv — raw numbers for every epoch (can open in Excel or Google Sheets)
