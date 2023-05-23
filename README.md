# swin_classification
Under Development



## Introduction to code usage

1. Download the dataset, the default used in the code is the MRI classification dataset，Download address:[https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset].
2. In `train.py` script to set `--data-path` to `mydataset` the absolute path of the folder.
3. Download pre-training weights, in `model.py`script, each model has a download address for pre-training
weights, you can download the corresponding pre-training weights according to the model you are using.
4. In `train.py` script to set `--weights` parameter to the downloaded pre-trained weight path.
5. Run `train.py` script to train model (`class_indices.json` File is automatically generated during training).
6. In `predict.py` script to import the same model as `train.py` script and set `model_weight_path` to the trained model weigth path (Saved in the weights folder by default).
7. In `predict.py` script to set `img_path` to the absolute paht of the image you need to pridect and set `json_path` to the absolute path of `class_indices.json` file.
8. If you want to use your own dataset, please follow the file structure of the default classification dataset (i.e. one category for one folder) and set the `num_classes` in `train.py` and `predict.py` scripts to the number of categories in your own dataset.
9. Pretrained model: swin_base_patch4_window12_384_22k.pth; swinv2_tiny_patch4_window8_256.pth
