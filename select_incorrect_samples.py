"""
This script picks out the images that are incorrectly predicted in the validation set 
and records them in record.txt
"""
import os
import json
import argparse
import sys

import torch
from torchvision import transforms
from tqdm import tqdm

from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model
from utils import read_split_data, read_test_data


def main(args, data=''):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    img_size = 384
    data_transform = {
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        
        "test": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    if data == 'val_dataset':
        _, _, val_images_path, val_images_label = read_split_data(args.data_path)
        # read validation dataset
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])
        
        # load validation dataset
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=nw,
                                                 collate_fn=val_dataset.collate_fn)
    
    elif data == 'test_dataset':
        test_images_path, test_images_label = read_test_data(args.data_path)
        # read test dataset
        test_dataset = MyDataSet(images_path=test_images_path,
                                 images_class=test_images_label,
                                 transform=data_transform["test"])
        
        # load test dataset
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=test_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model.eval()
    with torch.no_grad():
        if data == 'val_dataset':
            with open("record_val.txt", "w") as f:
                # validate
                data_loader = tqdm(val_loader, file=sys.stdout)
                for step, data in enumerate(data_loader):
                    images, labels = data
                    pred = model(images.to(device))
                    pred_classes = torch.max(pred, dim=1)[1]
                    contrast = torch.eq(pred_classes, labels.to(device)).tolist()
                    labels = labels.tolist()
                    pred_classes = pred_classes.tolist()
                    for i, flag in enumerate(contrast):
                        if flag is False:
                            file_name = val_images_path[batch_size * step + i]
                            true_label = class_indict[str(labels[i])]
                            false_label = class_indict[str(pred_classes[i])]
                            f.write(f"{file_name}  TrueLabel:{true_label}  PredictLabel:{false_label}\n")
        elif data == 'test_dataset':
            with open("record_test.txt", "w") as f:
                # test
                data_loader = tqdm(test_loader, file=sys.stdout)
                for step, data in enumerate(data_loader):
                    images, labels = data
                    pred = model(images.to(device))
                    pred_classes = torch.max(pred, dim=1)[1]
                    contrast = torch.eq(pred_classes, labels.to(device)).tolist()
                    labels = labels.tolist()
                    pred_classes = pred_classes.tolist()
                    for i, flag in enumerate(contrast):
                        if flag is False:
                            file_name = test_images_path[batch_size * step + i]
                            true_label = class_indict[str(labels[i])]
                            false_label = class_indict[str(pred_classes[i])]
                            f.write(f"{file_name}  TrueLabel:{true_label}  PredictLabel:{false_label}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)

    # The root directory where the dataset is located
    parser.add_argument('--data-path', type=str,
                        default="/dataset/Testing")

    # Training weight path
    parser.add_argument('--weights', type=str, default='/weights/model-6.pth',
                        help='initial weights path')
    # Whether to freeze weights
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt, data='test_dataset')
