import os
import json
import argparse
import sys

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from utils import read_split_data, read_test_data
from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model


class ConfusionMatrix(object):
    """
    Using matplotlib-3.2.1(windows and ubuntu)
    Requires additional installation of the prettytable library
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # Set x-axis coordinates label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # Set y-axis coordinates label
        plt.yticks(range(self.num_classes), self.labels)
        # Display colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # Label the graph with quantity/probability information
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # Note: here matrix[y, x] is not matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


def main(args, str=''):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

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
    
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    if str == 'val_dataset':
        _, _, val_images_path, val_images_label = read_split_data(args.data_path)
        # read validation dataset
        val_dataset = MyDataSet(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])
        
        # load validation dataset
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=val_dataset.collate_fn)
    
    elif str == 'test_dataset':
        test_images_path, test_images_label = read_test_data(args.data_path)
        # read test dataset
        test_dataset = MyDataSet(images_path=test_images_path,
                                 images_class=test_images_label,
                                 transform=data_transform["test"])

        # load test dataset
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=test_dataset.collate_fn)
    
    model = create_model(num_classes=args.num_classes)
    # load pretrain weights
    assert os.path.exists(args.weights), "cannot find {} file".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # read class_indict
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=args.num_classes, labels=labels)
    model.eval()
    with torch.no_grad():
        if str == 'val_dataset':
            for val_data in tqdm(val_loader, file=sys.stdout):
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
        elif str == 'test_dataset':
            for test_data in tqdm(test_loader, file=sys.stdout):
                test_images, test_labels = test_data
                outputs = model(test_images.to(device))
                outputs = torch.softmax(outputs, dim=1)
                outputs = torch.argmax(outputs, dim=1)
                confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=2)

    # The dataset absolute path
    parser.add_argument('--data-path', type=str,
                        default="/Users/tian/Downloads/thesis/Thesis/swin_transformer/dataset/Testing")

    # Training weight absolute path
    parser.add_argument('--weights', type=str, default='/Users/tian/Downloads/thesis/Thesis/swin_transformer/weights/model-6.pth',
                        help='initial weights path')
    
    # Whether to freeze weights
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt, str='test_dataset')
