import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model
from utils import read_test_data, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if os.path.exists('./weights') is False:
        os.mkdir('./weights')

    writer = SummaryWriter()

    test_images_path, test_images_label = read_test_data(args.data_path)

    img_size = 384
    data_transform = transforms.Compose([transforms.Resize(int(img_size * 1.14)),
                                         transforms.CenterCrop(img_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # testing dataset
    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform) 
    
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size>1 else 0, 8])    #number of workers 
    print('Using {} dataloader workers every process'.format(nw))
    # load test dataset
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn)
    
    # create model
    model = create_model(num_classes=args.num_classes).to(device)
    # load model weights
    model_weight_path = '/Users/tian/Downloads/thesis/Thesis/swin_transformer/weights/model-6.pth'
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    for epoch in range(args.epochs):
        test_loss, test_acc = evaluate(model=model,
                                       data_loader=test_loader,
                                       device=device,
                                       epoch=epoch,
                                       str='test_loader')

    tags = ['test_loss', 'test_acc']
    writer.add_scalar(tags[0], test_loss, epoch)
    writer.add_scalar(tags[1], test_acc, epoch)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=str, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)

    # Root directory of the test dataset
    parser.add_argument('--data-path', type=str,
                        default='/Users/tian/Downloads/thesis/Thesis/swin_transformer/dataset/Testing')
    
    
    opt = parser.parse_args()

    main(opt)
        