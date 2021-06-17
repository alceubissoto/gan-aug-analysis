import argparse
from comet_ml import ExistingExperiment
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision import datasets, models, transforms
import os
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from tqdm import tqdm
from scipy import misc
from dataset_loader import CSVDatasetWithName
from collections import OrderedDict  # pylint: disable=g-importing-member
import json

class AugmentOnTest:
    def __init__(self, dataset, n):
        self.dataset = dataset
        self.n = n

    def __len__(self):
        return self.n * len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i // self.n]


def main():
    exp = ExistingExperiment()
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the model')
    parser.add_argument('dataset', help='Path to dataset')
    parser.add_argument('csv', help='Path to csv')
    parser.add_argument('-n', type=int, default=50,
                        help='Number of image copies')
    parser.add_argument('--print-predictions', '-p', action='store_true',
                        help='Print the predicted value for each image')
    parser.add_argument('-jpg', action='store_true', help='Do not use labels, nor evaluate auc.')
    parser.add_argument('-name', help='Name of experiment')
    args = parser.parse_args()

    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5*0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    exp.log_parameter("test augmentation", str({'test': data_transform}))

    #image_name, None
    if args.jpg:
        train_ds = CSVDatasetWithName(
            os.path.join(args.dataset), os.path.join(args.csv), 'image', 'label',
            transform=data_transform, add_extension='.jpg', split=None)
    else:
        train_ds = CSVDatasetWithName(
            os.path.join(args.dataset), os.path.join(args.csv), 'image', 'label',
            transform=data_transform, add_extension='.png', split=None)

    dataset = AugmentOnTest(train_ds, args.n)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.n, shuffle=False, num_workers=8, pin_memory=True)

    name_of_dataset = args.name
    
    # Log dataset sample images to Comet
    num_samples = len(train_ds)
    for _ in range(10):
        value = np.random.randint(0, num_samples)
        data, name = train_ds.__getitem__(value)
        img = data[0].permute(1, 2, 0).numpy()
        exp.log_image(img, name="{} TEST {} groundtruth:{}".format(name, name_of_dataset, data[1]))
    
    # Normal (inceptionv4)
    model = torch.load(args.model)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_scores = []
    all_labels = []
    preds_dict = {}
    for data in tqdm(dataloader):
        (inputs, labels), name = data

        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())

        with torch.no_grad():
            outputs = model(inputs)
            scores = F.softmax(outputs, dim=1)[:, 1].cpu().data.numpy()

        preds_dict[name[0]] = scores.mean()
        all_scores.append(scores.mean())
        all_labels.append(labels.cpu().data[0].numpy())

    #if not args.submission:
    epoch_auc = roc_auc_score(all_labels, all_scores)
    print('auc: {}'.format(epoch_auc))
    exp.log_metric("auc " + name_of_dataset, epoch_auc)

    if args.print_predictions:
        for k, v in preds_dict.items():
            print("{},{}".format(k, v))

    exp.log_asset_data(json.dumps(str(preds_dict)), name=name_of_dataset)

if __name__ == '__main__':
    main()
