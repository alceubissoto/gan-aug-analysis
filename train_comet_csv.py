from itertools import islice
import os
from comet_ml import Experiment as Comet_Exp
import numpy as np
import pandas as pd
import pretrainedmodels as ptm
from sacred import Experiment
from sacred.observers import FileStorageObserver, TelegramObserver
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, sampler
from torchvision import models, datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
from dataset_loader import CSVDatasetWithName

from sacred.observers import RunObserver

np.set_printoptions(precision=4, suppress=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SetID(RunObserver):
    priority = 50  # very high priority

    def __init__(self, custom_id):
        self.custom_id = custom_id

    def started_event(self, ex_info, command, host_info, start_time,  config, meta_info, _id):
        return self.custom_id    # started_event should returns the _id 



ex = Experiment()
telegram_file = 'telegram.json'
if os.path.isfile(telegram_file):
    telegram_obs = TelegramObserver.from_config(telegram_file)
    ex.observers.append(telegram_obs)
fs_observer = FileStorageObserver.create('results-comet-gans')

@ex.config
def cfg():
    train_root = None
    train_csv = None
    val_root = None
    val_csv = None
    n_classes = 2
    exp_name = 'tmp'
    ex.observers.append(fs_observer)
    ex.observers.append(SetID(exp_name))
    epochs = 200  # maximum number of epochs
    batch_size = 32  # batch size
    num_workers = 8  # parallel jobs for data loading and augmentation
    model_name = 'inceptionv4'  # model: inceptionv4, densenet161, resnet152, senet154
    val_samples = 8  # number of samples per image in validation
    early_stopping_patience = 22  # patience for early stopping
    weighted_loss = False  # use weighted loss based on class imbalance
    balanced_loader = True  # balance classes in data loader
    lr = 0.001  # base learning rate
    exp_desc = ''
    ext = '.jpg'
    project_name = 'gan-skin'
    csv_image_field = 'image'
    csv_target_field = 'label'

def train_epoch(device, model, dataloaders, criterion, optimizer, phase,
                batches_per_epoch=None):
    losses = AverageMeter()
    accuracies = AverageMeter()
    all_preds = []
    all_labels = []
    if phase == 'train':
        model.train()
    else:
        model.eval()

    if batches_per_epoch:
        tqdm_loader = tqdm(
            islice(dataloaders['train'], 0, batches_per_epoch),
            total=batches_per_epoch)
    else:
        tqdm_loader = tqdm(dataloaders[phase])
    for data in tqdm_loader:
        (inputs, labels), name = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        if phase == 'train':
            optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        acc = torch.sum(preds == labels.data).item() / preds.shape[0]
        accuracies.update(acc)
        all_preds += list(F.softmax(outputs, dim=1).cpu().data.numpy())
        all_labels += list(labels.cpu().data.numpy())
        tqdm_loader.set_postfix(loss=losses.avg, acc=accuracies.avg)

    # Calculate multiclass AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_preds[:, 1])

    # Confusion Matrix
    print('\nConfusion matrix')
    cm = confusion_matrix(all_labels, all_preds.argmax(axis=1))
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    print(cmn)
    acc = np.trace(cmn) / cmn.shape[0]

    return {'loss': losses.avg, 'acc': acc, 'auc': auc}


def save_images(dataset, to, n=32):
    for i in range(n):
        img_path = os.path.join(to, 'img_{}.png'.format(i))
        save_image(dataset[i][0], img_path)


@ex.automain
def main(train_root, train_csv, val_root, val_csv, epochs, model_name, batch_size, 
         num_workers, val_samples, early_stopping_patience, n_classes, weighted_loss, 
         balanced_loader, lr, exp_name, exp_desc, ext, project_name, csv_image_field, csv_target_field,  _run):
    
    experiment = Comet_Exp(
        api_key="",
        project_name=project_name,
        workspace="",
    ) 

    tmp_write = open(exp_name + ".txt", "w")
    print(str(experiment.get_key()))
    tmp_write.write(str(experiment.get_key()))
    tmp_write.close()
 
    #comet logging
    experiment.log_parameter('train root', train_root)
    experiment.log_parameter('val root', val_root)
    experiment.log_parameter('train csv', train_csv)
    experiment.log_parameter('val csv', val_csv)
    experiment.log_parameter('model', model_name)
    experiment.log_parameter('weighted loss', weighted_loss)
    experiment.log_parameter('balanced loader', balanced_loader)
    experiment.log_parameter('learning rate', lr)
    experiment.log_parameter('batch size', batch_size)
    experiment.log_parameter('exp desc', exp_desc)
    experiment.set_name(exp_name)


    AUGMENTED_IMAGES_DIR = os.path.join(fs_observer.dir, 'images')
    CHECKPOINTS_DIR = os.path.join(fs_observer.dir, 'checkpoints')
    BEST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_best')
    LAST_MODEL_PATH = os.path.join(CHECKPOINTS_DIR, 'model_last')
    for directory in (AUGMENTED_IMAGES_DIR, CHECKPOINTS_DIR):
        os.makedirs(directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif model_name == 'inceptionv4':
        model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, n_classes)
    
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters())

    experiment.log_other('trainable params', trainable_params)
    experiment.log_other('total params', total_params)
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5*0.1),
        #transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        #transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(299, scale=(0.75, 1.0)),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5*0.1),
        #transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        #transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    
    experiment.log_parameter('augmentation', str(data_transforms))    
    #image_name, target
    train_ds = CSVDatasetWithName(
        train_root, train_csv, csv_image_field, csv_target_field,
        transform=data_transforms['train'], add_extension=ext, split=None)
    if val_root is not None:
        val_ds = CSVDatasetWithName(
            val_root, val_csv, csv_image_field, csv_target_field,
            transform=data_transforms['val'], add_extension=ext, split=None)
    else:
        val_ds = None


    datasets = {
        'train': train_ds,
        'val'  : val_ds,
    }

    if balanced_loader:
        data_sampler = sampler.WeightedRandomSampler(
            train_ds.sampler_weights, len(train_ds))
        shuffle = False
    else:
        data_sampler = None
        shuffle = True


    dl_train = DataLoader(datasets['train'], batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            sampler=data_sampler)
    if val_root is not None:
        dl_val = DataLoader(datasets['val'], batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            sampler=None)
    else:
        dl_val = None

    # Log dataset sample images to Comet
    num_samples = len(train_ds)
    for _ in range(10):
        value = np.random.randint(0, num_samples)
        data, name = train_ds.__getitem__(value)
        img = data[0].permute(1,2,0).numpy()
        experiment.log_image(img, name="{} TRAIN groundtruth:{}".format(name, data[1]))
    


    dataloaders = {
        'train': dl_train,
        'val': dl_val, 
    }

    if weighted_loss:
        print('Class weights')
        print(datasets['train'].class_weights_list)
        criterion = nn.CrossEntropyLoss(
            weight=torch.Tensor(datasets['train'].class_weights_list).cuda())
    else:
        criterion = nn.CrossEntropyLoss()


    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=0.001)

    if val_root is not None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,
                                                     min_lr=1e-5, patience=10)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[25],
                                                   gamma=0.1)

    metrics = {
        'train': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc']),
        'val': pd.DataFrame(columns=['epoch', 'loss', 'acc', 'auc'])
    }

    best_val_loss = 10000
    epochs_without_improvement = 0
    batches_per_epoch = None

    for epoch in range(epochs):
        print('train epoch {}/{}'.format(epoch+1, epochs))
        epoch_train_result = train_epoch(
            device, model, dataloaders, criterion, optimizer, 'train',
            batches_per_epoch)

        metrics['train'] = metrics['train'].append(
            {**epoch_train_result, 'epoch': epoch}, ignore_index=True)
        print('train', epoch_train_result)
        
        if val_root is not None:
            print('val epoch {}/{}'.format(epoch+1, epochs))
            epoch_val_result = train_epoch(
                device, model, dataloaders, criterion, optimizer, 'val', 
	        batches_per_epoch)

            metrics['val'] = metrics['val'].append(
                {**epoch_val_result, 'epoch': epoch}, ignore_index=True)
            print('val', epoch_val_result)

            scheduler.step(epoch_val_result['loss'])

            if epoch_val_result['loss'] < best_val_loss:
                best_val_loss = epoch_val_result['loss']
                epochs_without_improvement = 0
                torch.save(model, BEST_MODEL_PATH + '.pth')
                print('Best loss at epoch {}'.format(epoch))
            else:
                epochs_without_improvement += 1

            print('-' * 40)

            if epochs_without_improvement > early_stopping_patience:
                torch.save(model, LAST_MODEL_PATH + '.pth')
                break
        else:
            scheduler.step()

        metrics_comet = {'train/loss': epoch_train_result['loss'], 'train/auc': epoch_train_result['auc'], 'train/acc': epoch_train_result['acc'], 'val/loss': epoch_train_result['loss'], 'val/auc': epoch_val_result['auc'], 'val/acc': epoch_val_result['acc']}    
        experiment.log_metrics(metrics_comet, epoch=epoch+1)
        
        if epoch == (epochs-1):
            torch.save(model, LAST_MODEL_PATH + '.pth')

    if val_root is not None:
        phases = ['train', 'val']
    else:
        phases = ['train']

    for phase in phases:
        metrics[phase].epoch = metrics[phase].epoch.astype(int)
        metrics[phase].to_csv(os.path.join(fs_observer.dir, phase + '.csv'),
                              index=False)



    return experiment.get_key()
