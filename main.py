import argparse
import random
import os
import shutil
import json

import pandas as pd

import h5py
import numpy as np
import tqdm
import torch
import torch.nn as nn
import wandb
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.metrics import precision_score, recall_score, f1_score

from ResNet import *

workspace = './'
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='MNIST_synthetic.h5')
parser.add_argument('--experiment_name', type=str, default='final')
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--momentum", type=float, default=0.99, help="learning rate")
parser.add_argument("--num_workers", type=int, default=16, help="number of cpu threads to use during batch generation")
parser.add_argument("--device", type=str, default='cuda', help='use or use not gpu')
parser.add_argument('--seed', type=int, default=230, help='fix random seed')
parser.add_argument('--load_type', type=str, default='best', choices=['best', 'last'])


class TrainDataset(Dataset):
    def __init__(self, imgs, gt, transform=None):
        self.transform = transform
        self.imgs = imgs
        self.gt = gt

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        gt = torch.from_numpy(self.gt[idx])
        gt_real = torch.zeros(size=(5 * 11,))
        for i, cls in enumerate(gt):
            gt_real[i * 11 + cls] = 1
        return image, gt_real


class TestDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.transform = transform
        self.imgs = imgs

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image


def same_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RunningAverage:
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)

    shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint_path, args, model, start_epoch=None, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise ("File doesn't exist {}".format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    if start_epoch:
        start_epoch = checkpoint['epoch'] - 1

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return start_epoch


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def train(model, criterion, opt, train_dl, args):
    train_metric = {}
    loss_avg = RunningAverage()
    model.train()

    o = []
    g = []
    with tqdm.tqdm(total=len(train_dl), desc='train loop') as t:
        for i, (img, gt) in enumerate(train_dl):
            # transfer data to gpu
            img = img.to(args.device)
            gt = gt.to(args.device)

            # output and loss
            output = model(img)
            loss = criterion(output, gt)

            # clear grad
            opt.zero_grad()

            loss.backward()
            # update param
            opt.step()

            # update loss_avg
            loss_avg.update(loss.item())
            del loss
            t.set_postfix(loss='{:05.4f}'.format(loss_avg()))
            t.update()

            # collect for computing metric
            output = output.detach().cpu()
            o.append(output.numpy())
            g.append(gt.cpu().numpy())

    o = np.concatenate(o)
    g = np.concatenate(g)
    metrics = calculate_metrics(o, g)
    train_metric = metrics
    train_metric['train_loss'] = loss_avg()
    return train_metric


def val(model, criterion, val_dl, args):
    val_metric = {}
    loss_avg = RunningAverage()
    model.eval()

    o = []
    g = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(val_dl), desc='val loop') as t:
            for i, (img, gt) in enumerate(val_dl):
                # transfer data to cuda
                img = img.to(args.device)
                gt = gt.to(args.device)

                # output and loss
                output = model(img)
                loss = criterion(output, gt)

                # update loss_avg
                loss_avg.update(loss.item())
                del loss
                t.set_postfix(loss='{:05.4f}'.format(loss_avg()))
                t.update()

                # collect for computing metric
                output = output.detach().cpu()
                o.append(output.numpy())
                g.append(gt.cpu().numpy())

    o = np.concatenate(o)
    g = np.concatenate(g)
    metrics = calculate_metrics(o, g)
    val_metric = metrics
    val_metric['val_loss'] = loss_avg()

    return val_metric


def train_and_evaluate(model, criterion, opt, train_dl, val_dl, args):
    # init writer
    # writer = SummaryWriter(os.path.join(workspace, args.experiment_name, 'summaries'))

    # init min_val_loss
    best_epoch = None
    min_val_loss = np.inf

    # sch = ReduceLROnPlateau(opt, mode='min', patience=4, factor=0.1, verbose=True)

    for epoch in tqdm.trange(args.n_epochs):
        train_metric = train(model, criterion, opt, train_dl, args)

        val_metric = val(model, criterion, val_dl, args)
        val_metric['epoch'] = epoch

        wandb.log({'train/micro/precision': train_metric['micro/precision'],
                   'train/micro/recall': train_metric['micro/recall'],
                   'train/micro/f1': train_metric['micro/f1'],
                   'train/macro/precision': train_metric['macro/precision'],
                   'train/macro/recall': train_metric['macro/recall'],
                   'train/macro/f1': train_metric['macro/f1'],
                   'train/samples/precision': train_metric['samples/precision'],
                   'train/samples/recall': train_metric['samples/recall'],
                   'train/samples/f1': train_metric['samples/f1'],
                   'train/train_loss': train_metric['train_loss'],
                   'val/micro/precision': val_metric['micro/precision'],
                   'val/micro/recall': val_metric['micro/recall'],
                   'val/micro/f1': val_metric['micro/f1'],
                   'val/macro/precision': val_metric['macro/precision'],
                   'val/macro/recall': val_metric['macro/recall'],
                   'val/macro/f1': val_metric['macro/f1'],
                   'val/samples/precision': val_metric['samples/precision'],
                   'val/samples/recall': val_metric['samples/recall'],
                   'val/samples/f1': val_metric['samples/f1'],
                   'val/val_loss': val_metric['val_loss'],
                   })

        is_best_loss = val_metric['val_loss'] <= min_val_loss

        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optim_dict': opt.state_dict(),
                         },
                        is_best=is_best_loss,
                        checkpoint=os.path.join(workspace, args.experiment_name),
                        )

        # save best loss
        if is_best_loss:
            min_val_loss = val_metric['val_loss']
            print('Found new best loss:', val_metric['val_loss'])

            best_json_path = os.path.join(workspace, args.experiment_name, "metrics_val_best_loss.json")
            save_dict_to_json(val_metric, best_json_path)

        # save last
        last_json_path = os.path.join(workspace, args.experiment_name, 'metrics_val_last_weights.json')
        save_dict_to_json(val_metric, last_json_path)

    print('best epoch : {}'.format(epoch))


def predict(model, test_dl, args):
    # load best
    load_checkpoint(os.path.join(workspace, args.experiment_name, args.load_type + '.pth.tar'), args, model)

    predict = []
    # predict
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_dl), desc='test loop') as t:
            for i, img in enumerate(test_dl):
                # transfer data to cuda
                img = img.to(args.device)
                # output
                output = model(img)
                output = output.detach().cpu().numpy()
                predict.append(output)

                t.update()

    predict = np.concatenate(predict)
    shape = predict.shape
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0
    predict = np.argwhere(predict == 1)

    # process res, save to csv and return
    res = {}
    for x in predict:
        if str(x[0]) not in res.keys():
            res[str(x[0])] = ''

        res[str(x[0])] = res[str(x[0])] + (str(x[1] % 11))
    res = pd.DataFrame.from_dict(res, orient='index').reset_index()
    res.columns = ['Id', 'Label']
    res.to_csv(os.path.join(workspace, args.experiment_name, 'res.csv'), index=False)
    return res


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            }


def e(model, val_dl, args, train=False):
    # load best
    load_checkpoint(os.path.join(workspace, args.experiment_name, args.load_type + '.pth.tar'), args, model)

    o = []
    g = []
    with torch.no_grad():
        with tqdm.tqdm(total=len(val_dl), desc='val loop') as t:
            for i, (img, gt) in enumerate(val_dl):
                # transfer data to cuda
                img = img.to(args.device)

                # output and loss
                output = model(img)
                output = output.detach().cpu()
                o.append(output.numpy())
                g.append(gt.numpy())

                t.update()

    o = np.concatenate(o)
    g = np.concatenate(g)
    metrics = calculate_metrics(o, g)
    metrics_path = None
    if train:
        metrics_path = os.path.join(workspace, args.experiment_name, 'metrics_train.json')
    else:
        metrics_path = os.path.join(workspace, args.experiment_name, 'metrics_val.json')
    save_dict_to_json(metrics, metrics_path)


if __name__ == '__main__':
    # Load args, fix seed, mkdir experiment_name
    args = parser.parse_args()
    same_seed(args.seed)
    if not os.path.exists(os.path.join(workspace, args.experiment_name)):
        os.mkdir(os.path.join(workspace, args.experiment_name))

    # wandb.init(config=args, project='comp551', name='multi-lable' + 'final')

    # Load data
    data = h5py.File(args.data_path, 'r')
    train_data = data['train_dataset'][()]
    train_gt = data['train_labels'][()]
    test_data = data['test_dataset'][()]

    # Preprocessing and get dataset
    custom_transform = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(np.mean(train_data)), std=np.std(train_data)),
        ToTensorV2()
    ])

    train_ds = TrainDataset(train_data, train_gt, custom_transform)
    test_ds = TestDataset(test_data, custom_transform)

    # Get dataloader(split get val)
    train_size = int(0.8 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=(args.device is 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(args.device is 'cuda'))
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=(args.device is 'cuda'))

    # model, loss, opt
    model = resnet18().to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss().to(args.device)

    # wandb.watch(model)
    # train
    # train_and_evaluate(model, criterion, opt, train_dl, val_dl, args)
    # test and save csv
    pre = predict(model, test_dl, args)
    # eval trainset
    # e(model, train_dl, args, True)
    # eval valset
    # e(model, val_dl, args)
