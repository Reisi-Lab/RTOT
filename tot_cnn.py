import numpy as np
import pandas as pd
import pytorch_lightning as pl
import tensorly as tl
import functional as F
import plotly.express as px
import HL_data_io as io
import torch
import pickle
import os
import time

from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict
from itertools import product


class TOTCNN(pl.LightningModule):
    def __init__(self, **params):
        super().__init__()
        self.dims = params['dims']
        self.L = params['L']
        self.M = params['M']
        self.N = params['N']
        self.R = params['R']
        self.name = 'totcnn'

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=4)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4)
        self.conv4 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4)
        self.conv5 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4)
        if self.dims[-1] * self.dims[-2] == 203:
            self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(29, 1))
        elif self.dims[-1] * self.dims[-2] == (37 * 121):
            self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=1, padding=(14, 57), kernel_size=(2, 2))
        else:
            self.conv6 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(11, 11))

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.SELU()(self.conv2(x))
        x = torch.nn.SELU()(self.conv3(x))
        x = torch.nn.SELU()(self.conv4(x))
        x = torch.nn.SELU()(self.conv5(x))
        x = self.conv6(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pre = self(x)
        y_pre = y_pre - torch.mean(y_pre, dim=(2, 3), keepdim=True) + torch.mean(y, dim=(2, 3), keepdim=True)
        loss = torch.norm(y_pre - y) / torch.norm(y)
        return {'loss': loss}

    def training_epoch_end(self, outputs) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('loss_training', avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pre = self(x)
        y_pre = y_pre - torch.mean(y_pre, dim=(2, 3), keepdim=True) + torch.mean(y, dim=(2, 3), keepdim=True)
        loss = torch.norm(y_pre - y) / torch.norm(y)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.log('loss_valid', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-2)
        return optimizer


class CNNDataset(Dataset):
    def __init__(self, **params):
        self.dims = params['dims']
        self.y = params['y']
        self.x = params['x']

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.x[idx].reshape(1, self.dims[0], self.dims[1]), self.y[idx].reshape(1, self.dims[2], self.dims[3])
        return torch.tensor(x).float(), torch.tensor(y).float()


if __name__ == '__main__':
    replications = 20
    max_epochs = 100
    train_batch_size = 400
    test_batch_size = 100

    for percentile, Ru in product([0, .03, .05, .1, .15], [5, 7, 9]):
        folder = r'./experiment-results-2/sync-data-normal-{}-{}/'.format(percentile, Ru)
        logFolder = './cnn-2/log-sync-data-normal-{}-{}'.format(percentile, Ru)
        modelFolder = './cnn-2/model-sync-data-normal-{}-{}'.format(percentile, Ru)
        list_params = pickle.load(open(os.path.join(folder, 'list_params-p={}.p.split'.format(percentile)), 'rb'))
        dict_Bs = defaultdict(list)
        dict_rpes = defaultdict(list)

        for r in range(replications):
            params = list_params[r]
            data_train = CNNDataset(x=params['x'], y=params['y'], dims=params['dims'])
            data_test = CNNDataset(x=params['x_test'], y=params['y_test'], dims=params['dims'])
            trainLoader = DataLoader(dataset=data_train, batch_size=train_batch_size)
            testLoader = DataLoader(dataset=data_test, shuffle=False, batch_size=test_batch_size)
            dims = params['dims']

            print('============')

            logger = TensorBoardLogger(
                logFolder,
                name='log-{}'.format(r)
            )

            ckpt = ModelCheckpoint(
                monitor='loss_valid',
                dirpath=modelFolder,
                filename='model-{}-'.format(r) + '{loss_valid:.2f}',
                save_top_k=1,
                mode='min',
                period=1,
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                gpus=1,
                logger=logger,
                callbacks=[ckpt]
            )
            model = TOTCNN(**params)
            start = time.time()
            trainer.fit(model, train_dataloader=trainLoader, val_dataloaders=testLoader)
            end = time.time()

            torch.set_grad_enabled(False)
            model.eval()
            y_test = tl.partial_tensor_to_vec(params['y_test'], skip_begin=1)
            y_pre = model(torch.tensor(params['x_test']).reshape(-1, 1, dims[0], dims[1]).float()).cpu().numpy()
            y_pre = tl.partial_tensor_to_vec(y_pre, skip_begin=1)
            m_test = np.mean(y_test, axis=1).reshape(-1, 1)
            m_pre = np.mean(y_pre, axis=1).reshape(-1, 1)
            y_pre = y_pre - m_pre + m_test

            # data = dict(
            #     y=sum([y_pre[n].flatten().tolist() for n in range(len(y_pre))], []) + sum([y_test[n].flatten().tolist() for n in range(len(y_test))], []),
            #     time=[i for i in range(1, 204)] * 2 * len(y_pre),
            #     type=[model.name for _ in range(203)] * len(y_pre) + ['true' for _ in range(203)] * len(y_test),
            #     sample=[i for i in range(len(y_pre)) for _ in range(203)] + [i for i in range(len(y_pre)) for _ in range(203)]
            # )
            # df = pd.DataFrame(data=data)
            # fig = px.line(df, x='time', y='y', color='sample', line_group='type', line_dash='type')
            # fig.show()

            rpe = np.linalg.norm(y_pre - y_test) / np.linalg.norm(y_test)
            dict_rpes[model.name].append(rpe)
            msg = 'ru={}, replication={}, model={}, rpe={:.4f}, time={:.4f}'.format(Ru, r + 1, model.name, rpe, end - start)
            print(msg)
            print('============')
            torch.set_grad_enabled(True)

        if not os.path.exists(folder): os.makedirs(folder)
        pickle.dump(dict_rpes, open(os.path.join(folder, 'dict_rpes-p={}.p.split.cnn'.format(percentile)), 'wb'))
        print('done')
