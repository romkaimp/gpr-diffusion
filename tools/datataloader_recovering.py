import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def dataloader_reconstruction(batch_size: int=32,
                              datapath: str='../data/Lorenz/lorenz_data_rec0_1,0_0,1_1.npy',
                              test_datapath: str='../data/Lorenz/lorenz_data_rec_test0_1,0_0,1_1.npy',):
    data_rec = np.load(datapath)
    test_rec = np.load(test_datapath)
    print(data_rec.shape)

    class MyDatasetReconstruction(Dataset):
        def __init__(self, data_rec):
            self.data_rec = torch.from_numpy(data_rec)

        def __len__(self):
            return len(self.data_rec)

        def __getitem__(self, idx):
            return self.data_rec[idx]

    dataset = MyDatasetReconstruction(data_rec)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MyDatasetReconstruction(test_rec)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return dataloader, test_dataloader

def dataloader_prediction(batch_size=32,
                          datapath='lorenz_data_predict_train0_1,0_0,1_1.npy',
                          test_datapath='lorenz_data_predict_test0_1,0_0,1_1.npy',
                          val_datapath='lorenz_data_predict_train_val0_1,0_0,1_1.npy',
                          val_test_datapath='lorenz_data_predict_test_val0_1,0_0,1_1.npy',):
    data_pred = np.load(datapath)
    test_pred = np.load(test_datapath)

    val_data = np.load(val_datapath)
    val_test = np.load(val_test_datapath)

    class MyDatasetPrediction(Dataset):
        def __init__(self, data, target):
            self.data = data  # Загружаем X
            self.targets = target  # Загружаем y

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            x = self.data[idx]
            y = self.targets[idx]

            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            return x, y

    dataset = MyDatasetPrediction(data_pred, test_pred)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = MyDatasetPrediction(val_data, val_test)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return dataloader, val_dataloader