from prefect import task, flow

import numpy as np
import torch
from typing import Tuple
import optuna as opt
from prefect.task_runners import ConcurrentTaskRunner
from tqdm import tqdm

from tools.datataloader_recovering import dataloader_reconstruction, dataloader_prediction
from models.DDPM1d import UNet
from models.LSTM_AE import LSTMAutoencoder
import mlflow
from tools.visual import lorenz_visualize

@task(name="Importing data")
def import_data_rec(batch_size,
                    datapath: str="../data/Lorenz/lorenz_data_rec10_0,28_0,2_667.npy",
                    test_datapath: str='../data/Lorenz/lorenz_data_rec_test10_0,28_0,2_667.npy') -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return dataloader_reconstruction(batch_size, datapath, test_datapath)

@task(name="Training")
def reconstruction_training(parameters):
    #parameters = {'hidden_dim': 44, 'num_layers': 2, 'lr': 0.000434022706699842, 'batch_size': 16,
    #             'start_factor': 0.06440284343289981, 'total_iters': 2719}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = 3

    dataloader, val_dataloader = import_data_rec(batch_size=parameters['batch_size'],)

    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=parameters['hidden_dim'], num_layers=parameters['num_layers'],).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'], eps=1e-8, weight_decay=0.001)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=parameters['start_factor'],
        end_factor=1.0,
        total_iters=parameters['total_iters']
    )

    # exp_gamma = (1/10) ** (1/(20-5)) ≈ 0.86
    exp_gamma = (1 / 10) ** (1 / 17)
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp_gamma)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[30]  # переключение после 5-й эпохи
    )
    criterion = torch.nn.MSELoss()

    epochs = 300
    patience_counter = 0
    patience = 30
    best_val_loss = float("inf")

    with mlflow.start_run(run_name="training model for reconstruction"):
        for epoch in tqdm(range(epochs)):

            # training
            model.train()
            running_loss = 0.0
            # for x in tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            for batch_idx, x in enumerate(dataloader):
                x = x.to(device).float()
                optimizer.zero_grad()
                y, _ = model(x)
                loss = criterion(y, torch.flip(x, dims=[1]))
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            epoch_train_loss = running_loss / len(dataloader.dataset)
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                # for x in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                for x in val_dataloader:
                    x = x.to(device).float()
                    y, _ = model(x)
                    loss = criterion(y, torch.flip(x, dims=[1]))
                    val_loss += loss.item() * x.size(0)

            epoch_val_loss = val_loss / len(val_dataloader.dataset)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            if epoch < 50:
                scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs} iter{batch_idx + 1}- train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}")
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0

                experiment_dir = "./Lorenz_reconstruction_models"
                torch.save({
                    'hidden_dim': parameters['hidden_dim'],
                    'num_layers': parameters['num_layers'],
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                }, f'{experiment_dir}/checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'hidden_dim': parameters['hidden_dim'],
                    'num_layers': parameters['num_layers'],
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                }, f'{experiment_dir}/final.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.4f}")
                    break

@task(name="Checking pictures")
def check_pictures():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('Lorenz_reconstruction_models/final.pth', map_location=device, weights_only=False)
    lstm_ae = LSTMAutoencoder(3, hidden_dim=checkpoint['hidden_dim'], num_layers=checkpoint['num_layers'])
    lstm_ae.load_state_dict(checkpoint['model_state_dict'])
    lstm_ae.to(device)
    lstm_ae.eval()
    for param in lstm_ae.parameters():
        param.requires_grad = False

    dataloader, val_dataloader = import_data_rec(batch_size=2,)

    with torch.no_grad():
        for x in val_dataloader:
            x = x.to(device).float()
            y, _ = lstm_ae(x)
            break
    lorenz_visualize(x[0, ::3, :].cpu().numpy(), y[0, :, :].cpu().numpy(), ml_flow=False)
    print("x:", x[0, ::3, :].cpu().numpy())
    print("y:", y[0, ::3, :].cpu().numpy())

@task(name="Check data")
def check_data():
    dataloader, val_dataloader = import_data_rec(batch_size=2)
    for x in dataloader:
        x1 = x[0, :, :].numpy()
        x2 = x[1, :, :].numpy()
        lorenz_visualize(x1, x2, ml_flow=False, name="data for reconstruction")
        break

    for x in val_dataloader:
        x1 = x[0, :, :].numpy()
        x2 = x[1, :, :].numpy()
        lorenz_visualize(x1, x2, ml_flow=False, name="data for reconstruction validation")

@task(name="Training model for reconstruction")
def object_trainer(trial):
    #parameters definition
    input_dim = 3
    hidden_dim = trial.suggest_int("hidden_dim", 1, 128)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    start_factor = trial.suggest_float("start_factor", 1e-5, 0.5, log=True)
    total_iters = trial.suggest_int("total_iters", 1000, 5000)

    dataloader, val_dataloader = import_data_rec(batch_size=batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,
        end_factor=1.0,
        total_iters=total_iters
    )
    criterion = torch.nn.MSELoss()

    epochs = 10
    with mlflow.start_run():
        mlflow.log_param("hidden_size", hidden_dim)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("lr", lr)
        for epoch in range(epochs):

            #training
            model.train()
            running_loss = 0.0
            #for x in tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            for x in dataloader:
                x = x.to(device).float()
                optimizer.zero_grad()
                y, _ = model(x)
                loss = criterion(y, x)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)

            epoch_train_loss = running_loss / len(dataloader.dataset)
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)

            # validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                #for x in tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                for x in val_dataloader:
                    x = x.to(device).float()
                    y, _ = model(x)
                    loss = criterion(y, x)
                    val_loss += loss.item() * x.size(0)

            epoch_val_loss = val_loss / len(val_dataloader.dataset)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

            scheduler.step()

    return epoch_val_loss

@flow(#task_runner=ConcurrentTaskRunner(),
    name="Training models")
def training():

    mlflow.set_tracking_uri("http://localhost:5000/")
    # mlflow.set_experiment("Lorenz_reconstruction")
    # study = opt.create_study(direction='minimize')
    # study.optimize(object_trainer, n_trials=50, n_jobs=8)
    # with mlflow.start_run(run_name="optuna_final"):
    #     params = study.best_trial.params
    #     loss = study.best_trial.value
    #     mlflow.log_metric("loss", loss)
    #     mlflow.log_params(params)

    # mlflow.set_experiment("Lorenz reconstruction training")
    # params=  {'hidden_dim': 128, 'num_layers': 3, 'lr': 0.003434022706699842, 'batch_size': 16,
    #             'start_factor': 0.06440284343289981, 'total_iters': 2719}
    # reconstruction_training(params)

    check_pictures()
    #check_data()

    #[I 2025-10-02 22:57:14,669] Trial 45 finished with value: 2.9185908709882735e-07 and parameters: {'hidden_dim': 44, 'num_layers': 2, 'lr': 0.000434022706699842, 'batch_size': 16, 'start_factor': 0.06440284343289981, 'total_iters': 2719}. Best is trial 45 with value: 2.9185908709882735e-07.


if __name__ == '__main__':
    training()