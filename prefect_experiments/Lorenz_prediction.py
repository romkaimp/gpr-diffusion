from prefect import task, flow

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import optuna as opt
from prefect.task_runners import ConcurrentTaskRunner
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from tools.datataloader_recovering import dataloader_reconstruction, dataloader_prediction
from tools.visual import lorenz_visualize
from models.DDPM1d import UNet, UNetTiny, UNetTinyDiff
from models.LSTM_AE import LSTMAutoencoder
import mlflow
from mlflow.models.signature import Schema, ColSpec, infer_signature
from mlflow.tracking.client import MlflowClient

import subprocess

@task(name="Get gpu stats")
def get_gpu_stats():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv,nounits,noheader']
    )
    return result.decode('utf-8').strip().split('\n')

@task(name="Importing data for prediction")
def import_data_pred(batch_size,
                     datapath='../data/Lorenz/lorenz_data_predict_train10_0,28_0,2_667.npy',
                    test_datapath='../data/Lorenz/lorenz_data_predict_test10_0,28_0,2_667.npy',
                    val_datapath='../data/Lorenz/lorenz_data_predict_train_val10_0,28_0,2_667.npy',
                    val_test_datapath='../data/Lorenz/lorenz_data_predict_test_val10_0,28_0,2_667.npy',) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    return dataloader_prediction(batch_size, datapath, test_datapath, val_datapath, val_test_datapath)

@task(name="Mocking model for prediction")
def mock_model():
    source_channel = 1
    unet_base_channel = 128
    num_norm_groups = 32
    covariate_dim = 16
    lr = 2e-4
    eps = 1e-8
    parameters = {'hidden_dim': 44, 'num_layers': 2, 'lr': 0.000434022706699842, 'batch_size': 16, 'start_factor': 0.06440284343289981, 'total_iters': 2719}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm_ae = LSTMAutoencoder(3, 16, 1)
    weights = torch.load('../experiments/Lorenz/best_lstm_ae.pth', weights_only=True)
    lstm_ae.load_state_dict(weights)
    lstm_ae.to(device)

    unet = UNet(1, 32, 8, 16, 32).to(device)

    opt = torch.optim.Adam(unet.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=parameters['start_factor'],
        end_factor=1.0,
        total_iters=parameters['total_iters'],)

    dataloader_train, dataloader_val = import_data_pred(32,)

    epochs = 1000

    for x, y in dataloader_train:
        x, y = x.to(device), y.to(device)
        _, z = lstm_ae.encode_zero(x)
        print("z_shape", z.shape)
        print("x_shape", x.shape)
        x, y = x.unsqueeze(1), y.unsqueeze(1)
        y_hat = unet(torch.zeros(32, 1, 32, 3).to(device), torch.zeros(32).to(device), torch.zeros(32, 16).to(device), torch.zeros(32, 32).to(device))
        print("y_shape", y_hat.shape)
        break
    # with mlflow.start_run(run_name='training DDPM Lorenz'):
    #     for epoch in range(epochs):
    #         for x, y in tqdm(dataloader_reconstruction(source_channel,)):

@task()
def check_alfa_bars():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T = 100
    alphas = torch.linspace(start=0.9999, end=0.75, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)
    print(sqrt_alpha_bars_t[-1])

@task(name="Training model for prediction")
def train_model(params):
    def generate_Chol_cov_matrix(obs_times: torch.Tensor, lmbda: float = 0.1, time_dim: int = 512) -> torch.Tensor:
        device = obs_times.device
        diff = obs_times.unsqueeze(2) - obs_times.unsqueeze(1)  # [batch_size, time_dim, time_dim]
        K = torch.exp(- (diff ** 2) / (2 * lmbda ** 2))

        assert not torch.isnan(K).any(), "K contains NaN"
        assert not torch.isinf(K).any(), "K contains Inf"
        # Добавляем малый шум, безопасно расширяя по батчу
        eye = torch.eye(time_dim, device=obs_times.device).unsqueeze(0).expand(K.size(0), -1, -1)
        K = K + 1e-3 * eye

        # Проверяем K

        L = torch.linalg.cholesky(K)
        return L

    source_channel = 1
    time_dim = 512
    prediction_length = 2048
    lmbda = params["lmbda"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('Lorenz_reconstruction_models/final.pth', map_location=device, weights_only=False)
    covariate_dim = checkpoint['hidden_dim']
    lstm_ae = LSTMAutoencoder(3, hidden_dim=checkpoint['hidden_dim'], num_layers=checkpoint['num_layers'])
    lstm_ae.load_state_dict(checkpoint['model_state_dict'])
    lstm_ae.to(device)
    lstm_ae.eval()
    for param in lstm_ae.parameters():
        param.requires_grad = False

    unet = UNetTiny(source_channel, params['unet_base_channel'], params['num_norm_groups'], covariate_dim, time_dim).to(device)

    opt = torch.optim.Adam(unet.parameters(), lr=params['lr'], eps=params['eps'], weight_decay=0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=params['start_factor'],
        end_factor=1.0,
        total_iters=params['total_iters']
    )

    # exp_gamma = (1/10) ** (1/(20-5)) ≈ 0.86
    exp_gamma = (1 / 10) ** (1 / 17)
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=exp_gamma)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[30]  # переключение после 30-й эпохи
    )
    shake_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=5, T_mult=2, eta_min=1e-6)

    dataloader_train, dataloader_val = import_data_pred(params['batch_size'],)

    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    betas = 1 - alphas
    sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

    epochs = 1000
    patience = 30
    best_val_loss = float("inf")
    patience_counter = 0
    cosine_scheduler_patience = 5

    with mlflow.start_run(run_name="Lorenz prediction"):
        input_example = {
            "y_t": torch.randn(1, 1, 32, 3).numpy(),
            "t": torch.randint(0, 100, (1,)).numpy(),
            "z": torch.randn(1, 16).numpy(),
            "normalized_obs_times": torch.rand(1, 32).numpy()
        }

        output_example = {
            "pred": torch.randn(1, 1, 32, 3).numpy()
        }

        signature = infer_signature(input_example, output_example)

        mlflow.log_params({
            "source_channel": source_channel,
            "unet_base_channel": params['unet_base_channel'],
            "num_norm_groups": params['num_norm_groups'],
            "covariate_dim": covariate_dim,
            "time_dim": time_dim,
            "lr": params['lr'],
            "eps": params['eps'],
            "batch_size": params['batch_size'],
            "prediction_length": prediction_length,
            "lmbda": lmbda,
            "epochs": epochs,
        })

        for epoch in tqdm(range(epochs)):
            unet.train()
            train_losses = []

            for batch_idx, (x, y) in enumerate(dataloader_train):
                batch_size = y.shape[0]
                #print("bs:", batch_size)
                opt.zero_grad()
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    _, z = lstm_ae.encode_zero(x) # z - [batch_size, covariate_dim] lstm_ae dependent

                obs_times = torch.sort(torch.randperm(prediction_length)[:time_dim]).values
                obs_times = obs_times.unsqueeze(0)
                normalized_obs_times = (obs_times.float().to(device) + 1) / prediction_length # in [~0.001, .., 1] ; 1 - 128/128 prediction place
                chol_cov_matrix = generate_Chol_cov_matrix(normalized_obs_times.to(device), lmbda=lmbda,
                                                           time_dim=time_dim).to(device)

                normalized_obs_times = normalized_obs_times.repeat(batch_size, 1)
                chol_cov_matrix = chol_cov_matrix.repeat(batch_size, 1, 1)
                # obs_times - [batch_size, time_dim] Prediction Window Length dependent (time_dim = pwl)

                _, time_dim = obs_times.shape
                #print("batch_size", batch_size, time_dim)
                y_true = y[torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, time_dim),
                obs_times.long()] # [batch_size, time_dim, 3] ; 3 - Lorenz dependent

                eps = torch.randn_like(y_true).to(device) # [batch_size, time_dim, 3]
                #chol_cov_matrix = generate_Chol_cov_matrix(normalized_obs_times.to(device), lmbda=lmbda, time_dim=time_dim).to(device)

                eps_corr = torch.matmul(chol_cov_matrix, eps).unsqueeze(1).to(device)

                y_true = y_true.unsqueeze(1)

                t = torch.randint(T, (batch_size,)).to(device)

                y_t = sqrt_alpha_bars_t[t][:, None, None, None].float() * y_true + sqrt_one_minus_alpha_bars_t[t][:, None,
                                                                                None, None].float() * eps_corr
               # print("y:", y_true.shape, "t:", t.shape, "z:", z.shape, "norm_T", normalized_obs_times.shape,)
                model_out = unet(y_t.to(device), t, z.to(device), normalized_obs_times)
                loss = F.mse_loss(model_out, eps.unsqueeze(1), reduction="mean")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                opt.step()

                train_losses.append(loss.item())

            epoch_train_loss = sum(train_losses) / len(train_losses)
            unet.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in dataloader_val:
                    x, y = x_val.to(device), y_val.to(device)
                    batch_size = y.shape[0]
                    _, z = lstm_ae.encode_zero(x)  # z - [batch_size, covariate_dim] lstm_ae dependent

                    obs_times = torch.stack([
                        torch.sort(torch.randperm(prediction_length)[:time_dim]).values
                        for _ in range(batch_size)
                    ])
                    # obs_times - [batch_size, time_dim] Prediction Window Length dependent (time_dim = pwl)
                    normalized_obs_times = (obs_times.float().to(device) + 1) / prediction_length  # in [~0.001, .., 1] ; 1 - 128/128 prediction place

                    batch_size, time_dim = obs_times.shape
                    #print("batch_size", batch_size, time_dim)
                    y_true = y[torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, time_dim),
                    obs_times.long()]

                    eps = torch.randn_like(y_true).to(device)  # [batch_size, time_dim, 3]
                    chol_cov_matrix = generate_Chol_cov_matrix(normalized_obs_times.to(device), lmbda=lmbda, time_dim=time_dim).to(
                        device)

                    eps_corr = torch.matmul(chol_cov_matrix, eps).unsqueeze(1).to(device)

                    y_true = y_true.unsqueeze(1)

                    t = torch.randint(T, (batch_size,)).to(device)

                    y_t = sqrt_alpha_bars_t[t][:, None, None, None].float() * y_true + sqrt_one_minus_alpha_bars_t[t][:,
                                                                                       None,
                                                                                       None, None].float() * eps_corr

                    model_out = unet(y_t.to(device), t, z.to(device), normalized_obs_times)
                    val_loss = F.mse_loss(model_out, eps.unsqueeze(1), reduction="mean")
                    val_losses.append(val_loss.item())

                epoch_val_loss = sum(val_losses) / len(val_losses)

                if epoch % 20 == 0 and epoch > 0:
                    for x_val, y_val in dataloader_val:
                        x, y_true = x_val[0, :, :], y_val[0, :time_dim, :]
                        lorenz_visualize(x.numpy(), y_true.numpy(), name=f"data on {epoch} epoch", ml_flow=True)

                        alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
                        sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
                        sigma_t = torch.sqrt(sigma_t_squared)

                        _, z = lstm_ae.encode_zero(x.unsqueeze(0).to(device)) # [1, covariate_dim]
                        y = y_true[None, :, :].to(device)

                        obs_times = torch.arange(time_dim)[None, :]  # [1, time_dim] time_dim индекса подряд
                        # [1, time_dim] \in (0; time_dim/prediction_length]
                        normalized_obs_times = (obs_times.float().to(device) + 1) / prediction_length

                        eps = torch.randn_like(y).to(device)  # [1, time_dim, 3]
                        chol_cov_matrix = generate_Chol_cov_matrix(normalized_obs_times.to(device), lmbda=lmbda,
                                                                   time_dim=time_dim).to(device)
                        X_t = torch.matmul(chol_cov_matrix, eps).unsqueeze(1).to(device) # [1, 1, time_dim, 3]

                        for t in reversed(range(T)):
                            if t > 0:
                                z_eps = torch.randn_like(y).to(device)
                                z_corr = torch.matmul(chol_cov_matrix, z_eps).unsqueeze(1).to(device)
                            else:
                                z_corr = torch.zeros_like(X_t).to(device)

                            t_tensor = torch.tensor(t).unsqueeze(0).to(device)
                            print("X_t", X_t.shape)
                            epsilon = unet(X_t, t_tensor, z.to(device), normalized_obs_times).squeeze(1)
                            print("epsilon", epsilon.shape)
                            X_t = (1.0 / torch.sqrt(alphas[t])).float() * (X_t - ((1.0 - alphas[t]) / sqrt_one_minus_alpha_bars_t[t]).float() * torch.matmul(chol_cov_matrix, epsilon)) + \
                sigma_t[t].float() * z_corr

                        X_0 = X_t.squeeze(0, 1)

                        lorenz_visualize(x.numpy(), X_0.cpu().numpy(), name=f"sample on {epoch} epoch", true=y_true.cpu().numpy(), ml_flow=True)

                        break

            for idx, gpu in enumerate(get_gpu_stats()):
                name, mem_total, mem_used, mem_free, util = gpu.split(', ')
                mlflow.log_metric(f"gpu_{idx}_{name}_mem_used", float(mem_used), step=epoch)
                mlflow.log_metric(f"gpu_{idx}_{name}_utilization", float(util), step=epoch)

            # === логгинг ===
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

            print(f"Epoch {epoch + 1}/{epochs} iter{batch_idx+1}- train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}")

                # сохраняем лучшую модель
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                use_cosine = False
                # mlflow.pytorch.log_model(
                #     unet,
                #     name="best_unet",
                #     pip_requirements=[
                #         "torch==2.5.1+cu121",
                #         "torchvision==0.20.1+cu121"
                #     ],
                #     signature=signature
                # )
                experiment_dir = "./Lorenz_prediction_models"
                torch.save({
                    'unet_base_channel': params['unet_base_channel'],
                    'num_norm_groups': params['num_norm_groups'],
                    'covariate_dim': 127,
                    'time_dim': 512,
                    'prediction_length': 2048,
                    'epoch': epoch,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                }, f'{experiment_dir}/checkpoint_epoch_{epoch}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.4f}")
                    break
            if patience_counter >= cosine_scheduler_patience and not use_cosine:
                use_cosine = True
            if use_cosine == True:
                shake_scheduler.step()
            scheduler.step()

@task(name="Training model for prediction without correlated epsilon")
def train_model_wo_corr(params):
    source_channel = 1
    time_dim = 512
    prediction_length = 2048
    lmbda = params["lmbda"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('Lorenz_reconstruction_models/final.pth', map_location=device, weights_only=False)
    covariate_dim = checkpoint['hidden_dim']
    lstm_ae = LSTMAutoencoder(3, hidden_dim=checkpoint['hidden_dim'], num_layers=checkpoint['num_layers'])
    lstm_ae.load_state_dict(checkpoint['model_state_dict'])
    lstm_ae.to(device)
    lstm_ae.eval()
    for param in lstm_ae.parameters():
        param.requires_grad = False

    unet = UNetTinyDiff(source_channel, params['unet_base_channel'], params['num_norm_groups'], covariate_dim).to(device)

    opt = torch.optim.Adam(unet.parameters(), lr=params['lr'], eps=params['eps'], weight_decay=0.01)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        opt,
        start_factor=params['start_factor'],
        end_factor=1.0,
        total_iters=params['total_iters']
    )

    # exp_gamma = (1/10) ** (1/(20-5)) ≈ 0.86
    exp_gamma = (1 / 10) ** (1 / 17)
    decay_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=exp_gamma)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[30]  # переключение после 30-й эпохи
    )

    dataloader_train, dataloader_val = import_data_pred(params['batch_size'],)

    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    betas = 1 - alphas
    sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

    epochs = 1000
    patience = 30
    best_val_loss = float("inf")
    patience_counter = 0

    with mlflow.start_run(run_name="Lorenz prediction"):
        input_example = {
            "y_t": torch.randn(1, 1, 32, 3).numpy(),
            "t": torch.randint(0, 100, (1,)).numpy(),
            "z": torch.randn(1, 16).numpy(),
            "normalized_obs_times": torch.rand(1, 32).numpy()
        }

        output_example = {
            "pred": torch.randn(1, 1, 32, 3).numpy()
        }

        signature = infer_signature(input_example, output_example)

        mlflow.log_params({
            "source_channel": source_channel,
            "unet_base_channel": params['unet_base_channel'],
            "num_norm_groups": params['num_norm_groups'],
            "covariate_dim": covariate_dim,
            "time_dim": time_dim,
            "lr": params['lr'],
            "eps": params['eps'],
            "batch_size": params['batch_size'],
            "prediction_length": prediction_length,
            "lmbda": lmbda,
            "epochs": epochs,
        })

        for epoch in tqdm(range(epochs)):
            unet.train()
            train_losses = []

            for batch_idx, (x, y) in enumerate(dataloader_train):
                batch_size = y.shape[0]
                opt.zero_grad()
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    _, z = lstm_ae.encode_zero(x) # z - [batch_size, covariate_dim] lstm_ae dependent

                #print("batch_size", batch_size, time_dim)
                y_true = y[:, :time_dim, :] # [batch_size, time_dim, 3] ; 3 - Lorenz dependent

                eps = torch.randn_like(y_true).to(device).unsqueeze(1) # [batch_size, time_dim, 3]

                y_true = y_true.unsqueeze(1)

                t = torch.randint(T, (batch_size,)).to(device)
                #print(y_true.shape, y_true.dtype)
                y_t = sqrt_alpha_bars_t[t][:, None, None, None].float() * y_true + sqrt_one_minus_alpha_bars_t[t][:, None,
                                                                                None, None].float() * eps
                #print(y_t.shape)
                model_out = unet(y_t.to(device), t, z.to(device))
                loss = F.mse_loss(model_out, eps, reduction="mean")
                loss.backward()
                opt.step()

                train_losses.append(loss.item())
            scheduler.step()

            epoch_train_loss = sum(train_losses) / len(train_losses)
            unet.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in dataloader_val:
                    x, y = x_val.to(device), y_val.to(device)
                    batch_size = y.shape[0]
                    _, z = lstm_ae.encode_zero(x)  # z - [batch_size, covariate_dim] lstm_ae dependent

                    #print("batch_size", batch_size, time_dim)
                    y_true = y[:, :time_dim, :]

                    eps = torch.randn_like(y_true).to(device).unsqueeze(1)  # [batch_size, time_dim, 3]

                    y_true = y_true.unsqueeze(1)

                    t = torch.randint(T, (batch_size,)).to(device)

                    y_t = sqrt_alpha_bars_t[t][:, None, None, None].float() * y_true + sqrt_one_minus_alpha_bars_t[t][:,
                                                                                       None,
                                                                                       None, None].float() * eps

                    model_out = unet(y_t.to(device), t, z.to(device))
                    val_loss = F.mse_loss(model_out, eps, reduction="mean")
                    val_losses.append(val_loss.item())

                epoch_val_loss = sum(val_losses) / len(val_losses)

                if epoch % 20 == 0 and epoch > 0:
                    for x_val, y_val in dataloader_val:
                        x, y_true = x_val[0, :, :], y_val[0, :time_dim, :]
                        lorenz_visualize(x.numpy(), y_true.numpy(), name=f"diffusion data {epoch}", true=None, ml_flow=True)

                        alpha_bars_prev = torch.cat((torch.ones(1).to(device), alpha_bars[:-1]))
                        sigma_t_squared = (1.0 - alphas) * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
                        sigma_t = torch.sqrt(sigma_t_squared)

                        _, z = lstm_ae.encode_zero(x.unsqueeze(0).to(device)) # [1, covariate_dim]
                        y = y_true[None, :, :].to(device)

                        X_t = torch.randn_like(y).to(device).unsqueeze(1) # [1, 1, time_dim, 3]

                        for t in reversed(range(T)):
                            if t > 0:
                                z_corr = torch.randn_like(X_t).to(device)
                            else:
                                z_corr = torch.zeros_like(X_t).to(device)

                            t_tensor = torch.tensor(t).unsqueeze(0).to(device)

                            epsilon = unet(X_t, t_tensor, z.to(device)).squeeze(1)
                            X_t = (1.0 / torch.sqrt(alphas[t])).float() * (X_t - ((1.0 - alphas[t]) / sqrt_one_minus_alpha_bars_t[t]).float() * epsilon) + \
                sigma_t[t].float() * z_corr

                        X_0 = X_t.squeeze(0, 1)

                        lorenz_visualize(x.numpy(), X_0.cpu().numpy(), name=f"prediction on {epoch}", true=y_true.cpu().numpy(), ml_flow=True)

                        break

            for idx, gpu in enumerate(get_gpu_stats()):
                name, mem_total, mem_used, mem_free, util = gpu.split(', ')
                mlflow.log_metric(f"gpu_{idx}_{name}_mem_used", float(mem_used), step=epoch)
                mlflow.log_metric(f"gpu_{idx}_{name}_utilization", float(util), step=epoch)

            # === логгинг ===
            mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

            print(f"Epoch {epoch + 1}/{epochs} iter{batch_idx+1}- train_loss={epoch_train_loss:.4f}, val_loss={epoch_val_loss:.4f}")

                # сохраняем лучшую модель
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                # mlflow.pytorch.log_model(
                #     unet,
                #     name="best_unet",
                #     pip_requirements=[
                #         "torch==2.5.1+cu121",
                #         "torchvision==0.20.1+cu121"
                #     ],
                #     signature=signature
                # )
                experiment_dir = "./Lorenz_prediction_models"
                torch.save({
                    'unet_base_channel': params['unet_base_channel'],
                    'num_norm_groups': params['num_norm_groups'],
                    'covariate_dim': 127,
                    'time_dim': 512,
                    'prediction_length': 2048,
                    'epoch': epoch,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                }, f'{experiment_dir}/checkpoint_epoch_{epoch}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.4f}")
                    break

@task(name="Leveraging parameters")
def leverage_parameters(trial):
    def generate_Chol_cov_matrix(obs_times: torch.Tensor, lmbda: float = 0.1, time_dim: int = 512) -> torch.Tensor:
        device = obs_times.device
        diff = obs_times.unsqueeze(2) - obs_times.unsqueeze(1)  # [batch_size, time_dim, time_dim]
        K = torch.exp(- (diff ** 2) / (2 * lmbda ** 2))

        assert not torch.isnan(K).any(), "K contains NaN"
        assert not torch.isinf(K).any(), "K contains Inf"
        # Добавляем малый шум, безопасно расширяя по батчу
        eye = torch.eye(time_dim, device=obs_times.device).unsqueeze(0).expand(K.size(0), -1, -1)
        K = K + 1e-3 * eye

        # Проверяем K

        L = torch.linalg.cholesky(K)
        return L
    # === Гиперпараметры, которые будет подбирать Optuna ===

    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    eps = trial.suggest_float("eps", 1e-9, 1e-7, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128])
    start_factor = trial.suggest_float("start_factor", 0.01, 0.2)
    total_iters = trial.suggest_int("total_iters", 1000, 5000)
    lmbda = trial.suggest_float("lmbda", 0.01, 0.1)

    # === Архитектура ===
    source_channel = 1
    unet_base_channel = trial.suggest_categorical("unet_base_channel", [32, 64])
    num_norm_groups = trial.suggest_categorical("num_norm_groups", [8, 16,])
    time_dim = 512
    prediction_length = 2048
    epochs = 5   # для Optuna можно сократить

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # LSTM AE
    checkpoint = torch.load('Lorenz_reconstruction_models/final.pth', map_location=device, weights_only=False)
    covariate_dim = checkpoint['hidden_dim']
    lstm_ae = LSTMAutoencoder(3, hidden_dim=checkpoint['hidden_dim'], num_layers=checkpoint['num_layers'])
    lstm_ae.load_state_dict(checkpoint['model_state_dict'])
    lstm_ae.to(device)
    lstm_ae.eval()
    for param in lstm_ae.parameters():
        param.requires_grad = False

    # UNet
    unet = UNetTiny(source_channel, unet_base_channel, num_norm_groups, covariate_dim, time_dim).to(device)

    # Optimizer & Scheduler
    opt = torch.optim.Adam(unet.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=start_factor, end_factor=1.0, total_iters=total_iters)

    # Данные
    dataloader_train, dataloader_val = import_data_pred(batch_size)

    # DDPM параметры
    T = 1000
    alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
    alpha_bars = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

    with mlflow.start_run():
        # Логируем гиперпараметры
        mlflow.log_params({
            "lr": lr,
            "eps": eps,
            "batch_size": batch_size,
            "start_factor": start_factor,
            "total_iters": total_iters,
            "lmbda": lmbda,
            "unet_base_channel": unet_base_channel,
            "num_norm_groups": num_norm_groups
        })

        best_val_loss = float("inf")

        for epoch in range(5):  # epochs
            unet.train()
            train_losses = []
            for x, y in dataloader_train:
                batch_size = y.shape[0]
                opt.zero_grad()
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    _, z = lstm_ae.encode_zero(x)

                obs_times = torch.stack([torch.sort(torch.randperm(prediction_length)[:time_dim]).values for _ in range(batch_size)])
                normalized_obs_times = (obs_times.float().to(device) + 1) / prediction_length
                y_true = y[torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, time_dim), obs_times.long()]
                eps_sample = torch.randn_like(y_true).to(device)
                chol_cov_matrix = generate_Chol_cov_matrix(normalized_obs_times, lmbda=lmbda)
                eps_corr = torch.matmul(chol_cov_matrix, eps_sample).unsqueeze(1).to(device)
                y_true = y_true.unsqueeze(1)

                t = torch.randint(T, (batch_size,)).to(device)
                y_t = sqrt_alpha_bars_t[t][:, None, None, None].float() * y_true + \
                      sqrt_one_minus_alpha_bars_t[t][:, None, None, None].float() * eps_corr

                model_out = unet(y_t, t, z, normalized_obs_times)
                loss = F.mse_loss(model_out, eps_sample.unsqueeze(1), reduction="mean")
                loss.backward()
                opt.step()
                scheduler.step()
                train_losses.append(loss.item())

            # Validation
            unet.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in dataloader_val:
                    x, y = x_val.to(device), y_val.to(device)
                    _, z = lstm_ae.encode_zero(x)

                    obs_times = torch.stack([torch.sort(torch.randperm(2048)[:512]).values for _ in range(y.shape[0])])
                    normalized_obs_times = (obs_times.float().to(device) + 1) / 2048
                    y_true = y[torch.arange(y.shape[0], device=device).unsqueeze(1).expand(-1, 512), obs_times.long()]
                    eps_sample = torch.randn_like(y_true).to(device)
                    chol_cov_matrix = generate_Chol_cov_matrix(normalized_obs_times, lmbda=lmbda)
                    eps_corr = torch.matmul(chol_cov_matrix, eps_sample).unsqueeze(1).to(device)
                    y_true = y_true.unsqueeze(1)

                    t = torch.randint(T, (y.shape[0],)).to(device)
                    y_t = sqrt_alpha_bars_t[t][:, None, None, None].float() * y_true + \
                          sqrt_one_minus_alpha_bars_t[t][:, None, None, None].float() * eps_corr

                    model_out = unet(y_t, t, z, normalized_obs_times)
                    val_loss = F.mse_loss(model_out, eps_sample.unsqueeze(1), reduction="mean")
                    val_losses.append(val_loss.item())

            epoch_val_loss = sum(val_losses) / len(val_losses)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

        # Логируем финальную модель
        mlflow.log_metric("best_val_loss", best_val_loss)

    return best_val_loss

@task(name="Check datasets")
def check_datasets():
    dataloader_train, dataloader_val = import_data_pred(32, )

    for x, y in dataloader_train:
        print(x.shape, y.shape)
        break

    for x, y in dataloader_val:
        print(x.shape, y.shape)
        break

@flow(name="Training model for prediction")
def training():
    #check_datasets()
    #check_alfa_bars()
    #mock_model

    mlflow.set_tracking_uri("http://localhost:5000/")
    # mlflow.set_experiment("Lorenz prediction leveraging parameters")
    # study = opt.create_study(direction='minimize')
    # study.optimize(leverage_parameters, n_trials=50, n_jobs=4)
    # with mlflow.start_run(run_name="optuna_final"):
    #     params = study.best_trial.params
    #     loss = study.best_trial.value
    #     mlflow.log_metric("loss", loss)
    #     mlflow.log_params(params)

    # mlflow.set_experiment("Lorenz prediction training")
    # params = {'lr': 0.033454540725569146,
    #           'batch_size': 128,
    #           'eps': 1.53085720159338e-06,
    #           'lmbda': 0.010858864968773526,
    #           #'num_norm_groups': 8,
    #           'num_norm_groups': 8,
    #           'start_factor': 0.18906120674276894,
    #           'total_iters': 2954,
    #           #'unet_base_channel': 64,
    #           'unet_base_channel': 64
    #           }
    # train_model(params)

    mlflow.set_experiment("Lorenz prediction wo corr training")
    params = {'lr': 0.003454540725569146,
              'batch_size': 128,
              'eps': 1.53085720159338e-06,
              'lmbda': 0.010858864968773526,
              # 'num_norm_groups': 8,
              'num_norm_groups': 8,
              'start_factor': 0.18906120674276894,
              'total_iters': 2954,
              # 'unet_base_channel': 64,
              'unet_base_channel': 64
              }
    train_model_wo_corr(params)

if __name__ == '__main__':
    training()