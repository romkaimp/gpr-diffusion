import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        self.decoder = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim)
        return: reconstruction (batch, seq_len, input_dim), latent (batch, hidden_dim)
        """

        _, h_n = self.encoder(x)  # h_n: (num_layers, batch, hidden_dim)
        latent = h_n[-1]          # берём последнее скрытое (batch, hidden_dim)

        seq_len = x.size(1)
        latent_repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq_len, hidden_dim)

        dec_out, _ = self.decoder(latent_repeated)  # (batch, seq_len, hidden_dim)

        recon = self.output_layer(dec_out)  # (batch, seq_len, input_dim)

        return recon, latent

    def encode(self, x, h_0):
        _, h_n = self.encoder(x, h_0)  # h_n: (num_layers, batch, hidden_dim)
        latent = h_n[-1]

        return h_n, latent

    # Пример использования
if __name__ == "__main__":
    # Параметры
    seq_len = 50
    batch_size = 16
    input_dim = 1

    # Создадим модель и данные
    model = LSTMAutoencoder(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Случайные данные для примера
    x = torch.randn(batch_size, seq_len, input_dim)

    # Прямой проход
    recon, z = model(x)

    # Потеря и шаг оптимизации
    loss = criterion(recon, x)
    loss.backward()
    optimizer.step()

    #h_0 = torch.randn(1*1, batch_size, 32)

    #h_n, latent = model.encode(x, h_0)
    #print(h_n.shape, latent.shape)

    print("Output shape:", recon.shape)
    print("Latent vector shape:", z.shape)

