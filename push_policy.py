import torch
import torch.nn as nn

class PushPolicy(nn.Module):
    def __init__(self, input_dim=7, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_bc(X, y, epochs=200, lr=1e-3):
    model = PushPolicy(input_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    loss_history = []
    for epoch in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if epoch % 50 == 0:
            print(f"epoch {epoch:4d}  loss {loss.item():.5f}")

    return model, loss_history