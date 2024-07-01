import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from architecture import RecurrentAutoencoder
from load_data import load_data

def train_model(model, train_dataset, val_dataset, n_epochs, device, lr, early_stopping_patience=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for seq_true in train_dataset:
            optimizer.zero_grad()
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_losses = []
        model.eval()
        with torch.no_grad():
            for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        wandb.log({"val_loss": val_loss, "train_loss": train_loss})

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_wts)
    return model.eval(), history

def main():
    wandb.init()

    config = wandb.config
    batch_size = config.batch_size
    n_epochs = config.n_epochs
    embedding_dim = config.embedding_dim
    num_layers = config.num_layers
    seq_len = config.seq_len
    lr = config.lr

    # load data
    train_dataset, seq_len, n_features, val_dataset, test_normal_dataset, test_anomaly_dataset, data = load_data(batch_size=batch_size, seq_len=seq_len)
    
    # Define Model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentAutoencoder(seq_len, n_features, embedding_dim, num_layers, device=device)
    model = model.to(device)

    # Start Training
    model, history = train_model(
        model,
        train_dataset,
        val_dataset,
        n_epochs=n_epochs,
        device=device,
        lr=lr,
        early_stopping_patience=10  # Adjust this value as needed
    )

    # Save model with parameter values in the filename
    model_filename = f"{batch_size}_{embedding_dim}_{lr}_{n_epochs}_{num_layers}_{seq_len}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    main()