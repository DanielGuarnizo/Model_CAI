import wandb
from sweep_config import sweep_configuration
from train import main

sweep_id = wandb.sweep(sweep_configuration, project='Autoencoder for MediaPipe Landmarks')

def sweep_train():
    main()

wandb.agent(sweep_id, function=sweep_train)