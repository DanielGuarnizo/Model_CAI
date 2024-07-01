# sweep_configuration = {
#     'method': 'random',  # or 'grid' or 'bayes'
#     'metric': {
#       'name': 'val_loss',
#       'goal': 'minimize'   
#     },
#     'parameters': {
#         'embedding_dim': {
#             'values': [32, 64, 128]
#         },
#         'num_layers': {
#             'values': [1, 2, 3]
#         },
#         'lr': {
#             'values': [1e-4, 1e-3, 1e-2]
#         },
#         'seq_len': {
#             'values': [10, 20, 30]
#         },
#         'n_epochs': {
#             'value': 150  # you can make this a parameter if you want to test different epoch lengths
#         },
#         'batch_size': {
#             'value': 1  # fixed value
#         }
#     }
# }

# Configuration with fixed values
sweep_configuration = {
    'method': 'random',  # or 'grid' or 'bayes'
    'metric': {
      'name': 'val_loss',
      'goal': 'minimize'   
    },
    'parameters': {
        'embedding_dim': {
            'value': 128
        },
        'num_layers': {
            'value': 1
        },
        'lr': {
            'value': 1e-4
        },
        'seq_len': {
            'value': 40
        },
        'n_epochs': {
            'value': 150
        },
        'batch_size': {
            'value': 1
        }
    }
}

import wandb

# Initialize the sweep
sweep_id = wandb.sweep(sweep_configuration, project='Autoencoder for MediaPipe Landmarks')