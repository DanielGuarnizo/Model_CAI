import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def extract(df, seq_len):
    # Calculate the number of full sequences
    num_sequences = df.shape[0] // seq_len

    # Truncate the data to fit only complete sequences
    data_truncated = df.iloc[:num_sequences * seq_len]

    # Reshape the data
    reshaped_data = data_truncated.values.reshape((num_sequences, seq_len, df.shape[1]))

    # Convert reshaped data back to a DataFrame if needed
    reshaped_df = pd.DataFrame(reshaped_data.reshape(num_sequences, -1))
    print("Original shape:", df.shape)
    print("Truncated shape:", data_truncated.shape)
    print("Reshaped shape:", reshaped_data.shape)

    return reshaped_data



def create_dataset(df,seq_len):
    # Directly convert to a list of numpy arrays
    sequences = df.astype(np.float32).reshape(df.shape[0], seq_len, 132)
    
    # Convert the list of numpy arrays to a list of PyTorch tensors
    dataset = [torch.tensor(s).float() for s in sequences]

    # Stack the tensors along the default dimension (0)
    stacked_dataset = torch.stack(dataset)

    n_seq, seq_len, n_features = stacked_dataset.shape
    print("n_seq: ", n_seq)
    print("seq_len: ", seq_len)
    print("n_features: ", n_features)
    print("------------------")

    return stacked_dataset, seq_len, n_features



def load_data(batch_size, seq_len):
    # reshape the data to pass to the training model 
    path = "CSV_Dataset"
    data = {}
    for file in os.listdir(path):
        data[file.split(".")[0]] = extract(pd.read_csv(os.path.join(path, file)), seq_len)

    # we devide the dataset havinf 75% of training and 15% of val and test set 
    train_df, val_df = train_test_split(
    data["good"],
    test_size=0.15,
    random_state=RANDOM_SEED
    )

    # we devide the 15% dataset in to 66% of val test from the previous 15% and 33% of test set from the previous 15% 
    val_df, test_df = train_test_split(
    val_df,
    test_size=0.33, 
    random_state=RANDOM_SEED
    )

    anomaly_df = np.concatenate([v for k,v in data.items() if k != "good"], axis=0)

    train_dataset, seq_len, n_features = create_dataset(train_df,seq_len)
    val_dataset, _, _ = create_dataset(val_df, seq_len)
    test_normal_dataset, _, _ = create_dataset(test_df,seq_len)
    test_anomaly_dataset, _, _ = create_dataset(anomaly_df, seq_len)

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_normal_dataset = torch.utils.data.DataLoader(test_normal_dataset, batch_size=batch_size, shuffle=False)
    test_anomaly_dataset = torch.utils.data.DataLoader(test_anomaly_dataset, batch_size=batch_size, shuffle=False)

    return (train_dataset, seq_len, n_features, val_dataset, test_normal_dataset, test_anomaly_dataset, data)