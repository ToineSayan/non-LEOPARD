import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle
import torch


def load_data(dataset_name, num_train_obs=None):
    if dataset_name == 'GloVe':
        X, Z_raw, Y_raw, X_val, Z_val_raw, Y_val_raw, X_test, Z_test_raw, Y_test_raw =  load_glove(num_train_obs)
    if dataset_name == 'biasbios':
        X, Z_raw, Y_raw, X_val, Z_val_raw, Y_val_raw, X_test, Z_test_raw, Y_test_raw = load_biasbios(num_train_obs)
    if dataset_name == "deepmoji":
        X, Z_raw, Y_raw, X_val, Z_val_raw, Y_val_raw, X_test, Z_test_raw, Y_test_raw = load_deepmoji(num_train_obs)

    
    if Z_raw is not None:
        z_values = sorted(np.unique(Z_raw)) # list of concept values
        z_id2label = {i:z_values[i] for i in range(len(z_values))}
        z_label2id = {z_values[i]:i for i in range(len(z_values))}
        Z = np.array([z_label2id[z] for z in Z_raw]).astype(int)
        Z_val = np.array([z_label2id[z] for z in Z_val_raw]).astype(int)
        Z_test = np.array([z_label2id[z] for z in Z_test_raw]).astype(int)
    else: 
        z_id2label = None
        Z, Z_val, Z_test =  None, None, None
 
    if Y_raw is not None:
        y_values = sorted(np.unique(Y_raw)) # list of concept values
        y_id2label = {i:y_values[i] for i in range(len(y_values))}
        y_label2id = {y_values[i]:i for i in range(len(y_values))}
        Y = np.array([y_label2id[y] for y in Y_raw]).astype(int)
        Y_val =  np.array([y_label2id[y] for y in Y_val_raw]).astype(int)
        Y_test = np.array([y_label2id[y] for y in Y_test_raw]).astype(int)
    else: 
        y_id2label = None
        Y, Y_val, Y_test =  None, None, None


    return X, Z, Y, X_val, Z_val, Y_val, X_test, Z_test, Y_test, z_id2label, y_id2label
    

def load_dump(file_path):
    """
    Load data from a .pkl file.

    Args:
        file_path (str): The path to the .pkl file.

    Returns:
        The data from the .pkl file.
    """
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def form_dataset(male_words, fem_words, neut_words):
    X, Y = [], []

    for w, v in male_words.items():
        X.append(v)
        Y.append(0)

    for w, v in fem_words.items():
        X.append(v)
        Y.append(1)
    
    for w, v in neut_words.items():
        X.append(v)
        Y.append(2)

    return np.array(X), np.array(Y)



def load_glove(num_train_obs=None):
    data_path="datasets/GloVe"

    male_words = load_dump(os.path.join(data_path, 'male_words.pkl'))
    fem_words = load_dump(os.path.join(data_path, 'fem_words.pkl'))
    neut_words = load_dump(os.path.join(data_path, 'neut_words.pkl'))

    X, Y = form_dataset(male_words, fem_words, neut_words)
    
    X_train_dev, X_test, y_train_dev, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=0)
    X_train, X_dev, Y_train, Y_dev = train_test_split(
        X_train_dev, y_train_dev, test_size=0.3, random_state=0)

    if num_train_obs is not None:
        if num_train_obs < X_train.shape[0]:
            X_train = X_train[:num_train_obs]
            Y_train = Y_train[:num_train_obs]


    return X_train, Y_train, None, X_dev, Y_dev, None, X_test, Y_test, None, 




def load_deepmoji(num_train_obs=None):
    data_path = "datasets/deepmoji/"

    nspc_train = 40000
    split = "train"
    X_neg_aa = np.load(data_path + split + '/neg_pos.npy') # sentiment : negative, language: AAE
    X_neg_wh = np.load(data_path + split + '/neg_neg.npy') # sentiment : negative, language: SAE
    X_pos_aa = np.load(data_path + split + '/pos_pos.npy') # sentiment : positive, language: AAE
    X_pos_wh = np.load(data_path + split + '/pos_neg.npy') # sentiment : positive, language: SAE

    X = np.concatenate((X_neg_aa[:nspc_train], X_neg_wh[:nspc_train], X_pos_aa[:nspc_train], X_pos_wh[:nspc_train]), axis = 0)
    z = np.array(nspc_train*[1] + nspc_train*[0] + nspc_train*[1] + nspc_train*[0]) # 0: SAE (Standard American English) - 1: AAE (African-American English)
    y = np.array(nspc_train*[0] + nspc_train*[0] + nspc_train*[1] + nspc_train*[1]) # 0: "sad", "1": "happy"

    nspc_val = 2000
    split = "dev"
    X_neg_aa = np.load(data_path + split + '/neg_pos.npy') # sentiment : negative, language: AAE
    X_neg_wh = np.load(data_path + split + '/neg_neg.npy') # sentiment : negative, language: SAE
    X_pos_aa = np.load(data_path + split + '/pos_pos.npy') # sentiment : positive, language: AAE
    X_pos_wh = np.load(data_path + split + '/pos_neg.npy') # sentiment : positive, language: SAE

    X_val = np.concatenate((X_neg_aa[:nspc_val], X_neg_wh[:nspc_val], X_pos_aa[:nspc_val], X_pos_wh[:nspc_val]), axis = 0)
    z_val = np.array(nspc_val*[1] + nspc_val*[0] + nspc_val*[1] + nspc_val*[0]) # 0: SAE (Standard American English) - 1: AAE (African-American English)
    y_val = np.array(nspc_val*[0] + nspc_val*[0] + nspc_val*[1] + nspc_val*[1]) # 0: "sad", "1": "happy"

    nspc_test = 1999 # not as in KRaM
    split = "test"
    X_neg_aa = np.load(data_path + split + '/neg_pos.npy') # sentiment : negative, language: AAE
    X_neg_wh = np.load(data_path + split + '/neg_neg.npy') # sentiment : negative, language: SAE
    X_pos_aa = np.load(data_path + split + '/pos_pos.npy') # sentiment : positive, language: AAE
    X_pos_wh = np.load(data_path + split + '/pos_neg.npy') # sentiment : positive, language: SAE

    X_test = np.concatenate((X_neg_aa[:nspc_test], X_neg_wh[:nspc_test], X_pos_aa[:nspc_test], X_pos_wh[:nspc_test]), axis = 0)
    z_test = np.array(nspc_test*[1] + nspc_test*[0] + nspc_test*[1] + nspc_test*[0]) # 0: SAE (Standard American English) - 1: AAE (African-American English)
    y_test = np.array(nspc_test*[0] + nspc_test*[0] + nspc_test*[1] + nspc_test*[1]) # 0: "sad", "1": "happy"

    # shuffle the observations in the sets
    np.random.seed(0)
    indices = np.arange(4*nspc_train)
    np.random.shuffle(indices)
    X, z, y = X[indices], z[indices], y[indices]

    indices = np.arange(4*nspc_val)
    np.random.shuffle(indices)
    X_val, z_val, y_val = X_val[indices], z_val[indices], y_val[indices]

    indices = np.arange(4*nspc_test)
    np.random.shuffle(indices)
    X_test, z_test, y_test = X_test[indices], z_test[indices], y_test[indices]

    print(f"""
        Stats: 
        - Num samples: {4*(nspc_train + nspc_val + nspc_test)}
        - train/val/test: {X.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}
        - proportion per split (z): 
            WH: {1-np.mean(z):.4}/{1-np.mean(z_val):.4}/{1-np.mean(z_test):.4}
            AA: {np.mean(z):.4}/{np.mean(z_val):.4}/{np.mean(z_test):.4}
        - proportion per split (y): 
            neg: {1-np.mean(y):.4}/{1-np.mean(y_val):.4}/{1-np.mean(y_test):.4}
            pos: {np.mean(y):.4}/{np.mean(y_val):.4}/{np.mean(y_test):.4}
        - conditional proportions (y | z):
            neg | WH: {1-np.mean(y[z==0]):.4}/{1-np.mean(y_val[z_val==0]):.4}/{1-np.mean(y_test[z_test==0]):.4}
            neg | AA: {1-np.mean(y[z==1]):.4}/{1-np.mean(y_val[z_val==1]):.4}/{1-np.mean(y_test[z_test==1]):.4}
            pos | WH: {np.mean(y[z==0]):.4}/{np.mean(y_val[z_val==0]):.4}/{np.mean(y_test[z_test==0]):.4}
            pos | AA: {np.mean(y[z==1]):.4}/{np.mean(y_val[z_val==1]):.4}/{np.mean(y_test[z_test==1]):.4}
    """)

    return X, z, y, X_val, z_val, y_val, X_test, z_test, y_test 









def load_biasbios(num_train_obs=None):
    data_path="datasets/biasbios"
    
    # Load the original train, validation, test split
    with np.load(os.path.join(data_path, f'D_original.npz')) as balanced_data:
        X, Z_raw, Y_raw = balanced_data['train_X'], balanced_data[f'train_Z_gender'], balanced_data['train_Y']
        X_test, Z_test_raw, Y_test_raw = balanced_data['test_X'], balanced_data[f'test_Z_gender'], balanced_data['test_Y']
        
        X_val, Z_val_raw, Y_val_raw =  balanced_data['validation_X'], balanced_data[f'validation_Z_gender'], balanced_data['validation_Y']


    if num_train_obs is not None:
        if num_train_obs < X.shape[0]:
            X = X[:num_train_obs]
            Z_raw = Z_raw[:num_train_obs]
            Y_raw= Y_raw[:num_train_obs]

    
    return X, Z_raw, Y_raw, X_val, Z_val_raw, Y_val_raw, X_test, Z_test_raw, Y_test_raw




