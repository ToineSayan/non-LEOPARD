import torch
import numpy as np
import random
import yaml


from loss import init_loss
from eraser import init_model
from eval import *
from load_data import load_data

from training import train



##############################################################
# Process arguments
##############################################################

list_cfgs = [


    # ------------------------------------------------------------------
    # GloVe



    # {'dataset_name': 'GloVe', 'out_name': "FaRM", 'eraser_model': 'MLP', "loss_name": 'FaRM', 'hidden_sizes': [300, 300, 300], "learning_rate": 1.0e-3, 'num_epochs':500, 'scheduler_milestones': [400], "batch_size": 1024}, # OK 
    # {'dataset_name': 'GloVe', 'out_name': "KRaM", 'eraser_model': 'MLP', "loss_name": 'KRaM', 'hidden_sizes': [300, 300, 300], "learning_rate": 1.0e-3, 'num_epochs':500, 'scheduler_milestones': [400], "batch_size": 1024}, # OK 

    {'dataset_name': 'GloVe', 'out_name': "cascade_150", 'eraser_model': 'cascade_projection', 'projection_rank': 150,  "weight_decay": 0.0, "loss_name": "MMD", "gamma":200, "learning_rate": 1.0e-3, 'num_epochs':1000, 'scheduler_milestones': [500], 'batch_size': 10777, "rbf_n_kernels": 5}, 
    {'dataset_name': 'GloVe', 'out_name': "cascade_150", 'eraser_model': 'projection', 'projection_rank': 150,  "weight_decay": 0.0, "loss_name": "MMD", "gamma":200, "learning_rate": 1.0e-3, 'num_epochs':1000, 'scheduler_milestones': [500], 'batch_size': 10777, "rbf_n_kernels": 5}, 


    # ------------------------------------------------------------------
    # Bias in bios

    # {'dataset_name': 'biasbios', 'out_name': "FaRM", "loss_name": "FaRM", 'eraser_model': 'MLP', 'hidden_sizes': [768, 768, 768], 'batch_size':512, 'num_epochs':50, 'learning_rate': 0.001},
    # {'dataset_name': 'biasbios', 'out_name': "KRaM", "loss_name": "KRaM", 'eraser_model': 'MLP', 'hidden_sizes': [768, 768, 768], 'batch_size':512, 'num_epochs':50, 'learning_rate': 0.001},


    # {'dataset_name': 'biasbios', 'out_name': "LEACE", 'eraser_model': 'linear', 'num_epochs':0},


    {'dataset_name': 'biasbios', 'out_name': "cascade_50", 'eraser_model': 'cascade_projection', 'projection_rank': 50, 'batch_size': 8192, "loss_name": "MMD", "gamma":100, 'num_epochs':100, 'scheduler_milestones': [50], "rbf_n_kernels": 5}, 
    {'dataset_name': 'biasbios', 'out_name': "cascade_250", 'eraser_model': 'cascade_projection', 'projection_rank': 250, 'batch_size': 8192, "loss_name": "MMD", "gamma":100, 'num_epochs':100, 'scheduler_milestones': [50], "rbf_n_kernels": 5}, 


    # ------------------------------------------------------------------
    # Deepmoji (DIAL)

    # {'dataset_name': 'deepmoji', "loss_name": 'FaRM','out_name': "FaRM", 'eraser_model': 'MLP', 'hidden_sizes': [300, 300, 300, 300, 300, 300], 'num_epochs':50, 'learning_rate':0.001, 'batch_size': 2048},
    # {'dataset_name': 'deepmoji', "loss_name": 'KRaM','out_name': "KRaM", 'eraser_model': 'MLP', 'hidden_sizes': [300, 300, 300, 300, 300, 300], 'num_epochs':50, 'learning_rate':0.001, 'batch_size': 2048},

    {'dataset_name': 'deepmoji', 'out_name': "out", 'eraser_model': 'cascade_projection', 'projection_rank': 15, "weight_decay": 0.0, "loss_name": "MMD", "gamma":100, "batch_size": 2048, "rbf_n_kernels": 5}, # OK
    {'dataset_name': 'deepmoji', 'out_name': "out", 'eraser_model': 'projection', 'projection_rank': 15, "weight_decay": 0.0, "loss_name": "MMD", "gamma":100, "batch_size": 2048, "rbf_n_kernels": 5},




]   


for table_cfg in list_cfgs: # For each config, train the model
    args = {}
    for k,v in table_cfg.items():
        args[k] = v
    base_path = "./"
    # config_file = base_path + f'configs/{args["dataset_name"]}/base_config.yml'
    config_file = base_path + f'configs/base_config.yml'
    data_path = base_path + f'datasets/{args["dataset_name"]}'
    output_path = base_path + 'outputs'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ##############################################################
    # Load config and set parameters
    ##############################################################

    with open(config_file, 'r') as file: # most of the hyperparameters are defined in a basic config file (per dataset)
        cfg = yaml.safe_load(file)
    for k in args.keys(): # some hyperparameters are updated for the current run
        # if k in cfg.keys():
        cfg[k] = args[k]

    seed = cfg['seed']
    dataset_name = cfg['dataset_name']

    # print(args)
    print(cfg)
    print("----------------")
    # quit()



    # Output path where models will be stored
    id = f'{dataset_name}_{args["out_name"]}'
    out_name = '/eraser_' + id
    MODEL_PATH = output_path + out_name

    ##############################################################
    # Control Randomness
    ##############################################################

    # control of randomness
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    ##############################################################
    # Data loading
    ##############################################################

    # if None, use the full train observations.
    num_train_obs = None

    X, Z, _, _, _, _, _, _, _, z_id2label, _ = load_data(dataset_name)
    n, d = X.shape # n: number of observations in the train set, d: number of features
    k = len(z_id2label.keys()) # number of concept values

    print("num observations (train):", n)
    print("num features:", d)

    ##############################################################
    # Model training (erasure)
    ##############################################################


    # Initialize the model
    model = init_model(X = torch.tensor(X), Z = torch.tensor(Z), **cfg)
    model = model.to(device)

    # Erasure loss 
    erasure_criterion = init_loss(**cfg) 

    train(
        model, 
        erasure_criterion,
        torch.tensor(X),
        torch.tensor(Z),
        device = device,
        gradient_acumulation_steps = 1,
        **cfg
    )

    ##############################################################
    # Model evaluation 
    ##############################################################
    model.eval()
    if cfg["eraser_model"] in ['projection', 'cascade_projection']: 
        proj_save = {"P_": model.get_projector(approximation=True).detach().cpu().numpy(), "P": model.get_projector().detach().cpu().numpy()}
        np.save(MODEL_PATH, proj_save)

    # np.load(MODEL_PATH + ".npy", allow_pickle=True).item().get("P")

    model.cpu()
    X, Z, Y, _, _, _, X_test, Z_test, Y_test, _, _ = load_data(dataset_name)
    with torch.no_grad():
        X_erased = model(torch.tensor(X)).numpy()
        X_test_erased = model(torch.tensor(X_test)).numpy()

    import warnings
    with warnings.catch_warnings(action="ignore"):
        if dataset_name == 'biasbios':
            from eval_biasbios import eval_biasbios
            eval_biasbios(X_erased, Z, Y, X_test_erased, Z_test, Y_test, args["out_name"], output_path)
        elif dataset_name == 'GloVe':
            from eval_GloVe import eval_GloVe
            eval_GloVe(X_erased, X, Z, X_test_erased, Z_test)
            r_spearman, r_pearson = ws353_similarity_test(data_path, model)
            print(f"WS353 similarity test: \nSpearman r: {r_spearman.statistic:.2f} - Pearson r: {r_pearson.statistic:.2f}")
        elif dataset_name == 'deepmoji':
            from eval_deepmoji import eval_deepmoji
            eval_deepmoji(X_erased, Z, Y, X_test_erased, Z_test, Y_test, args["out_name"], output_path)

