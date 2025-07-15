
import numpy as np
import torch
import random
from eval import *

def eval_GloVe(X_erased, X, Z, X_test_erased, Z_test):


    ##############################################################
    # Model evaluation (linear+non linear erasure and Ak)
    ##############################################################

    results = {}

    # control of randomness
    seed = 1337 # new seed for evaluation
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    results['A50'] = f'{Ak(torch.tensor(X), torch.tensor(X_erased), 0.5):.2f}'
    print(f"Ak (k=0.50):", 'train',  results['A50'])
    results['A10'] = f'{Ak(torch.tensor(X), torch.tensor(X_erased), 0.1):.2f}'
    print(f"Ak (k=0.10):", 'train',  results['A10'])
    results['A01'] = f'{Ak(torch.tensor(X), torch.tensor(X_erased), 0.01):.2f}'
    print(f"Ak (k=0.01):", 'train',  results['A01'])


    print("Rank h(X):",  np.linalg.matrix_rank(X_erased, tol=1.0e-3))
    print("Rank h(X_test):",  np.linalg.matrix_rank(X_test_erased, tol=1.0e-3))


    # MDL 
    mdl = MDL(X_erased[Z != 2], Z[Z != 2], max_iter=1000)
    print("[train] MDL:", f'{mdl.get_score():.2f}', "kBits")


    score_train, score_test, std_train, std_test, _ = eval_clf(
            X_erased[Z != 2], Z[Z !=2], X_test_erased[Z_test != 2], Z_test[Z_test != 2],
            n_eval = 5, 
            save = False
    )
    print("h(X) binary - Probing after erasure (MLP classifier):", f'train: {score_train*100:.2f} ({std_train*100:.2f}), test: {score_test*100:.2f} ({std_test*100:.2f})')
    results['acc_non_lin'] = f'{score_test*100:.2f} ({std_test*100:.2f})'


    score_train, score_test, std_train, std_test, _ = eval_clf(
        X_erased, Z, X_test_erased, Z_test,
        n_eval = 5, 
        save = False
        )
    print("h(X) ternary - Probing after erasure (MLP classifier):", f'train: {score_train*100:.2f} ({std_train*100:.2f}), test: {score_test*100:.2f} ({std_test*100:.2f})')
    results['acc_non_lin_ter'] = f'{score_test*100:.2f} ({std_test*100:.2f})'

        
    print(results)



if __name__ == "__main__":
    from load_data import load_data

    X, Z, Y, _, _, _, X_test, Z_test, Y_test, _, _ = load_data("GloVe")



    import warnings
    with warnings.catch_warnings(action="ignore"):
        eval_GloVe(X, X, Z, X_test, Z_test)
