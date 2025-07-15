import numpy as np
import random
from eval import eval_clf, MDL,demographic_parity, TPR_RMS, TPR_corr_coef


def eval_deepmoji(X_erased, Z, Y, X_test_erased, Z_test, Y_test, name, output_path):

        results = {}

        # control of randomness
        seed = 1337 # new seed for evaluation
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        print("Rank h(X):",  np.linalg.matrix_rank(X_erased, tol=1.0e-3))
        print("Rank h(X_test):",  np.linalg.matrix_rank(X_test_erased, tol=1.0e-3))

        n_eval = 5

        # Probing
        score_train, score_test, std_train, std_test, _ = eval_clf(
            X_erased, Z, X_test_erased, Z_test,
            n_eval = n_eval, 
            save = False
            )
        print("f(X) - Probing after erasure (MLP classifier):", f'train: {score_train*100:.2f} ({std_train*100:.2f}), test: {score_test*100:.2f} ({std_test*100:.2f})')
        results['acc_non_lin'] = f'{score_test*100:.2f} ({std_test*100:.2f})'

        # Sentiment prediction
        score_train, score_test, std_train, std_test, clfs = eval_clf(
            X_erased, Y, X_test_erased, Y_test,
            n_eval = n_eval, 
            save = True, 
            output_path_and_name = output_path + f'/clf_dial_y_{name}'
            )
        print("Accuracy of predicting Y after erasure:", f'train: {score_train*100:.2f} ({std_train*100:.2f}), test: {score_test*100:.2f} ({std_test*100:.2f})')
        results["acc_y"] = f'{score_test*100:.2f} ({std_test*100:.2f})'
        print(results)



        # MDL 
        mdl = MDL(X_erased, Z, max_iter=20)
        print("[train] MDL:", f'{mdl.get_score():.2f}', "kBits")

        # Fairness evaluation
        dp_mean, dp_std = demographic_parity(X_test_erased, Y, Z_test, clfs)
        print("DP:", dp_mean, "std:", dp_std)

        mean_rms, std_rms = TPR_RMS(X_test_erased, Y_test, Z_test, 0, 1, clfs)
        print("RMS TPR-Gaps (nogender):", mean_rms, std_rms)

        mean_rms, std_rms = TPR_corr_coef(X_test_erased, Y_test, Z_test, 0, 1, clfs)
        print('Correlation coef. (nogender):', mean_rms, std_rms)   


if __name__=="__main__":
    
    from load_data import load_data

    X, Z, Y, _, _, _, X_test, Z_test, Y_test, _, _ = load_data("deepmoji")


    import warnings
    with warnings.catch_warnings(action="ignore"):
        eval_deepmoji(X, Z, Y, X_test, Z_test, Y_test, "orig_final", "./outputs")

