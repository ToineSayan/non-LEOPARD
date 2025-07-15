import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
from random import shuffle
from tqdm import tqdm



from statistics import mean, stdev

import pickle




##############################################################################
# Alignment 
##############################################################################


def Ak(X, fX, k):
    num_neighbors =  math.floor(k*X.size(dim=0))

    # Let's calculate the neighbors of original observations
    D = torch.cdist(X,X,p=2)
    knn_frontiers = torch.quantile(D, q=k, dim=1)[:,None]
    Nx = (D<=knn_frontiers)

    # Let's calculate the neighbors of transformed observations
    D = torch.cdist(fX,fX,p=2)
    knn_frontiers = torch.quantile(D, q=k, dim=1)[:,None]
    NfX = (D<=knn_frontiers)

    return torch.mean(torch.sum(Nx * NfX, dim=1).float()).item()/num_neighbors



##############################################################################
# Probing
##############################################################################


def eval_clf(X_train, Z_train, X_test, Z_test, n_eval=1, save=False, output_path_and_name=None):
    from sklearn.neural_network import MLPClassifier
    scores_train, scores_test = [], []
    clfs = []
    for i in range(n_eval):
        print("Training:", i+1, '/', n_eval)
        clf = MLPClassifier(max_iter=20).fit(X_train, Z_train)
        scores_train.append(clf.score(X_train, Z_train))
        scores_test.append(clf.score(X_test, Z_test))
        if save:
            o = output_path_and_name + f"_{i}.pt"
            with open(o,'wb') as f: pickle.dump(clf,f)
        clfs.append(clf)

    return (mean(scores_train), mean(scores_test), stdev(scores_train), stdev(scores_test), clfs) if n_eval > 1 else (scores_train[0], scores_test[0], 0.0, 0.0, clfs)



##############################################################################
# MDL
##############################################################################


class MDL:
    """
    Minimum Description Length (MDL)

    Computes the online code for MDL, implementing details provided
    in the paper -- https://arxiv.org/pdf/2003.12298.pdf
    cf. equation (3)

    Arguments:
        dataset: list of tuples [(x, y)], where x is a feature vector and y is the label
    """
    def __init__(self, X, Y, max_iter=1000):
        super(MDL, self).__init__()
        n = len(X)
        indices = np.arange(n)
        shuffle(indices)

        ratios = [
            0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.125, 0.25, 0.5,
            1
        ]
        self.block_indices = []
        for r in ratios:
             self.block_indices.append(indices[:int(r * n)])
        self.num_labels = len(np.unique(Y))
        self.X = X
        self.Y = Y

        self.max_iter = max_iter


    def get_score(self):
        from sklearn.neural_network import MLPClassifier
        # bits required for the first transmission
        score = len(self.block_indices[0]) * math.log(self.num_labels, 2)


        for i, indices in enumerate(self.block_indices[:-1]):
            print("Block:", i+1, '/', len(self.block_indices)-1)
            X_train = self.X[indices]
            Y_train = self.Y[indices]

            clf = MLPClassifier(max_iter=self.max_iter)
            clf.fit(X_train, Y_train)

            next_indices = self.block_indices[i + 1]
            X_test = self.X[next_indices]
            Y_test = self.Y[next_indices]

            Y_pred = clf.predict_proba(X_test)

            for y_gold, y_pred in zip(Y_test, Y_pred):
                try:
                    score -= math.log(y_pred[y_gold], 2)
                except:
                    pass

        return (score / 1024)  # Final output in Kbits






##############################################################################
# Relative error
##############################################################################


def relativeError(X, Y, output_torch = False):
    err = torch.mean(torch.norm(X - Y, dim=1)**2 / torch.norm(X, dim=1)**2)
    return err.item() if not output_torch else err



##############################################################################
# WS353
##############################################################################


def ws353_similarity_test(path_to_eval_file, model = None):
    import pickle
    from scipy.stats import spearmanr, pearsonr

    def read_ws353():
        ws353 = []
        with open(path_to_eval_file + '/evaluation' + '/ws353simrel/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt', "r") as f:
            for line in f:
                if line[0] != '#':
                    word_1, word_2, similarity_score = line.strip().split('\t')
                    ws353.append((None, word_1, word_2, similarity_score))
        return ws353
    
    with open(path_to_eval_file + '/glove-top-150k.pickle', "rb") as f:
        data = pickle.load(f)
    glove_150k = dict(zip(data['words'], data['vecs']))
    ws353 = read_ws353()

    X1, X2 = [], []
    Y_sim = []
    for _, w1, w2, sim in ws353:
        try:
            v1, v2 = glove_150k[w1], glove_150k[w2]
            X1.append(v1)
            X2.append(v2)
            Y_sim.append(float(sim))
        except:
            None
    X1, X2, Y_sim = torch.tensor(np.array(X1)), torch.tensor(np.array(X2)), torch.tensor(Y_sim)

    if not model == None:
        X1, X2 = model(X1), model(X2)

    
    cosine = torch.sum(X1 * X2, dim=1)/(torch.norm(X1, dim=1)*torch.norm(X2, dim=1))

    cosine, Y_sim = cosine.detach().numpy(), Y_sim.detach().numpy()
    return spearmanr(cosine, Y_sim), pearsonr(cosine, Y_sim)



##############################################################################
# Fairness evaluations
##############################################################################


def demographic_parity(X, Y, Z, clfs):
        
    def dp_single(y_hat, y, z):
        return  sum([abs(np.mean(y_hat[z == 0] == i) - np.mean(y_hat[z == 1] == i)) for i in np.unique(y)])

    dps = [dp_single(clf.predict(X), Y, Z) for clf in clfs]
    return np.mean(dps), np.std(dps)



def TPR_Gaps(y, Z, Y, ref_z_val, compared_z_val):
    """ 
    Calculate the TPR-Gaps for a z, compared to the counterfactual value z'
    for each Y-value   
    """
    filter_z_val = lambda z_val: Z == z_val # filter on Z values
    filter_y_val = lambda y_val: Y == y_val # filter on True Y predictions

    TPR_Gaps = dict()
    for y_val in np.unique(Y):
        TPR_ref = np.mean(
                    y[filter_z_val(ref_z_val)][filter_y_val(y_val)[filter_z_val(ref_z_val)]] == y_val
                )
        TPR_compared = np.mean(
                    y[filter_z_val(compared_z_val)][filter_y_val(y_val)[filter_z_val(compared_z_val)]] == y_val
                )
        TPR_Gaps[y_val] = TPR_ref - TPR_compared
    return TPR_Gaps

def Z_proportions(z_val, Z, Y): # gender imbalance
    filter_y_val = lambda y_val: Y == y_val # filter on True Y predictions
    Z_prop = dict()
    for y_val in np.unique(Y):
        Z_prop[y_val] = np.mean(Z[filter_y_val(y_val)] == z_val)
    return Z_prop


def TPR_RMS(X, Y, Z, z1_id, z2_id, clfs):
    list_TPR_gaps = [TPR_Gaps(clf.predict(X), Z, Y, z1_id, z2_id) for clf in clfs]
    RMS_dict = lambda D : np.sqrt(np.mean([D[y_val]**2 for y_val in np.unique(Y)]))
    list_RMS = [RMS_dict(D) for D in list_TPR_gaps]

    return np.mean(list_RMS), np.std(list_RMS)

def TPR_corr_coef(X, Y, Z, z1_id, z2_id, clfs):
    list_TPR_gaps = [TPR_Gaps(clf.predict(X), Z, Y, z1_id, z2_id) for clf in clfs]
    Z_props = Z_proportions(z1_id, Z, Y)
    Z_props_list = [Z_props[y_val] for y_val in np.unique(Y)]
    Corr_coef = lambda D : np.corrcoef(Z_props_list, [D[y_val] for y_val in np.unique(Y)])[0,1]
    list_corr_coefs = [Corr_coef(D) for D in list_TPR_gaps] 
    return np.mean(list_corr_coefs), np.std(list_corr_coefs)


