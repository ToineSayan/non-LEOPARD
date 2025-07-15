import torch.nn as nn
import torch
import numpy as np





#############################################################
# Generic class
#############################################################


class MLP(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, input_size, output_size, hidden_sizes = [], dropout=0.1):
        super().__init__()

        print("MLP", hidden_sizes, input_size, dropout)

        self.layers = nn.ModuleList()
        n_layers = len(hidden_sizes)+1
        in_to_out = [input_size] + hidden_sizes + [output_size]
        for i in range(n_layers):
            self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(in_to_out[i], in_to_out[i+1]))
            if not (i+1 == n_layers):
                self.layers.append(nn.ReLU())

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x 


    

#############################################################
# Erasers
#############################################################




class LinearEraser(nn.Module):
    """
    Erasure model that erases a concept 'linearly', i.e. in such a way that a linear predictor trained to predict 
    the concept of interest is unable to predict it better than a constant predictor (cf. the notion of linear guardedness). 
    Erasure is achieved by means of an orthogonal projection that removes (c-2) dimensions, where c is the total number of 
    values taken by the concept to be removed. The orthogonal projector used is derived from the oblique projector 
    calculated by the LEACE method.

    """
    def __init__(self, X, Z):
        super().__init__()
        # Calculate the base transformation
        from concept_erasure import LeaceEraser
        Z_1hot = torch.nn.functional.one_hot(Z)
        P_leace = LeaceEraser.fit(X, Z_1hot).P # oblique projection 
        _,_,Vh = torch.linalg.svd(P_leace)
        Ep = Vh[:-Z_1hot.shape[1]+1,:].T 
        P = Ep @ Ep.T # orthogonal projection
        
        self.register_buffer('Ep', Ep) # Tensor which is not a parameter, but should be part of the modules state.
        self.register_buffer('P', P) # Tensor which is not a parameter, but should be part of the modules state.

    def to_original_basis(self, x):
        return x @ self.Ep.T

    def forward(self, x, reduction = False, *args, **kwargs):
        x = x @ self.P.T if not reduction else x @ self.Ep # (n, input_size) -> (n, input_size - c)
        return x
    


        
#############################################################
# Projections
#############################################################



class RankProjection(nn.Module):
    """
    Dynamic MLP network with ReLU non-linearity
    """
    def __init__(self, input_size, rank, dropout=0.0):
        super().__init__()
        self.rank = rank # rank of the projector (number of dimensions kept)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.U = nn.Parameter(torch.eye(input_size, rank, requires_grad=True)) 
        # self.b = nn.Parameter(torch.zeros((input_size,), requires_grad=True))
        
        # to be updated after each training phase
        self.register_buffer('P', torch.empty(input_size, input_size))
        self.register_buffer('Ep', torch.empty(input_size, rank)) 
        self.projected = False

    
    def to_projector(self):
        _, U = torch.linalg.eigh(self.U @ self.U.T) # use eigh because UU.T is symmetrical / eigenvalues are sorted in ascending order
        U = U[:,-self.rank:]
        self.P.copy_(U @ U.T)
        self.Ep.copy_(U)
    
    def projection_properties_eval(self, X = None, p=2):
        return (torch.norm(self.U.T @ self.U - torch.eye(self.rank).to(self.U.device), p=p)**p)/(self.rank**2)
    
    def get_projector(self, approximation=False):
        if approximation:
            return self.U @ self.U.T
        else:
            if not self.projected:
                self.to_projector()
                self.projected = True
            return self.P

        
    def forward(self, x, reduction = True): 
        x = self.dropout_layer(x) 
        if self.training:
            self.projected = False # to project after each training phase
            x = x @ self.U @ self.U.T if not reduction else x @ self.U
        else:
            if not self.projected:
                self.to_projector()
                self.projected = True
            x = x @ self.P.T if not reduction else x @ self.Ep
        return x
    




class CascadeProjection(nn.Module):

    def __init__(self, X, Z, rank = None, dropout=0.0):
        super().__init__()
        input_size = X.shape[1]
        self.rank = rank
        self.projected = False
        self.linear_eraser = LinearEraser(X, Z)
        subspace_dimension = self.linear_eraser.Ep.shape[1]
        self.nonlinear_eraser = RankProjection(
            input_size = subspace_dimension,
            rank = rank,
            dropout = dropout
        )

        self.register_buffer('P',  torch.empty(input_size, input_size)) # Tensor which is not a parameter, but should be part of the modules state.
        if rank is not None:
            self.register_buffer('Ep', torch.empty(input_size, rank)) 
        else:
            self.register_buffer('Ep', torch.empty(input_size, input_size)) 


    def projection_properties_eval(self, X = None):
        return self.nonlinear_eraser.projection_properties_eval(self.linear_eraser(X, reduction=True)) if X is not None else self.nonlinear_eraser.projection_properties_eval()

    def forward(self, x, *args, **kwargs):
        x = self.linear_eraser(x, reduction=True) # (n, input_size) -> (n, input_size - c)
        x = self.nonlinear_eraser(x, reduction = False) # erasure in the orthogonal subspace
        x = self.linear_eraser.to_original_basis(x) # (n, input_size - c) -> (n, input_size) 
        return x

    def to_projector(self):
        self.nonlinear_eraser.to_projector() # here Ep and P have been calculated 
        self.Ep = self.linear_eraser.Ep @ self.nonlinear_eraser.Ep
        self.P.copy_(self.linear_eraser.Ep @ self.nonlinear_eraser.P @ self.linear_eraser.Ep.T) # note : this is a projection


    def get_projector(self, approximation=False):
        P_nlin = self.nonlinear_eraser.get_projector(approximation)
        return self.linear_eraser.Ep @ P_nlin @ self.linear_eraser.Ep.T





#############################################################
# Cascade MLP
#############################################################





class CascadeMLP(nn.Module):

    def __init__(self, X, Z, hidden_sizes = [], dropout = 0.1):
        super().__init__()
        self.linear_eraser = LinearEraser(X, Z)
        subspace_dimension = self.linear_eraser.Ep.shape[1]        


        print("cascade", hidden_sizes, subspace_dimension, dropout)

        self.nonlinear_eraser = MLP(
            input_size = subspace_dimension, 
            output_size = subspace_dimension, 
            hidden_sizes = hidden_sizes, 
            dropout = dropout
        )

    def forward(self, x, *args, **kwargs):
        x = self.linear_eraser(x, reduction=True) # (n, input_size) -> (n, input_size - c)
        x = self.nonlinear_eraser(x) # erasure in the orthogonal subspace
        x = self.linear_eraser.to_original_basis(x) # (n, input_size - c) -> (n, input_size) 
        return x














#############################################################
# Load functions
#############################################################


def init_model(eraser_model, X, Z = None, **kwargs):

    input_size = X.size(dim=1)
    output_size = input_size

    # Initialize the model
    if eraser_model == 'linear':
        model = LinearEraser(X, Z) # LEACE orthogonal
    elif eraser_model == 'MLP':
        model = MLP(input_size=input_size, output_size=output_size, hidden_sizes=kwargs.get('hidden_sizes'), dropout=kwargs.get('dropout', 0.0)) 
    # elif eraser_model == 'cascade_MLP':
    #     model = CascadeMLP(input_size, output_size, hidden_sizes, eraser_dropout) 
    elif eraser_model == 'projection':
        model = RankProjection(input_size=input_size, rank = kwargs.get('projection_rank'))
    elif eraser_model == 'cascade_projection':
        model = CascadeProjection(X, Z,  rank = kwargs.get('projection_rank'))
    elif eraser_model == 'cascade_MLP':
        model = CascadeMLP(X, Z, hidden_sizes=kwargs.get('hidden_sizes'), dropout=kwargs.get('dropout', 0.0))
    return model