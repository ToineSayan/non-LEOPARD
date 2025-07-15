import torch
import math



class RDF_loss(torch.nn.Module):
    """
    A generic class that defines RDF computation for use in various losses.
    """
    def __init__(self, eps_squared=0.5, sphere_radius=1.0):
        """
        eps (float): epsilon in RDF
        n_features (int): number of features
        sphere_radius (float): radius of the sphere on which representations are projected before RDF calculation
        """
        super().__init__()
        self.eps = math.sqrt(eps_squared)
        # self.scale_fn = torch.nn.LayerNorm(normalized_shape=n_features, elementwise_affine=False) # project on the sphere of radius sqrt(n_features)
        # self.n_features = n_features
        self.sphere_radius = sphere_radius
        # self.scale_coef = sphere_radius/math.sqrt(n_features)

        self.loss_infos = {}

    def print_current_loss(self, precision = 2):
        s = ', '.join([f"{k}: {v:.{precision}f}" for k,v in self.loss_infos.items()])
        return s

    def scale(self, X, rank=None):
        n_features = X.shape[1]
        
        scale_fn = torch.nn.LayerNorm(normalized_shape=n_features, elementwise_affine=False)
        scale_coef = self.sphere_radius/math.sqrt(n_features)
        
        return scale_fn(X)*scale_coef # on the unit sphere

    def rate(self, X, n_total=-1):
        """
        Empirical Discriminative Loss.
        """
        device = X.device
        n, d = X.shape
        # In some decompositions of RDF calculation, the value associated with the 
        # number of observations n may differ from the number of observations in X
        if n_total > 0:
            n = n_total
        I = torch.eye(d).to(device)
        scalar = d / (n * (self.eps ** 2))
        logdet = torch.logdet(I + scalar * (X.T@X))
        # logdet = torch.logdet(scalar * (X.T@X))
        return 0.5*logdet / math.log(2)
        

    def kernelized_rate(self, X, Z = None, kernel=None):
        device = X.device
        n, d = X.shape


        if Z is None:
            K = torch.ones((n,n)).to(device)
        else:
            K = torch.zeros((n,n)).to(device)
            for z_val in list(torch.unique(Z)):
                mask = Z == z_val  # Masque booléen pour Z
                condition = mask.unsqueeze(0) & mask.unsqueeze(1)

                if kernel is None:
                    K[condition] = 1.0
                elif kernel == 'rbf':
                    diff = X.unsqueeze(1) - X.unsqueeze(0)
                    norm = -0.5*torch.norm(diff, dim=2)**2
                    K[condition] = torch.exp(norm[condition])


        I = torch.eye(n).to(device)
        scalar = d / (n * (self.eps ** 2))
        logdet = (0.5 / math.log(2)) * torch.logdet(I + scalar * (X@X.T) * K)
        return logdet

    def forward(self, X, Z=None, n_total=-1):
        X = self.scale(X)
        if Z is not None:
            loss_value = sum([self.rate(X[Z==z], n_total=n_total) for z in torch.unique(Z)])
            # loss_value = self.kernelized_rate(X, Z)
        else:
            loss_value = self.rate(X, n_total=n_total)
            # loss_value = self.kernelized_rate(X)
        self.loss_infos = {"loss":loss_value.item()}
        return loss_value

        


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------




class KRaM_Loss_categorical(RDF_loss):
    """
    Implementation of the KRaM loss for a categorical concept defined in the article 
    'Robust Concept Erasure via Kernelized Rate-Distortion Maximization' (eq. 4)

    R(f(X)|K) - lambda*|R(f(X)) - alpha*R(X)|

    Note: coef. alpha is not in the article
    """
    def __init__(self, lambda_=0.7):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, fX, X, Z, *args):
        # Scale the inputs before RDF calculation

        fX = self.scale(fX)
        X = self.scale(X)

        lRDF = self.kernelized_rate(fX,Z) 
        if self.lambda_ > 0.0:
            b = self.kernelized_rate(X) # R(X)
            RZ = self.kernelized_rate(fX) # R(f(X))
            control = abs(RZ-b)  
        else:
            control = torch.tensor(0.0)

        loss_full = lRDF - self.lambda_*control 
        self.loss_infos = {"loss":loss_full.item(), "R(Z|K)": lRDF.item(), "ctrl": control.item()}
        return -loss_full # minus to maximize the loss during training




class FaRM_Loss_categorical(RDF_loss):
    """
    Implementation of the FaRM loss (unconstrained) for a categorical concept defined in the article 
    'Learning Fair Representations via Rate-Distortion Maximization' (eq. 3)

    R(f(X)|Pi^g) + R(f(X))
    """
    def __init__(self):
        super().__init__()

    def forward(self, fX, X, Z, **args):
        # Scale the inputs before RDF calculation
        fX = self.scale(fX)
        
        n = X.size(dim=0)
        lRDF = sum([len(Z[Z==z])*self.rate(fX[Z==z]) for z in torch.unique(Z)])/n # R(f(X)|Pi^g)
        RZ = self.rate(fX) # R(f(X))
        loss_full = lRDF + RZ
        self.loss_infos = {"loss":loss_full.item(), "R(Z|K)": lRDF.item(), "R(Z)": RZ.item()}
        return -loss_full # minus to maximize the loss during training


#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------


# MMD Loss

class RBF(torch.nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth="mean"): # remettre None pr bandwidth
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth
        # self.bandwidth_heuristic = bandwidth_heuristic

    def get_bandwidth(self, L2_distances):
        if self.bandwidth == "median":
            return L2_distances[L2_distances > 0].median()
        elif self.bandwidth == "mean":
            # print("here")
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        else:
            return self.bandwidth

    def forward(self, X):
        self.bandwidth_multipliers = self.bandwidth_multipliers.to(X.device)
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)
    


class MMDLoss(torch.nn.Module):

    def __init__(self, kernel_type='RBF', rbf_n_kernels = 5, rbf_kernel_bandwidth = "mean"):
        super().__init__()
        if kernel_type == "RBF":
            self.rbf_kernel_bandwidth = rbf_kernel_bandwidth
            self.rbf_n_kernels = rbf_n_kernels
            self.kernel = RBF(n_kernels = rbf_n_kernels, bandwidth = rbf_kernel_bandwidth)


    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        n, m = X_size, K.shape[0] - X_size

        # XX = K[:X_size, :X_size].mean()
        XX = K[:X_size, :X_size].sum()/(n*(n-1))
        XY = K[:X_size, X_size:].mean()
        # YY = K[X_size:, X_size:].mean()
        YY = K[X_size:, X_size:].sum()/(m*(m-1))
        return torch.sqrt(XX - 2 * XY + YY)
    

class MMD_Loss_categorical(torch.nn.Module):
    """
    MMD LOSS
    """
    def __init__(self, rbf_n_kernels = 5, rbf_kernel_bandwidth = "mean"):
        super().__init__()
        self.loss_infos = {}
        self.MMD = MMDLoss(rbf_n_kernels = rbf_n_kernels, rbf_kernel_bandwidth = rbf_kernel_bandwidth)   

    def forward(self, fX, X, Z, *args):
        z_values = torch.unique(Z)
        couples = []
        for i in range(len(z_values)):
            for j in range(i + 1, len(z_values)):
                couples.append((z_values[i], z_values[j]))


        # loss_mmd = sum([self.MMD(fX[Z == z1], fX[Z == z2]) for z1,z2 in couples])
        loss_mmd = sum([self.MMD(fX[Z == z1], fX[Z == z2])**2 for z1,z2 in couples])

        # loss_mmd = max([self.MMD(fX[Z == z1], fX[Z == z2]) for z1,z2 in couples]) * len(couples)
        loss_full = loss_mmd

        self.loss_infos = {"loss":loss_full.item(), "MMD": loss_mmd.item()}
        return loss_full
    

def init_loss(loss_name, **kwargs):

    if loss_name == 'FaRM':
        criterion = FaRM_Loss_categorical()
    elif loss_name == 'KRaM':
        criterion = KRaM_Loss_categorical()
    elif loss_name == 'MMD':
        criterion = MMD_Loss_categorical(
            rbf_n_kernels = kwargs.get('rbf_n_kernels', 5), 
            rbf_kernel_bandwidth = kwargs.get('rbf_kernel_bandwidth', "mean"),
            # bandwidth_heuristic = kwargs.get('bandwidth_heuristic', "median")
            )
    else:
        criterion = None
    return criterion