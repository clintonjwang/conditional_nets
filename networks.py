from torch import relu
import torch
from torch.nn.parameter import Parameter
import torch.distributions.beta
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms
nn = torch.nn
dtype = torch.float

class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=.2)
        self.fc1 = nn.Linear(320, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 28, 28)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.softmax(self.fc2(x), dim=1)

    
class NaiveCNN(nn.Module):
    def __init__(self, loss_type='mse'):
        super(NaiveCNN, self).__init__()        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=.2)
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(129, 64)
        self.fc4 = nn.Linear(64, 1 if loss_type == 'mse' else 7)
        self.bn1 = nn.BatchNorm1d(128)
        self.loss_type = loss_type

    def forward(self, x, u):      
        u = u.view(x.shape[0], 1)  
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        ux = F.relu(self.fc1(x))
        ux = self.bn1(ux)
        ux = F.relu(self.fc2(ux))
        ux = F.dropout(ux, training=self.training)
        ux = torch.cat([ux, u], 1)
        ux = F.relu(self.fc3(ux))
        p_Y_UX = self.fc4(ux)
        if self.loss_type == 'mse':
            p_Y_UX = p_Y_UX.view(-1)
        
        return p_Y_UX
        
    def classify(self, x, u):
        x = x.view(x.shape[0], 1, 28, 28)
        return self.forward(x,u)


class MixModelCNN(nn.Module):
    def __init__(self, K):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MixModelCNN, self).__init__()
        # we want to prevent the variance of the predicted beta distributions from contracting to a singularity
        # thus we can set a prior in which the inverse standard deviation follows a half-normal distribution
        sig = 500
        self.l_reg = 1/(2*sig**2)

        # log normal prior with m+s=ln(100), m-s=ln(.01)
        #mu = (np.log(100)-np.log(.1))/2
        #sig = np.log(100)-mu
        #l_reg = N/2/sig**2
        
        #torch.log(torch.linspace(.1,6,K, dtype=dtype, requires_grad=True)))
        #self.a_u = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        #self.b_u = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        self.sigmu_u = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        self.lnnu_u = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        self.sigmu_y = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        self.lnnu_y = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        #self.a_y = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        #self.b_y = Parameter(torch.randn(1,K, dtype=dtype, requires_grad=True))
        #self.p_z = Parameter(torch.tensor(torch.log(torch.ones(1,K)/K), dtype=dtype, requires_grad=True))
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=.2)
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, K)
        self.bn1 = nn.BatchNorm1d(128)
        self.K = K

    def forward(self, x, uy):
        """
        xyu must be 3*N
        """
        #p_z = self.get_p_z()
        
        mu_u = 1/(1+torch.exp(-self.sigmu_u))
        nu_u = torch.exp(self.lnnu_u)
        self.a_u = torch.log(mu_u*nu_u)
        self.b_u = torch.log((1-mu_u)*nu_u)
        
        mu_y = 1/(1+torch.exp(-self.sigmu_y))
        nu_y = torch.exp(self.lnnu_y)
        self.a_y = torch.log(mu_y*nu_y)
        self.b_y = torch.log((1-mu_y)*nu_y)
        
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        p_Z_X = F.softmax(self.fc3(x), dim=1)
        #p_X_Z = p_Z_X / p_z # p_x is a constant and hence disappears in the objective function
        
        U_z = torch.distributions.beta.Beta(torch.exp(self.a_u), torch.exp(self.b_u))
        Y_z = torch.distributions.beta.Beta(torch.exp(self.a_y), torch.exp(self.b_y))
        f_YU_z = torch.exp(U_z.log_prob(uy[:,:1].float()) + Y_z.log_prob(uy[:,-1:].float()))
        
        #p_ZYU = f_YU_z * p_z
        p_XYU = (p_Z_X * f_YU_z).sum(1)# (p_X_Z * p_ZYU).sum(1)
        neg_log_ll = -torch.log(p_XYU).sum()
        
        # beta dist inverse stdev half-normal prior (precision has stretched chi-square dist with k=1)
        var_U = mu_u * (1-mu_u) / (1+nu_u)
        var_Y = mu_y * (1-mu_y) / (1+nu_y)
        """var_U = torch.exp(self.a_u) * torch.exp(self.b_u) / (
                torch.exp(self.a_u) + torch.exp(self.b_u) )**2 / (
                torch.exp(self.a_u) + torch.exp(self.b_u) + 1)
        var_Y = torch.exp(self.a_y) * torch.exp(self.b_y) / (
                torch.exp(self.a_y) + torch.exp(self.b_y) )**2 / (
                torch.exp(self.a_y) + torch.exp(self.b_y) + 1)"""
        
        ab_reg = ((1/var_U).sum() + (1/var_Y).sum()) * self.l_reg * x.shape[0]
        
        return neg_log_ll + ab_reg
        
    #def get_p_z(self):
    #    return torch.exp(self.p_z) / torch.exp(self.p_z).sum()
    
    def classify(self, x, u, num_classes=7):
        #p_z = self.get_p_z()
        y = torch.linspace(.001, .999, num_classes).repeat(x.shape[0],1).view(x.shape[0], -1, 1).cuda()
        u = u.view(-1, 1)

        mu_u = 1/(1+torch.exp(-self.sigmu_u))
        nu_u = torch.exp(self.lnnu_u)
        self.a_u = torch.log(mu_u*nu_u)
        self.b_u = torch.log((1-mu_u)*nu_u)
        
        mu_y = 1/(1+torch.exp(-self.sigmu_y))
        nu_y = torch.exp(self.lnnu_y)
        self.a_y = torch.log(mu_y*nu_y)
        self.b_y = torch.log((1-mu_y)*nu_y)
        
        x = torch.reshape(x, (x.shape[0], 1, *x.shape[1:]))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        p_Z_X = F.softmax(self.fc3(x), dim=1).view(x.shape[0], 1, self.K)

        U_z = torch.distributions.beta.Beta(torch.exp(self.a_u), torch.exp(self.b_u))
        Y_z = torch.distributions.beta.Beta(torch.exp(self.a_y), torch.exp(self.b_y))
        f_U_Z = torch.exp(U_z.log_prob(u)).view(x.shape[0], 1, self.K)
        f_Y_Z = torch.exp(Y_z.log_prob(y))

        p_Z_UX = f_U_Z * p_Z_X # this should be divided by f_U_X, but that is a constant since X and U are given
        f_Y_UX = (f_Y_Z * p_Z_UX).sum(2)
        p_Y_UX = f_Y_UX / f_Y_UX.sum(1, keepdim=True)

        return p_Y_UX#.max(1)


class MixModel(torch.nn.Module):
    """Only the mixture model, without images. Outdated."""
    def __init__(self, K):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MixModel, self).__init__()
        self.a_u = Parameter(torch.tensor(torch.randn(1,K), dtype=dtype, requires_grad=True))
        self.b_u = Parameter(torch.tensor(torch.randn(1,K), dtype=dtype, requires_grad=True))
        self.a_y = Parameter(torch.tensor(torch.randn(1,K), dtype=dtype, requires_grad=True))
        self.b_y = Parameter(torch.tensor(torch.randn(1,K), dtype=dtype, requires_grad=True))
        self.p_z = Parameter(torch.tensor(torch.ones(1,K)/K, dtype=dtype, requires_grad=True))

    def forward(self, yu):
        """
        yu must be 2*N
        """
        U = torch.distributions.beta.Beta(torch.exp(self.a_u), torch.exp(self.b_u))
        Y = torch.distributions.beta.Beta(torch.exp(self.a_y), torch.exp(self.b_y))
        f_YU = torch.exp(Y.log_prob(yu[:,:1]) + U.log_prob(yu[:,-1:]))
        neg_log_ll = -torch.log((self.p_z*f_YU).sum(1)).sum()
        
        # log normal prior over a,b
        #ab_reg = ((torch.cat([self.a_u, self.a_y, self.b_u, self.b_y]) - mu)**2).sum() * l_reg
        
        # beta dist inverse stdev half-normal prior (precision has stretched chi-square dist with k=1)
        var_beta_u = torch.exp(self.a_u) * torch.exp(self.b_u) / (
                torch.exp(self.a_u) + torch.exp(self.b_u) )**2 / (
                torch.exp(self.a_u) + torch.exp(self.b_u) + 1)
        
        var_beta_y = torch.exp(self.a_y) * torch.exp(self.b_y) / (
                torch.exp(self.a_y) + torch.exp(self.b_y) )**2 / (
                torch.exp(self.a_y) + torch.exp(self.b_y) + 1)
        
        ab_reg = ((1/var_beta_u).sum() + (1/var_beta_y).sum()) * l_reg
        
        return neg_log_ll + ab_reg

    
class ProbClipper(object):
    """Unused."""
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        #if hasattr(module, 'p_z'):
        p_z = module.p_z.data
        p_z.sub_(torch.relu(torch.min(p_z)))
        p_z.div_(torch.sum(p_z))
        #else:
            #raise ValueError('p_z parameter missing')
            
            
def cutoff_loss(x, a, b):
    """Unused."""
    eps = .01
    x = x.cuda()
    a = torch.tensor([a], dtype=torch.double).cuda()
    b = torch.tensor([b], dtype=torch.double).cuda()
    return (relu(x-b)/torch.abs(b+eps) + relu(a-x)/torch.abs(a+eps))