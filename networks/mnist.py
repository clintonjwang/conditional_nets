import torch
import torch.nn.functional as F
import niftiutils.nn.submodules as subm
nn = torch.nn

class BaseCNN(nn.Module):
    dims=(1, 28, 28)
    def __init__(self):
        super(BaseCNN, self).__init__()        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop = nn.Dropout2d(p=.2)
        self.fc1 = nn.Linear(320, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0], *self.dims)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.softmax(self.fc2(x), dim=1)

    
class NaiveCNN(BaseCNN):
    def __init__(self, loss_type='mse', num_vars=1):
        super(NaiveCNN, self).__init__()
        self.fc2 = nn.Linear(128+num_vars, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1 if loss_type == 'mse' else 7)
        self.bn1 = nn.BatchNorm1d(128)
        self.loss_type = loss_type
        self.num_vars = num_vars

    def forward(self, x, u):      
        u = u.view(x.shape[0], self.num_vars)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        ux = F.relu(self.fc1(x))
        ux = self.bn1(ux)
        ux = torch.cat([ux, u], 1)
        ux = F.relu(self.fc2(ux))
        ux = F.dropout(ux, training=self.training)
        ux = F.relu(self.fc3(ux))
        p_Y_UX = self.fc4(ux)
        if self.loss_type == 'mse':
            p_Y_UX = p_Y_UX.view(-1)
        
        return p_Y_UX
        
    def classify(self, x, u):
        x = x.view(x.shape[0], 1, 28, 28)
        return self.forward(x,u)
