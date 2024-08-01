import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, action_size, seed=42):
        super(DuelingDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size of the output from the last convolutional layer
        self.conv_output_size = self._get_conv_output_size((3, 128, 128))

        # Value stream
        self.value_fc1 = nn.Linear(in_features=self.conv_output_size, out_features=512)
        self.value_fc2 = nn.Linear(in_features=512, out_features=1)

        # Advantage stream
        self.adv_fc1 = nn.Linear(in_features=self.conv_output_size, out_features=512)
        self.adv_fc2 = nn.Linear(in_features=512, out_features=action_size)

    def _get_conv_output_size(self, shape):
        x = torch.rand(1, *shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)

        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)

        adv = F.relu(self.adv_fc1(x))
        adv = self.adv_fc2(adv)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q = value + adv - adv.mean(1, keepdim=True)
        return q