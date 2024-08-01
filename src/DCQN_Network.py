import torch
import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
    """
    Neural network architecture for the MsPacmanDeterministic-v0 environment.

    This network takes game screen images as input and outputs Q-values for each possible action.
    It consists of convolutional layers for feature extraction followed by fully connected layers
    for Q-value estimation.

    Attributes:
        conv1, conv2, conv3, conv4 (nn.Conv2d): Convolutional layers
        bn1, bn2, bn3, bn4 (nn.BatchNorm2d): Batch normalization layers
        fc1, fc2, fc3 (nn.Linear): Fully connected layers
    """

    def __init__(self, action_size: int, seed: int = 42):
        """
        Initialize the Network.

        Args:
            action_size (int): Number of possible actions in the environment.
            seed (int): Random seed for reproducibility.
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Convolutional layers
        # Each layer is followed by batch normalization to stabilize learning
        # Kernel sizes and strides are chosen to efficiently downsample the input
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Fully connected layers
        # The input size (12800) is calculated based on the output shape of the last conv layer
        # Output sizes decrease gradually to map the high-dimensional input to action values
        self.fc1 = nn.Linear(in_features=12800, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Perform forward pass through the network.

        Args:
            state (torch.Tensor): Input tensor representing the game state (screen image).
                Expected shape: (batch_size, 3, height, width)

        Returns:
            torch.Tensor: Q-values for each possible action.
                Shape: (batch_size, action_size)
        """
        # Pass input through convolutional layers
        # ReLU activation is applied after each conv + batch norm operation
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Flatten the tensor for the fully connected layers
        # Keep the batch dimension (dim 0) and flatten the rest
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        # ReLU activation is applied after the first two FC layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # The final layer outputs Q-values without activation
        return self.fc3(x)