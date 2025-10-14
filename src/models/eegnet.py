"""
EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces

Reference:
Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018).
EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces.
Journal of Neural Engineering, 15(5), 056013.

Optimized for:
- Small sample sizes (50-100 trials per subject)
- Low-channel EEG (8-16 channels)
- Motor imagery classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet model for EEG motor imagery classification.

    Architecture optimized for 8-channel, 2-second trials at 160Hz.

    Parameters:
    -----------
    n_channels : int
        Number of EEG channels (default: 8 for ADS1299)
    n_timepoints : int
        Number of time samples (default: 320 for 2s @ 160Hz)
    n_classes : int
        Number of output classes (default: 2 for left/right hand)
    dropout_rate : float
        Dropout probability (default: 0.5)
    F1 : int
        Number of temporal filters (default: 8)
    D : int
        Depth multiplier (default: 2)
    F2 : int
        Number of pointwise filters (default: 16)
    """

    def __init__(self, n_channels=8, n_timepoints=320, n_classes=2,
                 dropout_rate=0.5, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes

        # Layer 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Layer 2: Depthwise convolution (spatial filter)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                         groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)

        # Layer 3: Separable convolution
        self.separable_conv = nn.Conv2d(F1 * D, F2, (1, 16),
                                         padding='same', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate flattened size
        self.flatten_size = self._get_flatten_size(n_channels, n_timepoints)

        # Fully connected layer
        self.fc = nn.Linear(self.flatten_size, n_classes)

    def _get_flatten_size(self, n_channels, n_timepoints):
        """Calculate the size after convolutions and pooling"""
        # Simulate forward pass to get size
        x = torch.zeros(1, 1, n_channels, n_timepoints)
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.avg_pool1(x)
        x = self.separable_conv(x)
        x = self.avg_pool2(x)
        return x.numel()

    def forward(self, x):
        """
        Forward pass

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_channels, n_timepoints)

        Returns:
        --------
        torch.Tensor
            Output logits of shape (batch_size, n_classes)
        """
        # Add channel dimension: (batch, channels, time) -> (batch, 1, channels, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Block 1: Temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Block 2: Depthwise (spatial) convolution
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        # Block 3: Separable convolution
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)

        # Flatten and classify
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x

    def predict_proba(self, x):
        """Get probability predictions"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_num_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EEGNetLarge(EEGNet):
    """
    Larger variant of EEGNet for more complex datasets
    """
    def __init__(self, n_channels=8, n_timepoints=320, n_classes=2,
                 dropout_rate=0.5):
        super().__init__(n_channels, n_timepoints, n_classes,
                        dropout_rate, F1=16, D=2, F2=32)


def test_model():
    """Test EEGNet with dummy data"""
    print("Testing EEGNet Model")
    print("=" * 60)

    # Test configurations
    configs = [
        {'n_channels': 8, 'n_timepoints': 320, 'name': '8-ch, 2s @ 160Hz'},
        {'n_channels': 3, 'n_timepoints': 320, 'name': '3-ch, 2s @ 160Hz'},
        {'n_channels': 16, 'n_timepoints': 640, 'name': '16-ch, 4s @ 160Hz'},
    ]

    for config in configs:
        print(f"\nConfiguration: {config['name']}")
        print("-" * 60)

        # Create model
        model = EEGNet(n_channels=config['n_channels'],
                      n_timepoints=config['n_timepoints'],
                      n_classes=2)

        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, config['n_channels'], config['n_timepoints'])

        # Forward pass
        output = model(x)
        proba = model.predict_proba(x)

        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Probability shape: {proba.shape}")
        print(f"Parameters: {model.get_num_parameters():,}")
        print(f"Output logits (sample): {output[0].detach().numpy()}")
        print(f"Probabilities (sample): {proba[0].detach().numpy()}")

        # Check gradient flow
        loss = F.cross_entropy(output, torch.tensor([0, 1, 0, 1]))
        loss.backward()
        print(f"Loss: {loss.item():.4f}")
        print("✓ Gradient flow check passed")

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    test_model()