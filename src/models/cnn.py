import torch
import torch.nn as nn
import torch.nn.functional as F

class DualViewCNN(nn.Module):
    """
    Dual-view 1D CNN for Exoplanet Transit Classification.
    Accepts global and local phase-folded light curves as input.
    """
    def __init__(self, global_size: int = 200, local_size: int = 80, num_classes: int = 4):
        super(DualViewCNN, self).__init__()
        
        # Global view feature extractor (200 points)
        self.global_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output size: 100
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output size: 50
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)   # Output size: 25
        )
        
        self.global_fc = nn.Sequential(
            nn.Linear(64 * 25, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Local view feature extractor (80 points)
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output size: 40
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # Output size: 20
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)   # Output size: 10
        )
        
        self.local_fc = nn.Sequential(
            nn.Linear(64 * 10, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Joint classification head
        self.fc_combined = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, global_input: torch.Tensor, local_input: torch.Tensor) -> torch.Tensor:
        # Inputs should have shape [batch, size]. Reshape to [batch, 1, size] for Conv1d
        if len(global_input.shape) == 1:
            global_input = global_input.unsqueeze(0)
        if len(local_input.shape) == 1:
            local_input = local_input.unsqueeze(0)
            
        g = global_input.unsqueeze(1)
        l = local_input.unsqueeze(1)
        
        # Process global view
        g_feat = self.global_conv(g)
        g_flat = g_feat.view(g_feat.size(0), -1)
        g_embed = self.global_fc(g_flat)
        
        # Process local view
        l_feat = self.local_conv(l)
        l_flat = l_feat.view(l_feat.size(0), -1)
        l_embed = self.local_fc(l_flat)
        
        # Combine views
        combined = torch.cat((g_embed, l_embed), dim=1)
        logits = self.fc_combined(combined)
        return logits
