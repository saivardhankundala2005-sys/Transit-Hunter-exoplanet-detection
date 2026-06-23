import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)

class GradCAM1D:
    """Class to perform Grad-CAM (Class Activation Mapping) on 1D convolutional layers."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self.forward_hook = self.target_layer.register_forward_hook(self._save_activation)
        # register_full_backward_hook is preferred in newer PyTorch versions, 
        # but register_backward_hook works across a wider compatibility range.
        try:
            self.backward_hook = self.target_layer.register_full_backward_hook(self._save_gradient)
        except AttributeError:
            self.backward_hook = self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Remove hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()

    def generate(self, 
                 global_input: np.ndarray, 
                 local_input: np.ndarray, 
                 class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Generate Grad-CAM activation map for the local input."""
        self.model.eval()
        
        # Convert arrays to tensors
        g_tensor = torch.FloatTensor(global_input)
        l_tensor = torch.FloatTensor(local_input)
        if len(g_tensor.shape) == 1:
            g_tensor = g_tensor.unsqueeze(0)
        if len(l_tensor.shape) == 1:
            l_tensor = l_tensor.unsqueeze(0)

        # Forward pass
        logits = self.model(g_tensor, l_tensor)
        
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
            
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM hooks failed to capture gradients/activations. Returning uniform values.")
            return np.ones(local_input.shape[-1]) / 2.0, class_idx

        # Calculate weight of each channel based on average gradient
        # gradients shape: [batch, channels, length]
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        
        # Weighted sum of channel activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
        
        # Normalize cam between 0.0 and 1.0
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        # Interpolate cam back to the original size of the local input
        cam_resized = F.interpolate(cam, size=l_tensor.shape[-1], mode='linear', align_corners=False)
        cam_array = cam_resized.squeeze().cpu().numpy()
        
        return cam_array, class_idx


class ExplainabilityEngine:
    """Engine to generate both Deep Learning (Grad-CAM) and Physics (SHAP) explanations."""

    def __init__(self, cnn_model: Optional[nn.Module] = None, physics_classifier: Optional[Any] = None):
        self.cnn_model = cnn_model
        self.physics_classifier = physics_classifier

    def get_cnn_explanation(self, 
                            global_input: np.ndarray, 
                            local_input: np.ndarray, 
                            class_idx: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Compute Grad-CAM maps on the local transit window."""
        if self.cnn_model is None:
            raise ValueError("CNN model not loaded inside ExplainabilityEngine.")
            
        # Find target layer: final Conv1d layer of local view path
        target_layer = self.cnn_model.local_conv[8]
        
        grad_cam = GradCAM1D(self.cnn_model, target_layer)
        try:
            cam, predicted_idx = grad_cam.generate(global_input, local_input, class_idx)
        finally:
            grad_cam.remove_hooks()
            
        return cam, predicted_idx

    def get_physics_explanation(self, 
                                 sample_features: Dict[str, float], 
                                 feature_names: List[str],
                                 background_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute feature importances / SHAP contributions for a single sample.
        Falls back to local perturbation sensitivity if shap is unavailable.
        """
        if self.physics_classifier is None:
            raise ValueError("Physics classifier model not loaded in ExplainabilityEngine.")
            
        # Convert dictionary to array in correct order
        x_sample = np.array([sample_features[name] for name in feature_names]).reshape(1, -1)
        
        try:
            import shap
            logger.info("Using SHAP to explain physics model prediction...")
            
            # Use TreeExplainer or KernelExplainer depending on target model type
            model_to_explain = self.physics_classifier.model
            
            if background_data is not None:
                explainer = shap.Explainer(model_to_explain, background_data)
            else:
                explainer = shap.Explainer(model_to_explain)
                
            shap_values = explainer(x_sample)
            
            # shap_values shape can be [1, num_features] (binary classification or tree explainer layout)
            # or [1, num_features, num_classes] (multiclass)
            vals = shap_values.values
            if len(vals.shape) == 3:
                # Average absolute values across classes or select target class
                # We show class contribution for index 0 (Exoplanet Transit)
                vals_exoplanet = vals[0, :, 0]
            else:
                vals_exoplanet = vals[0]
                
            contributions = {name: float(val) for name, val in zip(feature_names, vals_exoplanet)}
            return contributions
            
        except (ImportError, Exception) as e:
            logger.warning(f"SHAP failed or not installed ({str(e)}). Using perturbation sensitivity analysis.")
            return self._calculate_perturbation_sensitivity(x_sample, feature_names)

    def _calculate_perturbation_sensitivity(self, 
                                            x_sample: np.ndarray, 
                                            feature_names: List[str]) -> Dict[str, float]:
        """
        Analytical explainability fallback: perturb features slightly and measure impact 
        on the Exoplanet class probability.
        """
        base_probs = self.physics_classifier.predict_proba(x_sample)[0]
        base_prob = base_probs[0]  # Exoplanet class probability
        
        sensitivities = {}
        for idx, name in enumerate(feature_names):
            # Perturb feature by 10% of its value (or +0.01 if zero)
            x_perturbed = x_sample.copy()
            val = x_perturbed[0, idx]
            delta = val * 0.1 if val != 0 else 0.05
            
            x_perturbed[0, idx] += delta
            new_probs = self.physics_classifier.predict_proba(x_perturbed)[0]
            new_prob = new_probs[0]
            
            # Sensitivity = change in probability / delta
            sensitivities[name] = float(new_prob - base_prob)
            
        return sensitivities

    def plot_gradcam_overlay(self, 
                             local_phase: np.ndarray, 
                             local_flux: np.ndarray, 
                             cam: np.ndarray, 
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot binned local flux overlaid with Grad-CAM heatmap."""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot local light curve
        ax.plot(local_phase, local_flux, 'o-', color='black', label='Folded Light Curve', alpha=0.5)
        
        # Overlay heatmap using a scatter plot with colored markers
        # cmap 'hot' or 'plasma' where bright colors indicate high influence
        sc = ax.scatter(local_phase, local_flux, c=cam, cmap='plasma', s=80, zorder=5, label='Grad-CAM Importance')
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Importance (Influence on Class Decision)')
        
        ax.set_xlabel('Phase')
        ax.set_ylabel('Normalized Flux (Offset)')
        ax.set_title('Explainable AI: CNN Grad-CAM Attribution Map')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Grad-CAM overlay saved to {save_path}")
            
        return fig

    def plot_feature_importance(self, 
                                 contributions: Dict[str, float], 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot vertical bar chart showing how each physics feature contributed to prediction."""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Sort features by magnitude of contribution
        sorted_features = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        names = [item[0].replace('_', ' ').title() for item in sorted_features]
        values = [item[1] for item in sorted_features]
        
        # Colors: green for positive contribution, red for negative
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, align='center', color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # top-down listing
        ax.set_xlabel('Prediction Score Impact (Direction & Strength)')
        ax.set_title('Explainable AI: Physics Feature Attribution')
        
        # Add centerline
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
            
        return fig
