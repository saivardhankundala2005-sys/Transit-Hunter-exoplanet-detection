import numpy as np
import os
import pickle
import torch
import logging
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from src.models.cnn import DualViewCNN

logger = logging.getLogger(__name__)

class PhysicsClassifier:
    """Classifier utilizing physics-based engineered features."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Use XGBoost if available, otherwise fallback to RandomForest
        if xgb is not None:
            logger.info("Initializing XGBoost classifier for the Physics branch.")
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_state,
                eval_metric="mlogloss"
            )
        else:
            logger.info("XGBoost not available. Initializing RandomForest classifier.")
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the physics classifier."""
        logger.info("Training Physics branch classifier...")
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict classes."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def save(self, file_path: str) -> None:
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Physics model saved to {file_path}")

    def load(self, file_path: str) -> None:
        """Load model checkpoint."""
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Physics model loaded from {file_path}")


class HybridEnsemble:
    """Ensemble combining deep learning (CNN) and physics branch classifiers."""

    def __init__(self, 
                 cnn_model: Optional[DualViewCNN] = None, 
                 physics_model: Optional[PhysicsClassifier] = None,
                 cnn_weight: float = 0.4):
        self.cnn_model = cnn_model
        self.physics_model = physics_model
        self.cnn_weight = cnn_weight
        self.class_names = [
            "Exoplanet Transit",
            "Eclipsing Binary",
            "Stellar Blend",
            "Detector Artifact"
        ]

    def predict_proba(self, 
                      global_curve: np.ndarray, 
                      local_curve: np.ndarray, 
                      physics_features: np.ndarray) -> np.ndarray:
        """Combine predictions from both branches."""
        # CNN probabilities
        if self.cnn_model is not None:
            self.cnn_model.eval()
            with torch.no_grad():
                g_tensor = torch.FloatTensor(global_curve)
                l_tensor = torch.FloatTensor(local_curve)
                if len(g_tensor.shape) == 1:
                    g_tensor = g_tensor.unsqueeze(0)
                if len(l_tensor.shape) == 1:
                    l_tensor = l_tensor.unsqueeze(0)
                    
                logits = self.cnn_model(g_tensor, l_tensor)
                cnn_probs = torch.softmax(logits, dim=1).numpy()
        else:
            # Equal probability dummy
            cnn_probs = np.ones((len(physics_features), 4)) * 0.25
            
        # Physics probabilities
        if self.physics_model is not None:
            # Reshape features to 2D if single sample
            if len(physics_features.shape) == 1:
                features_2d = physics_features.reshape(1, -1)
            else:
                features_2d = physics_features
            physics_probs = self.physics_model.predict_proba(features_2d)
        else:
            physics_probs = np.ones((len(physics_features), 4)) * 0.25
            
        # Ensembled probability
        final_probs = self.cnn_weight * cnn_probs + (1.0 - self.cnn_weight) * physics_probs
        return final_probs

    def classify_candidate(self, 
                           global_curve: np.ndarray, 
                           local_curve: np.ndarray, 
                           physics_features: Dict[str, float]) -> Dict[str, Any]:
        """Classify a single exoplanet candidate."""
        # Extract features array in the correct order
        feature_keys = [
            "transit_depth", "transit_duration", "period", "epoch",
            "ingress_slope", "egress_slope", "symmetry_score", "u_v_score",
            "rms_noise", "odd_even_difference", "transit_signal_strength"
        ]
        features_array = np.array([physics_features[k] for k in feature_keys]).reshape(1, -1)
        
        probs = self.predict_proba(global_curve, local_curve, features_array)[0]
        pred_class_idx = np.argmax(probs)
        pred_class_name = self.class_names[pred_class_idx]
        confidence = probs[pred_class_idx]
        
        return {
            "class_idx": int(pred_class_idx),
            "class_name": pred_class_name,
            "confidence": float(confidence),
            "probabilities": {name: float(prob) for name, prob in zip(self.class_names, probs)}
        }
