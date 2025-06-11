# LLM to ingest text and output label

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VulnerabilitySeverityClassifier:
    """
    A class for predicting vulnerability severity levels and CVSS scores from text descriptions.
    
    This class encapsulates the CIRCL vulnerability severity classification model and provides
    a clean interface for predicting severity levels (low, medium, high, critical) and 
    corresponding CVSS scores from vulnerability descriptions.
    
    Attributes:
        model_name (str): Name/path of the pre-trained model
        device (str): Device where the model is loaded ('cpu', 'cuda', or 'auto')
        severity_labels (List[str]): Available severity categories
        cvss_mapping (Dict[str, float]): Mapping from severity to CVSS scores
        is_loaded (bool): Whether the model is successfully loaded
    """
    
    # Default configuration
    DEFAULT_MODEL_NAME = "CIRCL/vulnerability-severity-classification-roberta-base"
    DEFAULT_SEVERITY_LABELS = ["low", "medium", "high", "critical"]
    DEFAULT_CVSS_MAPPING = {
        "low": 3.0,      # 0.1-3.9 range
        "medium": 6.0,   # 4.0-6.9 range  
        "high": 8.0,     # 7.0-8.9 range
        "critical": 9.5  # 9.0-10.0 range
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "auto",
        severity_labels: Optional[List[str]] = None,
        cvss_mapping: Optional[Dict[str, float]] = None,
        max_length: int = 512,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize the VulnerabilitySeverityClassifier.
        
        Args:
            model_name: Name or path of the pre-trained model. Defaults to CIRCL model.
            device: Device to load model on ('cpu', 'cuda', 'auto'). 'auto' selects GPU if available.
            severity_labels: Custom severity labels. Must match model output classes.
            cvss_mapping: Custom mapping from severity labels to CVSS scores.
            max_length: Maximum token length for input text (default: 512).
            torch_dtype: PyTorch data type for model. Auto-selected if None.
        """
        # Configuration
        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.severity_labels = severity_labels or self.DEFAULT_SEVERITY_LABELS.copy()
        self.cvss_mapping = cvss_mapping or self.DEFAULT_CVSS_MAPPING.copy()
        self.max_length = max_length
        
        # Validate inputs
        self._validate_configuration()
        
        # Device management
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype or self._get_optimal_dtype()
        
        # Model components (initialized as None)
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        # Load model
        self._load_model()
    
    def _validate_configuration(self) -> None:
        """Validate the configuration parameters."""
        if not self.severity_labels:
            raise ValueError("severity_labels cannot be empty")
        
        if not self.cvss_mapping:
            raise ValueError("cvss_mapping cannot be empty")
        
        # Check that all severity labels have corresponding CVSS mappings
        missing_mappings = set(self.severity_labels) - set(self.cvss_mapping.keys())
        if missing_mappings:
            raise ValueError(f"Missing CVSS mappings for severity labels: {missing_mappings}")
        
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available. Using CPU.")
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        
        return device
    
    def _get_optimal_dtype(self) -> torch.dtype:
        """Get optimal data type based on device."""
        if self.device == "cuda":
            return torch.float16  # Memory efficient for GPU
        else:
            return torch.float32  # Standard for CPU
    
    def _load_model(self) -> None:
        """Load the tokenizer and model with error handling."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model with device and dtype configuration
            model_kwargs = {
                "torch_dtype": self.torch_dtype,
            }
            
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            logger.info(f"✅ Model loaded successfully on {self.model.device}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
            self.model = None
            self.tokenizer = None
            warnings.warn(f"Model loading failed: {e}. Predictions will return 'unknown'.")
    
    def predict(
        self, 
        descriptions: Union[str, List[str]], 
        return_probabilities: bool = False
    ) -> Union[Tuple[List[str], List[float]], Tuple[List[str], List[float], List[List[float]]]]:
        """
        Predict severity levels and CVSS scores from vulnerability descriptions.
        
        Args:
            descriptions: Single description string or list of descriptions.
            return_probabilities: If True, also return prediction probabilities.
        
        Returns:
            Tuple containing:
            - predicted_severities: List of predicted severity labels
            - predicted_cvss: List of predicted CVSS scores
            - probabilities (optional): List of probability distributions
        
        Raises:
            ValueError: If descriptions is empty or invalid.
        """
        # Input validation and normalization
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        
        if not descriptions:
            raise ValueError("descriptions cannot be empty")
        
        # Handle model not loaded
        if not self.is_loaded:
            logger.warning("Model not loaded. Returning 'unknown' predictions.")
            unknown_severities = ["unknown"] * len(descriptions)
            zero_cvss = [0.0] * len(descriptions)
            if return_probabilities:
                zero_probs = [[0.0] * len(self.severity_labels)] * len(descriptions)
                return unknown_severities, zero_cvss, zero_probs
            return unknown_severities, zero_cvss
        
        # Perform prediction
        return self._predict_batch(descriptions, return_probabilities)
    
    def _predict_batch(
        self, 
        descriptions: List[str], 
        return_probabilities: bool = False
    ) -> Union[Tuple[List[str], List[float]], Tuple[List[str], List[float], List[List[float]]]]:
        """Internal batch prediction method."""
        try:
            # Tokenize descriptions
            inputs = self.tokenizer(
                descriptions,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to model device
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_indices = torch.argmax(probabilities, dim=-1)
            
            # Convert to CPU numpy arrays
            predicted_indices_np = predicted_indices.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            
            # Map to severity labels and CVSS scores
            predicted_severities = [self.severity_labels[idx] for idx in predicted_indices_np]
            predicted_cvss = [self.cvss_mapping[severity] for severity in predicted_severities]
            
            if return_probabilities:
                return predicted_severities, predicted_cvss, probabilities_np.tolist()
            
            return predicted_severities, predicted_cvss
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return safe defaults
            unknown_severities = ["unknown"] * len(descriptions)
            zero_cvss = [0.0] * len(descriptions)
            if return_probabilities:
                zero_probs = [[0.0] * len(self.severity_labels)] * len(descriptions)
                return unknown_severities, zero_cvss, zero_probs
            return unknown_severities, zero_cvss
    
    def predict_single(
        self, 
        description: str, 
        return_probabilities: bool = False
    ) -> Union[Tuple[str, float], Tuple[str, float, List[float]]]:
        """
        Predict severity for a single description (convenience method).
        
        Args:
            description: Single vulnerability description.
            return_probabilities: If True, also return prediction probabilities.
        
        Returns:
            Tuple containing:
            - predicted_severity: Predicted severity label
            - predicted_cvss: Predicted CVSS score
            - probabilities (optional): Probability distribution
        """
        result = self.predict([description], return_probabilities)
        
        if return_probabilities:
            severities, cvss_scores, probs = result
            return severities[0], cvss_scores[0], probs[0]
        else:
            severities, cvss_scores = result
            return severities[0], cvss_scores[0]
    
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for predictions."""
        return self.is_loaded and self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Union[str, List[str], Dict[str, float], bool]]:
        """Get information about the loaded model and configuration."""
        return {
            "model_name": self.model_name,
            "device": str(self.model.device) if self.model else "not_loaded",
            "torch_dtype": str(self.torch_dtype),
            "severity_labels": self.severity_labels.copy(),
            "cvss_mapping": self.cvss_mapping.copy(),
            "max_length": self.max_length,
            "is_loaded": self.is_loaded,
            "is_ready": self.is_ready()
        }
    
    def update_cvss_mapping(self, new_mapping: Dict[str, float]) -> None:
        """
        Update the CVSS mapping for severity levels.
        
        Args:
            new_mapping: New mapping from severity labels to CVSS scores.
        
        Raises:
            ValueError: If mapping is invalid or incomplete.
        """
        # Validate new mapping
        missing_labels = set(self.severity_labels) - set(new_mapping.keys())
        if missing_labels:
            raise ValueError(f"Missing mappings for severity labels: {missing_labels}")
        
        self.cvss_mapping = new_mapping.copy()
        logger.info("CVSS mapping updated successfully")
    
    def __repr__(self) -> str:
        """String representation of the classifier."""
        status = "ready" if self.is_ready() else "not_ready"
        device = str(self.model.device) if self.model else "not_loaded"
        return f"VulnerabilitySeverityClassifier(model='{self.model_name}', device='{device}', status='{status}')"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()

if __name__ == "__main__":
    classifier = VulnerabilitySeverityClassifier()
    print(classifier.predict("A vulnerability in the OpenSSL library allows attackers to bypass authentication and gain unauthorized access to sensitive data.")) #For the second time this month, a Fediverse project reports a critical vulnerability. The devs are on top of it: admins, update your servers!https://wedistribute.org/2024/02/pixelfed-cve/
