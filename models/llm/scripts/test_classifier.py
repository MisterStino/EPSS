#!/usr/bin/env python3
"""
Simple test script for VulnerabilitySeverityClassifier class.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from label import VulnerabilitySeverityClassifier

def test_basic_functionality():
    """Test basic functionality of the classifier."""
    print("Testing VulnerabilitySeverityClassifier...")
    
    # Initialize classifier
    classifier = VulnerabilitySeverityClassifier()
    
    # Test model info
    info = classifier.get_model_info()
    print(f"Model loaded: {info['is_loaded']}")
    print(f"Model ready: {info['is_ready']}")
    print(f"Device: {info['device']}")
    
    # Test single prediction
    test_description = "Buffer overflow vulnerability allowing remote code execution"
    
    if classifier.is_ready():
        severity, cvss = classifier.predict_single(test_description)
        print(f"\nTest Prediction:")
        print(f"Description: {test_description}")
        print(f"Predicted Severity: {severity}")
        print(f"Predicted CVSS: {cvss}")
        
        # Test batch prediction
        descriptions = [
            "SQL injection vulnerability in login form",
            "Cross-site scripting in user comments",
            "Authentication bypass in admin panel"
        ]
        
        severities, cvss_scores = classifier.predict(descriptions)
        print(f"\nBatch Predictions:")
        for desc, sev, cvss in zip(descriptions, severities, cvss_scores):
            print(f"  {desc[:50]}... -> {sev} ({cvss})")
    
    else:
        print("❌ Model not ready for testing")
    
    print(f"\n✅ Basic test completed!")

if __name__ == "__main__":
    test_basic_functionality() 