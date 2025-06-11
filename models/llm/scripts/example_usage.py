#!/usr/bin/env python3
"""
Example usage of the VulnerabilitySeverityClassifier class.

This script demonstrates how to use the refactored class-based approach
to replace the functionality from classify.py.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from label import VulnerabilitySeverityClassifier

def main():
    """Main function demonstrating the VulnerabilitySeverityClassifier usage."""
    
    print("="*80)
    print("VULNERABILITY SEVERITY PREDICTION - CLASS-BASED APPROACH")
    print("="*80)
    
    # Initialize the classifier
    print("\n1. Initializing VulnerabilitySeverityClassifier...")
    classifier = VulnerabilitySeverityClassifier(
        device="auto",  # Automatically select GPU if available
        max_length=512
    )
    
    # Check if model loaded successfully
    print(f"   Model Status: {'✅ Ready' if classifier.is_ready() else '❌ Not Ready'}")
    print(f"   {classifier}")
    
    # Display model information
    model_info = classifier.get_model_info()
    print(f"\n2. Model Information:")
    print(f"   Device: {model_info['device']}")
    print(f"   Data Type: {model_info['torch_dtype']}")
    print(f"   Severity Labels: {model_info['severity_labels']}")
    print(f"   CVSS Mapping: {model_info['cvss_mapping']}")
    
    # Load and process data (same as original classify.py)
    print(f"\n3. Loading and processing CVE data...")
    try:
        df = pd.read_csv('catalogs_processed/data/historical_nvd.csv')
        print(f"   Loaded {len(df):,} records from historical NVD data")
        
        # Filter for recent CVEs (2023+)
        df = df[df['cve_id'].str.startswith('CVE-2023-') | 
                df['cve_id'].str.startswith('CVE-2024-') | 
                df['cve_id'].str.startswith('CVE-2025-')]
        
        # Select relevant columns and filter for quality
        columns_needed = ['cve_id', 'cvss_score', 'description']
        df_filtered = df[columns_needed].copy()
        
        # Keep only rows with both description and CVSS score
        df_final = df_filtered[
            df_filtered['description'].notna() & 
            df_filtered['cvss_score'].notna()
        ].copy()
        
        print(f"   After filtering: {len(df_final):,} records with description and CVSS score")
        
    except FileNotFoundError:
        print("   ⚠️  Historical NVD data not found. Using sample data for demonstration.")
        # Create sample data for demonstration
        df_final = pd.DataFrame({
            'cve_id': [
                'CVE-2023-0001', 'CVE-2023-0002', 'CVE-2023-0003', 
                'CVE-2024-0001', 'CVE-2024-0002'
            ],
            'cvss_score': [7.5, 9.8, 4.3, 6.1, 8.8],
            'description': [
                'A buffer overflow vulnerability in the network parsing component allows remote code execution.',
                'Critical authentication bypass vulnerability allowing unauthorized administrative access.',
                'Information disclosure vulnerability exposing sensitive configuration data.',
                'SQL injection vulnerability in user input validation allowing database manipulation.',
                'Cross-site scripting vulnerability enabling malicious script execution in user browsers.'
            ]
        })
    
    # Test predictions on sample data
    print(f"\n4. Testing predictions on sample data...")
    test_df = df_final.head(10).copy()  # Test on first 10 records
    
    # Get descriptions for prediction
    descriptions = test_df['description'].fillna("").tolist()
    
    # Make predictions using the class
    if classifier.is_ready():
        print(f"   Making predictions for {len(descriptions)} descriptions...")
        
        # Get predictions with probabilities
        predicted_severities, predicted_cvss, probabilities = classifier.predict(
            descriptions, 
            return_probabilities=True
        )
        
        # Add predictions to dataframe
        test_df['ai_severity'] = predicted_severities
        test_df['ai_cvss_score'] = predicted_cvss
        test_df['cvss_difference'] = test_df['ai_cvss_score'] - test_df['cvss_score']
        test_df['abs_cvss_difference'] = abs(test_df['cvss_difference'])
        
        # Display results
        print(f"\n5. Prediction Results:")
        print("="*100)
        print(f"{'CVE ID':<15} {'Original':<8} {'Predicted':<9} {'Difference':<10} {'Severity':<10} {'Confidence':<10}")
        print("-"*100)
        
        for idx, (_, row) in enumerate(test_df.iterrows()):
            confidence = max(probabilities[idx]) * 100  # Max probability as confidence
            print(f"{row['cve_id']:<15} {row['cvss_score']:<8.1f} {row['ai_cvss_score']:<9.1f} "
                  f"{row['cvss_difference']:+<10.1f} {row['ai_severity']:<10} {confidence:<10.1f}%")
        
        # Calculate and display statistics
        print(f"\n6. Performance Statistics:")
        print("="*50)
        mae = test_df['abs_cvss_difference'].mean()
        rmse = np.sqrt((test_df['cvss_difference']**2).mean())
        
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
        print(f"Original CVSS range: {test_df['cvss_score'].min():.1f} - {test_df['cvss_score'].max():.1f}")
        print(f"AI CVSS range: {test_df['ai_cvss_score'].min():.1f} - {test_df['ai_cvss_score'].max():.1f}")
        
        # Severity distribution
        print(f"\nSeverity Distribution:")
        severity_counts = test_df['ai_severity'].value_counts()
        for severity, count in severity_counts.items():
            percentage = (count / len(test_df)) * 100
            print(f"  {severity}: {count} CVEs ({percentage:.1f}%)")
        
        # Accuracy within thresholds
        within_05 = (test_df['abs_cvss_difference'] <= 0.5).sum()
        within_10 = (test_df['abs_cvss_difference'] <= 1.0).sum()
        within_15 = (test_df['abs_cvss_difference'] <= 1.5).sum()
        
        print(f"\nAccuracy within thresholds:")
        print(f"  Within ±0.5: {within_05}/{len(test_df)} ({within_05/len(test_df)*100:.1f}%)")
        print(f"  Within ±1.0: {within_10}/{len(test_df)} ({within_10/len(test_df)*100:.1f}%)")
        print(f"  Within ±1.5: {within_15}/{len(test_df)} ({within_15/len(test_df)*100:.1f}%)")
        
    else:
        print("   ❌ Model not ready. Cannot make predictions.")
    
    # Demonstrate single prediction
    print(f"\n7. Single Prediction Example:")
    print("-"*50)
    sample_description = "Remote code execution vulnerability in web application allowing arbitrary command execution"
    
    if classifier.is_ready():
        severity, cvss, probs = classifier.predict_single(
            sample_description, 
            return_probabilities=True
        )
        
        print(f"Description: {sample_description}")
        print(f"Predicted Severity: {severity}")
        print(f"Predicted CVSS Score: {cvss}")
        print(f"Confidence: {max(probs)*100:.1f}%")
        print(f"Probability Distribution:")
        for label, prob in zip(classifier.severity_labels, probs):
            print(f"  {label}: {prob*100:.1f}%")
    else:
        print("   ❌ Model not ready for single prediction.")
    
    print(f"\n✅ Example completed successfully!")

if __name__ == "__main__":
    main() 