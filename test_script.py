#!/usr/bin/env python3
"""Test key functions from contained_classify.py"""

from contained_classify import severity_from_v2_score, choose_metric, CVE_REGEX, CVSS_PROXY
import re

def test_v2_score_mapping():
    """Test CVSS v2 score to severity mapping"""
    print("=== TESTING CVSS v2 SCORE MAPPING ===")
    test_cases = [
        (0.0, "NONE"),
        (2.0, "LOW"),
        (3.9, "LOW"),
        (4.0, "MEDIUM"),
        (6.9, "MEDIUM"),
        (7.0, "HIGH"),
        (10.0, "HIGH")
    ]
    
    for score, expected in test_cases:
        result = severity_from_v2_score(score)
        status = "✅" if result == expected else "❌"
        print(f"  {status} Score {score} → {result} (expected {expected})")

def test_cvss_proxy_scores():
    """Test CVSS proxy score mapping"""
    print("\n=== TESTING CVSS PROXY SCORES ===")
    for severity, score in CVSS_PROXY.items():
        print(f"  {severity}: {score}")

def test_cve_regex():
    """Test CVE ID extraction regex"""
    print("\n=== TESTING CVE REGEX ===")
    test_texts = [
        "CVE-2024-1234 is a critical vulnerability",
        "Check CVE-2023-56789 and cve-2022-1111",
        "Multiple: CVE-2024-0001, CVE-2024-0002",
        "Invalid: CVE-24-123, CVE-2024-ABC",
        "No CVEs here"
    ]
    
    for text in test_texts:
        matches = CVE_REGEX.findall(text)
        print(f"  Text: '{text[:30]}...'")
        print(f"  CVEs: {matches}")

def test_version_priority():
    """Test CVSS version selection priority"""
    print("\n=== TESTING VERSION PRIORITY ===")
    
    # Test data with multiple versions
    test_metrics = {
        "cvssMetricV2": [{"cvssData": {"version": "2.0", "baseScore": 7.5}}],
        "cvssMetricV31": [{"cvssData": {"version": "3.1", "baseScore": 8.2}}],
        "cvssMetricV40": [{"cvssData": {"version": "4.0", "baseScore": 9.1}}]
    }
    
    chosen_data, version_key = choose_metric(test_metrics)
    print(f"  Available versions: {list(test_metrics.keys())}")
    print(f"  Chosen: {version_key}")
    print(f"  Score: {chosen_data.get('baseScore') if chosen_data else 'None'}")

if __name__ == "__main__":
    test_v2_score_mapping()
    test_cvss_proxy_scores()
    test_cve_regex()
    test_version_priority()
    print("\n✅ All tests completed!") 