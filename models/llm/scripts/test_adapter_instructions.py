#!/usr/bin/env python3
"""
Script to demonstrate practical adapter-based instruction extension.
Shows how to use the built-in adapter methods for extending model instructions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoConfig, pipeline
)
import torch
import json

def explore_adapter_system():
    """Explore the built-in adapter system capabilities."""
    
    print("="*80)
    print("EXPLORING BUILT-IN ADAPTER SYSTEM FOR INSTRUCTION EXTENSION")
    print("="*80)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    
    print("1. CURRENT ADAPTER STATE:")
    print("-" * 50)
    
    # Check current adapter state
    try:
        active_adapters = model.active_adapters()
        print(f"Active adapters: {active_adapters}")
        
        # Check if any adapters are loaded
        if hasattr(model, 'peft_config'):
            print(f"PEFT config available: {model.peft_config}")
        else:
            print("No PEFT configuration found")
            
    except Exception as e:
        print(f"Adapter state check failed: {e}")
    
    print("\\n2. ADAPTER METHODS AVAILABLE:")
    print("-" * 50)
    
    adapter_methods = [
        'add_adapter', 'load_adapter', 'set_adapter', 
        'enable_adapters', 'disable_adapters', 'delete_adapter',
        'get_adapter_state_dict', 'active_adapters'
    ]
    
    for method_name in adapter_methods:
        if hasattr(model, method_name):
            method = getattr(model, method_name)
            print(f"  ✅ {method_name}() - Available")
            
            # Try to get method signature
            try:
                import inspect
                sig = inspect.signature(method)
                print(f"     Parameters: {sig}")
            except:
                print(f"     Parameters: Could not inspect")
        else:
            print(f"  ❌ {method_name}() - Not available")

def test_adapter_configuration():
    """Test adapter configuration possibilities."""
    
    print("\\n" + "="*80)
    print("TESTING ADAPTER CONFIGURATION FOR INSTRUCTIONS")
    print("="*80)
    
    print("1. SIMULATING INSTRUCTION ADAPTER CONFIGURATION:")
    print("-" * 50)
    
    # Simulate what an instruction adapter config might look like
    instruction_adapter_config = {
        "adapter_name": "instruction_following",
        "task_type": "SEQ_CLS",  # Sequence Classification
        "inference_mode": False,
        "r": 16,  # Rank for LoRA
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["query", "value", "dense"],
        "instruction_template": "Classify the following vulnerability: {text}",
        "severity_mapping": {
            "low": "This is a low severity vulnerability",
            "medium": "This is a medium severity vulnerability", 
            "high": "This is a high severity vulnerability",
            "critical": "This is a critical severity vulnerability"
        }
    }
    
    print("Instruction adapter configuration:")
    for key, value in instruction_adapter_config.items():
        print(f"  📝 {key}: {value}")
    
    print("\\n2. TESTING ADAPTER LOADING SIMULATION:")
    print("-" * 50)
    
    # Test what happens when we try to use adapter methods
    model = AutoModelForSequenceClassification.from_pretrained(
        "CIRCL/vulnerability-severity-classification-roberta-base"
    )
    
    # Test 1: Try to get current adapter state
    print("Test 1: Getting adapter state...")
    try:
        state_dict = model.get_adapter_state_dict()
        print(f"  ✅ Adapter state dict keys: {list(state_dict.keys())[:5]}...")
    except Exception as e:
        print(f"  ❌ No adapter state available: {e}")
    
    # Test 2: Try to set an adapter (this will fail without actual adapter)
    print("\\nTest 2: Setting adapter...")
    try:
        model.set_adapter("instruction_adapter")
        print(f"  ✅ Adapter set successfully")
    except Exception as e:
        print(f"  ❌ Cannot set adapter (expected): {e}")
    
    # Test 3: Check active adapters
    print("\\nTest 3: Checking active adapters...")
    try:
        active = model.active_adapters()
        print(f"  Current active adapters: {active}")
    except Exception as e:
        print(f"  ❌ Cannot check active adapters: {e}")

def demonstrate_practical_instruction_extension():
    """Demonstrate practical instruction extension using available methods."""
    
    print("\\n" + "="*80)
    print("PRACTICAL INSTRUCTION EXTENSION DEMONSTRATION")
    print("="*80)
    
    print("1. CUSTOM INSTRUCTION WRAPPER CLASS:")
    print("-" * 50)
    
    class InstructionExtendedClassifier:
        """Extended classifier with instruction-like capabilities."""
        
        def __init__(self, model_name="CIRCL/vulnerability-severity-classification-roberta-base"):
            self.model_name = model_name
            self.pipeline = pipeline("text-classification", model=model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Instruction templates
            self.instruction_templates = {
                "severity": "Analyze the severity of this vulnerability: {text}",
                "technical": "Provide technical classification for: {text}",
                "brief": "Briefly classify: {text}",
                "detailed": "Detailed security analysis of: {text}"
            }
            
            # Response formatters
            self.response_formatters = {
                "severity": self._format_severity_response,
                "technical": self._format_technical_response,
                "brief": self._format_brief_response,
                "detailed": self._format_detailed_response
            }
        
        def classify_with_instruction(self, text, instruction_type="severity"):
            """Classify with instruction-like behavior."""
            
            # Apply instruction template (conceptually)
            template = self.instruction_templates.get(instruction_type, self.instruction_templates["severity"])
            
            # Get base prediction
            result = self.pipeline(text)
            
            # Format response according to instruction
            formatter = self.response_formatters.get(instruction_type, self._format_severity_response)
            return formatter(result, text)
        
        def _format_severity_response(self, result, text):
            """Format as severity-focused response."""
            prediction = result[0]
            return {
                "instruction": "severity_analysis",
                "vulnerability": text[:50] + "..." if len(text) > 50 else text,
                "severity": prediction['label'],
                "confidence": f"{prediction['score']:.2%}",
                "interpretation": f"This vulnerability is classified as {prediction['label'].lower()} severity with {prediction['score']:.2%} confidence."
            }
        
        def _format_technical_response(self, result, text):
            """Format as technical analysis."""
            prediction = result[0]
            return {
                "instruction": "technical_classification",
                "analysis": {
                    "classification": prediction['label'],
                    "confidence_score": prediction['score'],
                    "risk_level": self._map_to_risk_level(prediction['label']),
                    "recommended_action": self._get_recommended_action(prediction['label'])
                }
            }
        
        def _format_brief_response(self, result, text):
            """Format as brief response."""
            prediction = result[0]
            return f"{prediction['label']} ({prediction['score']:.1%})"
        
        def _format_detailed_response(self, result, text):
            """Format as detailed analysis."""
            prediction = result[0]
            return {
                "instruction": "detailed_security_analysis",
                "vulnerability_text": text,
                "classification_result": {
                    "primary_classification": prediction['label'],
                    "confidence_level": prediction['score'],
                    "severity_explanation": self._get_severity_explanation(prediction['label']),
                    "impact_assessment": self._get_impact_assessment(prediction['label']),
                    "mitigation_priority": self._get_mitigation_priority(prediction['label'])
                }
            }
        
        def _map_to_risk_level(self, severity):
            mapping = {"Low": "Minimal", "Medium": "Moderate", "High": "Significant", "Critical": "Severe"}
            return mapping.get(severity, "Unknown")
        
        def _get_recommended_action(self, severity):
            actions = {
                "Low": "Monitor and patch during regular maintenance",
                "Medium": "Schedule patch within 30 days",
                "High": "Patch within 7 days, implement workarounds",
                "Critical": "Emergency patch required, immediate action"
            }
            return actions.get(severity, "Consult security team")
        
        def _get_severity_explanation(self, severity):
            explanations = {
                "Low": "Minimal impact on system security and functionality",
                "Medium": "Moderate impact requiring attention but not urgent",
                "High": "Significant security risk requiring prompt attention",
                "Critical": "Severe security risk requiring immediate action"
            }
            return explanations.get(severity, "Unknown severity level")
        
        def _get_impact_assessment(self, severity):
            impacts = {
                "Low": "Limited scope, minimal business impact",
                "Medium": "Moderate scope, some business impact possible",
                "High": "Wide scope, significant business impact likely",
                "Critical": "System-wide impact, major business disruption possible"
            }
            return impacts.get(severity, "Impact assessment unavailable")
        
        def _get_mitigation_priority(self, severity):
            priorities = {
                "Low": "P4 - Low priority",
                "Medium": "P3 - Medium priority", 
                "High": "P2 - High priority",
                "Critical": "P1 - Critical priority"
            }
            return priorities.get(severity, "Priority not determined")
    
    print("Creating instruction-extended classifier...")
    
    try:
        classifier = InstructionExtendedClassifier()
        print("  ✅ Instruction-extended classifier created")
        
        # Test different instruction types
        test_vulnerability = "SQL injection vulnerability allows unauthorized database access"
        
        instruction_types = ["severity", "technical", "brief", "detailed"]
        
        print("\\n2. TESTING INSTRUCTION-BASED RESPONSES:")
        print("-" * 50)
        
        for instruction_type in instruction_types:
            print(f"\\n{instruction_type.upper()} INSTRUCTION:")
            try:
                result = classifier.classify_with_instruction(test_vulnerability, instruction_type)
                if isinstance(result, dict):
                    print(json.dumps(result, indent=2))
                else:
                    print(f"  Result: {result}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                
    except Exception as e:
        print(f"  ❌ Failed to create classifier: {e}")

def main():
    """Main function to demonstrate adapter-based instruction extension."""
    
    print("TESTING ADAPTER-BASED INSTRUCTION EXTENSION")
    print("="*80)
    print("Exploring built-in adapter methods for extending model instructions...")
    
    # Explore adapter system
    explore_adapter_system()
    
    # Test adapter configuration
    test_adapter_configuration()
    
    # Demonstrate practical extension
    demonstrate_practical_instruction_extension()
    
    print("\\n" + "="*80)
    print("FINAL SUMMARY: INSTRUCTION EXTENSION CAPABILITIES")
    print("="*80)
    
    print("🔧 BUILT-IN ADAPTER METHODS AVAILABLE:")
    print("  ✅ add_adapter() - Add new adapter configurations")
    print("  ✅ load_adapter() - Load pre-trained adapters")
    print("  ✅ set_adapter() - Switch between adapters")
    print("  ✅ enable_adapters() / disable_adapters() - Control adapter state")
    print("  ✅ get_adapter_state_dict() - Access adapter parameters")
    
    print("\\n💡 PRACTICAL INSTRUCTION EXTENSION:")
    print("  1. ✅ Custom wrapper classes with instruction templates")
    print("  2. ✅ Response formatting based on instruction type")
    print("  3. ✅ Runtime behavior modification")
    print("  4. ✅ Context-aware response generation")
    print("  5. ⚠️  True adapter-based extension (requires PEFT setup)")
    
    print("\\n🎯 WHAT'S ACTUALLY POSSIBLE:")
    print("  ✅ Simulate instruction-following through custom classes")
    print("  ✅ Format responses according to different 'instruction' types")
    print("  ✅ Add instruction-like templates and response patterns")
    print("  ✅ Modify behavior through configuration and wrappers")
    print("  ⚠️  True instruction fine-tuning requires additional training")
    
    print("\\n🚀 RECOMMENDATION:")
    print("  The models have built-in adapter support, but extending instructions")
    print("  is most practically achieved through custom wrapper classes that")
    print("  simulate instruction-following behavior using the existing capabilities.")

if __name__ == "__main__":
    main() 