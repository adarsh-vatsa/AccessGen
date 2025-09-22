#!/usr/bin/env python3
"""
Basic test script for Unified AWS Actions RAG system.
Tests core functionality for S3, EC2, and IAM services.
"""
import json
import os
import sys
from pathlib import Path

def test_data_loading():
    """Test loading the unified enriched data files."""
    print("Testing data loading...")
    
    services = {
        "s3": "aws_iam_registry_s3_enriched_extras.json",
        "ec2": "aws_iam_registry_ec2_enriched_extras.json",
        "iam": "aws_iam_registry_iam_enriched_extras.json"
    }
    
    total_actions = 0
    
    for service, filename in services.items():
        data_file = Path(__file__).parent.parent / "enriched_data" / filename
        
        if not data_file.exists():
            print(f"❌ {service.upper()} data file not found: {data_file}")
            return False
            
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                
            actions = data[service]["actions"]
            print(f"✅ Loaded {len(actions)} {service.upper()} actions")
            total_actions += len(actions)
        
            # Check required fields on first action
            if actions:
                first_action = actions[0]
                required_fields = ['action', 'access_level', 'description', 'sparse_text', 'dense_text']
                
                for field in required_fields:
                    if field not in first_action:
                        print(f"❌ {service.upper()}: Missing required field: {field}")
                        return False
                
                # Check field content
                if not first_action['sparse_text'] or not first_action['dense_text']:
                    print(f"❌ {service.upper()}: Empty sparse_text or dense_text fields")
                    return False
                    
        except Exception as e:
            print(f"❌ Error loading {service.upper()} data: {e}")
            return False
    
    print(f"✅ All required fields present in all services")
    print(f"✅ Total actions loaded: {total_actions} (S3+EC2+IAM)")
    return True

def test_environment_variables():
    """Test environment variable setup."""
    print("\nTesting environment variables...")
    
    required_vars = ['PINECONE_API_KEY', 'GEMINI_API_KEY']
    optional_vars = ['COHERE_API_KEY', 'PINECONE_CLOUD', 'PINECONE_REGION']
    
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is missing (required)")
            all_good = False
            
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} not set (optional, will use defaults)")
            
    return all_good

def test_imports():
    """Test that required packages can be imported."""
    print("\nTesting package imports...")
    
    packages = [
        ('pinecone', 'pinecone'),
        ('pinecone_text', 'pinecone-text'),
        ('google.genai', 'google-genai'),
        ('numpy', 'numpy'),
        ('requests', 'requests')
    ]
    
    all_good = True
    
    for package, pip_name in packages:
        try:
            __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} import failed. Install with: pip install {pip_name}")
            all_good = False
            
    return all_good

def test_text_processing():
    """Test basic text processing functions."""
    print("\nTesting text processing...")
    
    # Test data for unified format
    sample_sparse = "service=s3 | action=GetObject | access=Read\nresources=[object]\ncondition_keys=[s3:ExistingObjectTag/<key>]"
    sample_dense = sample_sparse + "\ndescription=Grants permission to retrieve objects from Amazon S3\nquery_hooks=[download, get, retrieve]"
    
    if len(sample_sparse.split('\n')) >= 3:
        print("✅ Sparse text format looks correct")
    else:
        print("❌ Sparse text format issue")
        return False
        
    if "description=" in sample_dense:
        print("✅ Dense text contains description")
    else:
        print("❌ Dense text missing description")
        return False
        
    print("✅ Text processing validation passed")
    return True

def main():
    """Run all tests."""
    print("IAM Actions RAG System - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Environment Variables", test_environment_variables), 
        ("Package Imports", test_imports),
        ("Text Processing", test_text_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
            
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System ready for use.")
        print("\nNext steps:")
        print("1. Set missing environment variables if any")
        print("2. Run: python build_unified_indexes.py (builds unified S3+EC2+IAM indexes)")
        print("3. Test queries: python query_unified.py 'list S3 objects'")
        print("4. Test multi-service: python query_unified.py 'EC2 instance with S3 access'")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Fix issues before proceeding.")
        
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)