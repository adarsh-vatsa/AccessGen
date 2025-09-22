#!/usr/bin/env python3
"""
Comprehensive test suite for IAM policy generator
Tests 15 different scenarios across S3, EC2, and IAM services
Saves all outputs to experiments/policies and experiments/tests directories
"""

import subprocess
import json
import os
from datetime import datetime
from pathlib import Path

# Test scenarios covering different services and principal types
TEST_SCENARIOS = [
    # S3 Scenarios (5)
    {
        "id": "s3_public_read",
        "query": "allow public read access to website-content bucket",
        "services": ["s3"],
        "description": "Public S3 bucket for website hosting"
    },
    {
        "id": "s3_developer_upload", 
        "query": "developers need to upload and manage files in project-data bucket",
        "services": ["s3"],
        "description": "Developer team S3 access"
    },
    {
        "id": "s3_backup_versioning",
        "query": "backup service needs to store files with versioning in backup-bucket",
        "services": ["s3"],
        "description": "Backup service with versioning"
    },
    {
        "id": "s3_user_download",
        "query": "allow users to download files from shared-documents bucket",
        "services": ["s3"],
        "description": "User download access"
    },
    {
        "id": "s3_list_only",
        "query": "grant ability to list all S3 buckets without access to objects",
        "services": ["s3"],
        "description": "List buckets only"
    },
    
    # EC2 Scenarios (5)
    {
        "id": "ec2_admin_full",
        "query": "admin team needs full control over EC2 instances",
        "services": ["ec2"],
        "description": "Admin full EC2 access"
    },
    {
        "id": "ec2_dev_launch",
        "query": "developers can launch and terminate development instances",
        "services": ["ec2"],
        "description": "Developer instance management"
    },
    {
        "id": "ec2_monitoring",
        "query": "monitoring service needs to check EC2 instance status and metrics",
        "services": ["ec2"],
        "description": "Read-only monitoring access"
    },
    {
        "id": "ec2_start_stop",
        "query": "operators need to start and stop production instances",
        "services": ["ec2"],
        "description": "Start/stop operations only"
    },
    {
        "id": "ec2_describe_only",
        "query": "allow describing EC2 instances and their attributes",
        "services": ["ec2"],
        "description": "Read-only describe access"
    },
    
    # IAM Scenarios (3)
    {
        "id": "iam_user_mgmt",
        "query": "HR team needs to create and manage IAM users",
        "services": ["iam"],
        "description": "User management for HR"
    },
    {
        "id": "iam_role_creation",
        "query": "DevOps engineers need to create and attach roles for services",
        "services": ["iam"],
        "description": "Role management for DevOps"
    },
    {
        "id": "iam_readonly",
        "query": "auditors need to review all IAM policies and configurations",
        "services": ["iam"],
        "description": "Read-only audit access"
    },
    
    # Cross-Service Scenarios (2)
    {
        "id": "ec2_s3_instance",
        "query": "EC2 instances need to read configuration files from config-bucket and write logs to logs-bucket",
        "services": None,  # Let it auto-detect
        "description": "EC2 instance with S3 access"
    },
    {
        "id": "lambda_full_stack",
        "query": "Lambda functions need to access S3 data and manage EC2 instances",
        "services": None,  # Let it auto-detect
        "description": "Lambda with S3 and EC2 permissions"
    }
]

def run_policy_generator(scenario):
    """Run policy generator for a scenario and save outputs"""
    print(f"\nProcessing: {scenario['id']} - {scenario['description']}")
    print(f"Query: {scenario['query']}")
    print("-" * 60)
    
    # Build command - Note: without --no-save, it will save to experiments dir
    cmd = ["python", "../src/policy_generator.py", scenario["query"]]
    
    if scenario["services"]:
        cmd.extend(["--services"] + scenario["services"])
    
    try:
        # Run the generator
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"  ❌ Error: {result.stderr[:200]}")
            return False
        
        # Parse the output to verify generation
        output = result.stdout
        
        # Check if policy was generated
        if "GENERATED IAM POLICY" in output and "TEST CONFIGURATION" in output:
            print(f"  ✅ Successfully generated policy and test config")
            
            # Check for principals
            if "${PRINCIPAL_PLACEHOLDER}" in output:
                print(f"  📝 Uses principal placeholder (no principal specified)")
            elif '"*"' in output and "principals" in output:
                print(f"  🌍 Uses wildcard principal (public access)")
            elif "arn:aws:iam" in output:
                print(f"  👤 Uses specific IAM principal")
            elif ".amazonaws.com" in output:
                print(f"  🖥️  Uses service principal")
            
            # Check for saved files
            if "Saved files:" in output:
                files_section = output[output.find("Saved files:"):]
                print(f"  💾 Files saved to experiments directory")
            
            return True
        else:
            print(f"  ❌ Failed to generate outputs")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ Timeout - query took too long")
        return False
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return False

def verify_saved_files():
    """Verify that files were saved in experiments directory"""
    experiments_dir = Path("../experiments")
    policies_dir = experiments_dir / "policies"
    tests_dir = experiments_dir / "tests"
    
    print("\n" + "="*60)
    print("VERIFYING SAVED FILES")
    print("="*60)
    
    # Check directories exist
    if not policies_dir.exists():
        print(f"❌ Policies directory not found: {policies_dir}")
        return
    
    if not tests_dir.exists():
        print(f"❌ Tests directory not found: {tests_dir}")
        return
    
    # List saved files
    policy_files = list(policies_dir.glob("*.json"))
    test_files = list(tests_dir.glob("*.json"))
    
    print(f"\n📁 Policies directory: {policies_dir}")
    print(f"   Found {len(policy_files)} policy files")
    
    # Show recent policy files
    recent_policies = sorted(policy_files, key=lambda x: x.stat().st_mtime)[-5:]
    for pf in recent_policies:
        size = pf.stat().st_size
        print(f"   - {pf.name} ({size} bytes)")
    
    print(f"\n📁 Tests directory: {tests_dir}")
    print(f"   Found {len(test_files)} test config files")
    
    # Show recent test files
    recent_tests = sorted(test_files, key=lambda x: x.stat().st_mtime)[-5:]
    for tf in recent_tests:
        size = tf.stat().st_size
        print(f"   - {tf.name} ({size} bytes)")
    
    # Sample one file to show structure
    if recent_policies:
        sample_policy = recent_policies[-1]
        print(f"\n📄 Sample policy file: {sample_policy.name}")
        with open(sample_policy, 'r') as f:
            data = json.load(f)
            print(f"   Query: {data.get('query', 'N/A')[:60]}...")
            policy = data.get('policy', {})
            if policy.get('Statement'):
                print(f"   Statements: {len(policy['Statement'])}")
                first_stmt = policy['Statement'][0]
                actions = first_stmt.get('Action', [])
                if isinstance(actions, str):
                    actions = [actions]
                print(f"   First action: {actions[0] if actions else 'N/A'}")
    
    if recent_tests:
        sample_test = recent_tests[-1]
        print(f"\n📄 Sample test file: {sample_test.name}")
        with open(sample_test, 'r') as f:
            data = json.load(f)
            config = data.get('config', {})
            rules = config.get('rules', [])
            if rules:
                principals = rules[0].get('principals', [])
                print(f"   Service: {config.get('service', 'N/A')}")
                print(f"   Rules: {len(rules)}")
                print(f"   First principal: {principals[0] if principals else 'N/A'}")

def main():
    print("="*60)
    print("COMPREHENSIVE IAM POLICY GENERATOR TEST SUITE")
    print("="*60)
    print(f"Testing {len(TEST_SCENARIOS)} scenarios across S3, EC2, and IAM")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Track results
    results = {
        "success": [],
        "failed": []
    }
    
    # Run all test scenarios
    for scenario in TEST_SCENARIOS:
        success = run_policy_generator(scenario)
        if success:
            results["success"].append(scenario["id"])
        else:
            results["failed"].append(scenario["id"])
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Successful: {len(results['success'])}/{len(TEST_SCENARIOS)}")
    print(f"❌ Failed: {len(results['failed'])}/{len(TEST_SCENARIOS)}")
    
    if results["success"]:
        print(f"\nSuccessful scenarios:")
        for sid in results["success"]:
            scenario = next(s for s in TEST_SCENARIOS if s["id"] == sid)
            print(f"  ✅ {sid}: {scenario['description']}")
    
    if results["failed"]:
        print(f"\nFailed scenarios:")
        for sid in results["failed"]:
            scenario = next(s for s in TEST_SCENARIOS if s["id"] == sid)
            print(f"  ❌ {sid}: {scenario['description']}")
    
    # Verify saved files
    verify_saved_files()
    
    print("\n" + "="*60)
    print("Test suite complete!")
    print("Check experiments/policies and experiments/tests for all outputs")

if __name__ == "__main__":
    main()