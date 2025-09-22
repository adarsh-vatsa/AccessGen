#!/usr/bin/env python3
"""
Run remaining test scenarios (EC2, IAM, and cross-service)
Skips S3 scenarios that were already tested
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path

# Remaining test scenarios (skipping already tested S3 ones)
REMAINING_SCENARIOS = [
    # S3 - Only the one not yet tested
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
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario['id']}")
    print(f"Description: {scenario['description']}")
    print(f"Query: {scenario['query']}")
    print("-" * 60)
    
    # Build command - without --no-save to save files
    cmd = ["python", "../src/policy_generator.py", scenario["query"]]
    
    if scenario["services"]:
        cmd.extend(["--services"] + scenario["services"])
    
    try:
        # Run the generator
        print(f"⏳ Generating policy and test configuration...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ Error: {result.stderr[:200]}")
            return False
        
        output = result.stdout
        
        # Extract key information
        if "GENERATED IAM POLICY" in output and "TEST CONFIGURATION" in output:
            print(f"✅ Successfully generated outputs")
            
            # Check principal type
            if "${PRINCIPAL_PLACEHOLDER}" in output:
                print(f"📝 Principal: Placeholder (unspecified)")
            elif '"*"' in output and "principals" in output:
                print(f"🌍 Principal: Wildcard (public/all)")
            elif "arn:aws:iam::${ACCOUNT_ID}:role/" in output:
                print(f"👤 Principal: IAM Role")
            elif "arn:aws:iam::${ACCOUNT_ID}:user/" in output:
                print(f"👤 Principal: IAM User")
            elif ".amazonaws.com" in output:
                print(f"🖥️  Principal: Service principal")
            
            # Extract actions count
            test_start = output.find('"service":')
            if test_start > 0:
                test_section = output[test_start:]
                action_count = test_section.count('"actions"')
                print(f"📋 Rules generated: {action_count}")
            
            # Check if files were saved
            if "Saved files:" in output:
                print(f"💾 Files saved to experiments directory")
            
            return True
        else:
            print(f"❌ Failed to generate outputs")
            print(f"Output preview: {output[:300]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def main():
    print("="*60)
    print("REMAINING TEST SCENARIOS")
    print("="*60)
    print(f"Running {len(REMAINING_SCENARIOS)} additional scenarios")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Track results
    success_count = 0
    failed_scenarios = []
    
    # Run each scenario
    for i, scenario in enumerate(REMAINING_SCENARIOS, 1):
        print(f"\nProgress: {i}/{len(REMAINING_SCENARIOS)}")
        
        if run_policy_generator(scenario):
            success_count += 1
        else:
            failed_scenarios.append(scenario["id"])
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}/{len(REMAINING_SCENARIOS)}")
    print(f"❌ Failed: {len(failed_scenarios)}/{len(REMAINING_SCENARIOS)}")
    
    if failed_scenarios:
        print(f"\nFailed scenarios:")
        for sid in failed_scenarios:
            print(f"  - {sid}")
    
    # Check final file count
    experiments_dir = Path("../experiments")
    policy_count = len(list((experiments_dir / "policies").glob("*.json")))
    test_count = len(list((experiments_dir / "tests").glob("*.json")))
    
    print(f"\n📁 Total files in experiments:")
    print(f"   Policies: {policy_count}")
    print(f"   Tests: {test_count}")
    
    print("\nTest suite complete! Check experiments/ for all generated files.")

if __name__ == "__main__":
    main()