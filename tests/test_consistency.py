#!/usr/bin/env python3
"""
Test script to verify consistency between IAM policy and test configuration
Ensures placeholders and values match across both outputs
"""

import subprocess
import json
import sys

def run_policy_generator(query, services=None):
    """Run policy generator and return both outputs"""
    cmd = ["python", "../src/policy_generator.py", query, "--no-save"]
    if services:
        cmd.extend(["--services"] + services)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout
    
    # Extract IAM policy and test config from output
    try:
        # Find JSON sections
        policy_start = output.find('{\n  "Version": "2012-10-17"')
        policy_end = output.find('\n}', policy_start) + 2
        policy_json = output[policy_start:policy_end]
        
        test_start = output.find('{\n  "service":', policy_end)
        test_end = output.find('\n}', test_start) + 2
        test_json = output[test_start:test_end]
        
        policy = json.loads(policy_json)
        test_config = json.loads(test_json)
        
        return policy, test_config
    except Exception as e:
        print(f"Error parsing output: {e}")
        print("Output:", output[:500])
        return None, None

def check_resource_consistency(policy, test_config):
    """Check if resources match between policy and test config"""
    policy_resources = set()
    for statement in policy.get("Statement", []):
        resources = statement.get("Resource", [])
        if isinstance(resources, str):
            resources = [resources]
        policy_resources.update(resources)
    
    test_resources = set()
    for rule in test_config.get("rules", []):
        test_resources.update(rule.get("resources", []))
    
    return policy_resources, test_resources

def check_action_consistency(policy, test_config):
    """Check if actions match between policy and test config"""
    policy_actions = set()
    for statement in policy.get("Statement", []):
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        policy_actions.update(actions)
    
    test_actions = set()
    for rule in test_config.get("rules", []):
        test_actions.update(rule.get("actions", []))
    
    return policy_actions, test_actions

# Test cases
test_cases = [
    {
        "query": "allow reading S3 objects",
        "services": ["s3"],
        "expected_placeholders": ["${BUCKET_NAME}", "${PRINCIPAL_PLACEHOLDER}"]
    },
    {
        "query": "developers need to upload files to data-bucket",
        "services": ["s3"],
        "expected_specific": ["data-bucket", "developers"]
    },
    {
        "query": "allow public read access to website bucket",
        "services": ["s3"],
        "expected_principal": "*"
    },
    {
        "query": "EC2 instances need to access S3",
        "services": None,  # Test multi-service
        "expected_principal": "ec2.amazonaws.com"
    }
]

def main():
    print("Testing IAM Policy and Test Configuration Consistency")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['query']}")
        print("-" * 40)
        
        policy, test_config = run_policy_generator(test["query"], test.get("services"))
        
        if not policy or not test_config:
            print("  ❌ Failed to generate outputs")
            continue
        
        # Check resource consistency
        policy_res, test_res = check_resource_consistency(policy, test_config)
        print(f"  Policy resources: {policy_res}")
        print(f"  Test resources:   {test_res}")
        
        if policy_res == test_res:
            print("  ✅ Resources match!")
        else:
            print("  ⚠️  Resource mismatch")
        
        # Check action consistency
        policy_acts, test_acts = check_action_consistency(policy, test_config)
        
        if policy_acts == test_acts:
            print("  ✅ Actions match!")
        else:
            print("  ⚠️  Actions differ (may be split across rules)")
            print(f"     Policy: {policy_acts}")
            print(f"     Test:   {test_acts}")
        
        # Check for expected placeholders
        if "expected_placeholders" in test:
            all_text = json.dumps(policy) + json.dumps(test_config)
            for placeholder in test["expected_placeholders"]:
                if placeholder in all_text:
                    print(f"  ✅ Found placeholder: {placeholder}")
                else:
                    print(f"  ❌ Missing placeholder: {placeholder}")
        
        # Check principal in test config
        if "expected_principal" in test:
            principals = []
            for rule in test_config.get("rules", []):
                principals.extend(rule.get("principals", []))
            
            if test["expected_principal"] in str(principals):
                print(f"  ✅ Correct principal: {test['expected_principal']}")
            else:
                print(f"  ❌ Wrong principal. Got: {principals}")

    print("\n" + "=" * 60)
    print("Consistency test complete!")


if __name__ == "__main__":
    main()
