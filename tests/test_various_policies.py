#!/usr/bin/env python3
"""
Test various S3 policy generation scenarios
"""
import json
import subprocess
import sys
from pathlib import Path

def run_one_policy(query: str, description: str):
    """Test a single policy generation"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"{'='*80}")
    print(f"Query: {query}")
    print("-" * 80)
    
    # Run policy generator
    cmd = [
        sys.executable,
        str(Path(__file__).parent.parent / "src" / "policy_generator.py"),
        query,
        "--no-save"  # Don't save to avoid clutter during testing
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse output to extract policy
        output_lines = result.stdout.split('\n')
        
        # Find policy section
        policy_start = None
        policy_end = None
        for i, line in enumerate(output_lines):
            if "GENERATED IAM POLICY" in line:
                policy_start = i + 2
            elif policy_start and "TEST CONFIGURATION" in line:
                policy_end = i - 2
                break
        
        if policy_start and policy_end:
            policy_json = '\n'.join(output_lines[policy_start:policy_end])
            try:
                policy = json.loads(policy_json)
                
                # Extract actions from policy
                actions = []
                for statement in policy.get("Statement", []):
                    stmt_actions = statement.get("Action", [])
                    if isinstance(stmt_actions, str):
                        stmt_actions = [stmt_actions]
                    actions.extend(stmt_actions)
                
                print(f"✓ Policy generated successfully")
                print(f"  Actions included: {', '.join(sorted(set(actions)))}")
                
                # Check for Allow/Deny statements
                effects = [stmt.get("Effect") for stmt in policy.get("Statement", [])]
                if "Deny" in effects:
                    print(f"  Contains DENY statements: Yes")
                
                # Check if expanded query was used
                if "Expanded query:" in result.stderr:
                    for line in result.stderr.split('\n'):
                        if "Expanded query:" in line:
                            print(f"  {line.split('INFO - ')[1] if 'INFO - ' in line else line}")
                
                return True, policy
                
            except json.JSONDecodeError:
                print(f"✗ Failed to parse policy JSON")
                return False, None
        else:
            print(f"✗ Could not find policy in output")
            if "error" in result.stdout.lower() or "error" in result.stderr.lower():
                print(f"  Error detected in output")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 60 seconds")
        return False, None
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False, None


def main():
    """Test various policy scenarios"""
    
    test_cases = [
        # Basic Operations
        ("I need to create and delete S3 buckets",
         "Bucket Management"),
        
        ("I want to host a static website on S3",
         "Static Website Hosting"),
        
        ("I need to manage object tags and metadata",
         "Object Tagging and Metadata"),
        
        # Advanced Features
        ("I need to set up cross-region replication between S3 buckets",
         "Cross-Region Replication"),
        
        ("I want to manage S3 lifecycle policies and intelligent tiering",
         "Lifecycle and Intelligent Tiering"),
        
        ("I need to work with S3 multipart uploads for large files",
         "Multipart Upload Management"),
        
        # Security & Compliance
        ("I need to encrypt objects and manage encryption keys in S3",
         "Encryption Management"),
        
        ("I want to configure S3 access logging and CloudTrail integration",
         "Logging and Auditing"),
        
        ("I need to manage S3 access points and VPC endpoints",
         "Access Points and VPC"),
        
        # Data Operations
        ("I need to run S3 Select queries on objects and analyze data",
         "S3 Select and Analytics"),
        
        ("I want to restore objects from Glacier and manage archive tiers",
         "Glacier and Archive Management"),
        
        ("I need to manage S3 inventory reports and analytics configurations",
         "Inventory and Analytics"),
        
        # Complex Scenarios
        ("I need full S3 admin access but prevent deletion of production buckets",
         "Admin with Deletion Protection"),
        
        ("I want to manage S3 batch operations and jobs",
         "Batch Operations"),
        
        ("I need to configure CORS, bucket policies, and public access blocks",
         "Bucket Security Configuration"),
    ]
    
    # Track results
    successful = 0
    failed = 0
    policies = {}
    
    for query, description in test_cases:
        success, policy = run_one_policy(query, description)
        if success:
            successful += 1
            policies[description] = policy
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total tests: {len(test_cases)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(test_cases)*100:.1f}%")
    
    # Save all successful policies
    if policies:
        output_file = Path("test_results_various_policies.json")
        with open(output_file, "w") as f:
            json.dump(policies, f, indent=2)
        print(f"\nPolicies saved to: {output_file}")


if __name__ == "__main__":
    main()
