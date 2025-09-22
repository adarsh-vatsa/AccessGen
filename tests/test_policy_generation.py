#!/usr/bin/env python3
"""
Test script for IAM Policy Generator
Demonstrates various use cases and capabilities
"""
import json
import sys
import os
import pytest
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.policy_generator import IAMPolicyGeneratorV2 as IAMPolicyGenerator

# Skip these tests if Gemini API key is not present (LLM-dependent)
pytestmark = pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="Requires GEMINI_API_KEY for LLM generation")

def test_basic_scenarios():
    """Test various common S3 permission scenarios"""
    
    # Initialize generator with lower threshold for testing
    generator = IAMPolicyGenerator(
        score_threshold=0.0005,  # Lower threshold to capture more actions
        max_actions=20,  # Allow more actions for complex scenarios
        model="models/gemini-2.5-pro"
    )
    
    test_cases = [
        {
            "name": "Basic Read/Write",
            "query": "I need to upload files to S3 buckets and read them back"
        },
        {
            "name": "Static Website Hosting",
            "query": "I want to host a static website on S3 with public read access to objects"
        },
        {
            "name": "Backup Management",
            "query": "I need to create backups in S3, manage versioning, and restore objects when needed"
        },
        {
            "name": "Data Analytics",
            "query": "I need to read large datasets from S3 for analytics, list objects with specific prefixes, and get object metadata"
        },
        {
            "name": "Multipart Upload",
            "query": "I need to upload very large files using multipart upload and abort incomplete uploads"
        },
        {
            "name": "Cross-Region Replication",
            "query": "I want to set up cross-region replication for my S3 buckets"
        },
        {
            "name": "Lifecycle Management",
            "query": "I need to configure lifecycle policies to automatically transition objects to different storage classes and delete old objects"
        },
        {
            "name": "Access Control",
            "query": "I need to manage bucket policies, ACLs, and configure CORS for web applications"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_case['name']}")
        print(f"{'='*70}")
        print(f"Query: {test_case['query']}")
        print("-" * 70)
        
        try:
            # Generate policy
            result = generator.generate_policy(test_case['query'])
            
            if result["status"] == "success":
                # Validate the policy
                is_valid, errors = generator.validate_policy(result["iam_policy"])
                
                print("\nGenerated Policy:")
                print(json.dumps(result["iam_policy"], indent=2))
                
                print("\nTop Actions Considered:")
                for action in result["metadata"]["top_actions"][:5]:
                    status = "✓ INCLUDED" if action["included"] else "✗ EXCLUDED"
                    print(f"  {status}: {action['action']} (score: {action['score']:.4f})")
                
                print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
                if not is_valid:
                    print("Errors:", errors)
                
                # Save result
                result["test_case"] = test_case["name"]
                result["validation"] = {"valid": is_valid, "errors": errors if not is_valid else []}
                results.append(result)
                
            else:
                print(f"ERROR: {result['message']}")
                results.append(result)
                
        except Exception as e:
            print(f"Exception during test: {e}")
            results.append({
                "test_case": test_case["name"],
                "status": "error",
                "message": str(e)
            })
    
    # Save all results
    output_file = Path("test_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Test Results Summary")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r.get("status") == "success")
    valid = sum(1 for r in results if r.get("status") == "success" and r.get("validation", {}).get("valid", False))
    
    print(f"Total test cases: {len(test_cases)}")
    print(f"Successful generations: {successful}/{len(test_cases)}")
    print(f"Valid policies: {valid}/{successful}")
    print(f"\nDetailed results saved to: {output_file}")
    
    return results


def test_edge_cases():
    """Test edge cases and error handling"""
    
    generator = IAMPolicyGenerator()
    
    edge_cases = [
        {
            "name": "Ambiguous Query",
            "query": "I need some S3 permissions"
        },
        {
            "name": "Highly Specific",
            "query": "I need exactly PutObject and GetObject permissions for objects with the prefix 'logs/' in my bucket"
        },
        {
            "name": "Security-Focused",
            "query": "I need to audit S3 access, monitor object access patterns, and ensure encryption"
        }
    ]
    
    print("\n" + "="*70)
    print("EDGE CASE TESTING")
    print("="*70)
    
    for test_case in edge_cases:
        print(f"\nTest: {test_case['name']}")
        print(f"Query: {test_case['query']}")
        print("-" * 40)
        
        result = generator.generate_policy(test_case['query'])
        
        if result["status"] == "success":
            print("✓ Policy generated successfully")
            actions = []
            for stmt in result["iam_policy"].get("Statement", []):
                if isinstance(stmt.get("Action"), list):
                    actions.extend(stmt["Action"])
                elif stmt.get("Action"):
                    actions.append(stmt["Action"])
            print(f"  Actions included: {', '.join(actions[:5])}")
            if len(actions) > 5:
                print(f"  ... and {len(actions) - 5} more")
        else:
            print(f"✗ Generation failed: {result.get('message', 'Unknown error')}")


def interactive_mode():
    """Interactive mode for testing custom queries"""
    
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Enter your S3 permission requirements in natural language.")
    print("Type 'quit' to exit.\n")
    
    generator = IAMPolicyGenerator()
    
    while True:
        query = input("Your requirement: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        print("\nGenerating policy...")
        result = generator.generate_policy(query)
        
        if result["status"] == "success":
            print("\n" + "-"*40)
            print("GENERATED POLICY:")
            print("-"*40)
            print(json.dumps(result["iam_policy"], indent=2))
            
            if "explanation" in result:
                print("\n" + "-"*40)
                print("EXPLANATION:")
                print("-"*40)
                print(result["explanation"])
            
            # Offer to save
            save = input("\nSave this policy? (y/n): ").strip().lower()
            if save == 'y':
                filename = input("Filename (without .json): ").strip()
                if filename:
                    with open(f"{filename}.json", "w") as f:
                        json.dump(result["iam_policy"], f, indent=2)
                    print(f"Policy saved to {filename}.json")
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IAM Policy Generator")
    parser.add_argument("--mode", choices=["basic", "edge", "interactive", "all"], 
                       default="basic", help="Test mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "basic" or args.mode == "all":
        test_basic_scenarios()
    
    if args.mode == "edge" or args.mode == "all":
        test_edge_cases()
    
    if args.mode == "interactive":
        interactive_mode()
