#!/usr/bin/env python3
"""
Natural Language to IAM Policy Generator v2
Generates both IAM policies and testing format for fuzzer validation
"""
import json
import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import re
import hashlib

# Add pinecone directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pinecone"))

from query_unified import UnifiedQueryEngine
try:
    from src.guards.registry_guard import RegistryGuard
except Exception:
    try:
        from guards.registry_guard import RegistryGuard  # fallback when running as module
    except Exception:
        RegistryGuard = None  # type: ignore
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    import google.generativeai as genai
    types = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class IAMPolicyGeneratorV2:
    """Generates IAM policies and testing configurations from natural language queries"""
    
    def __init__(self, 
                 data_path: str = None,
                 score_threshold: float = 0.0002,
                 max_actions: int = 15,
                 model: str = "models/gemini-2.5-pro",
                 experiments_dir: str = None,
                 use_query_expansion: bool = True,
                 target_services: List[str] = None,
                 use_vector_search: bool = True):
        """
        Initialize the policy generator
        
        Args:
            data_path: Path to enriched actions data (unified or service-specific)
            score_threshold: Minimum relevance score to include an action
            max_actions: Maximum number of actions to send to LLM
            model: Gemini model to use for generation
            experiments_dir: Directory to save experiments (policies and tests)
            use_query_expansion: Whether to use service router for query expansion
            target_services: List of services to target ['s3', 'ec2', 'iam'] or None for auto-detect
        """
        # Initialize unified query engine with optional service router
        self.use_vector_search = use_vector_search
        self.query_engine = None
        if self.use_vector_search:
            self.query_engine = UnifiedQueryEngine(use_service_router=use_query_expansion)
        self.score_threshold = score_threshold
        self.max_actions = max_actions
        self.model = model
        self.use_query_expansion = use_query_expansion
        self.target_services = target_services
        
        # Set up experiments directory (keep it at parent level)
        self.experiments_dir = Path(experiments_dir or Path(__file__).parent.parent / "experiments")
        self.policies_dir = self.experiments_dir / "policies"
        self.tests_dir = self.experiments_dir / "tests"
        
        # Create directories if they don't exist
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment")
            sys.exit(1)
        
        if types:  # Using google.genai
            self.genai_client = genai.Client(api_key=api_key)
        else:  # Using google.generativeai
            genai.configure(api_key=api_key)
            self.genai_client = None
        
        # Load unified actions data (contains S3, EC2, and IAM)
        # The unified engine loads all three service files by default
        if self.query_engine is not None:
            self.query_engine.load_unified_actions_data()
            logger.info(f"Loaded unified actions data (S3, EC2, IAM)")
    
    def search_relevant_actions(self, query: str, top_k: int = 30) -> tuple[List[Dict[str, Any]], str]:
        """
        Search for relevant AWS actions based on natural language query
        
        Args:
            query: Natural language description of required permissions
            top_k: Number of results to retrieve initially
            
        Returns:
            Tuple of (List of relevant actions with scores, search query used)
        """
        # The unified engine handles query expansion via service router if enabled
        logger.info(f"Searching for actions: {query}")
        
        # Use target services if specified, otherwise let router auto-detect
        results = self.query_engine.unified_search(
            query_text=query,
            services=self.target_services,
            top_m=top_k,
            use_query_expansion=self.use_query_expansion
        )
        
        # Log expanded query if it was expanded
        # (The expansion happens inside unified_search now)
        search_query = query  # Keep original for compatibility
        
        # Filter by score threshold and limit to max_actions
        filtered = []
        for r in results:
            if r['rerank_score'] >= self.score_threshold:
                filtered.append(r)
                if len(filtered) >= self.max_actions:
                    break
        
        logger.info(f"Found {len(filtered)} relevant actions above threshold {self.score_threshold}")
        return filtered, search_query
    
    def _format_action_context(self, actions: List[Dict[str, Any]], include_scores: bool = False) -> str:
        """Format action search results for LLM context"""
        lines = []
        for i, action in enumerate(actions, 1):
            # Handle both S3-only and multi-service formats
            service = action.get('service', 's3')
            lines.append(f"{i}. Action: {service}:{action['action']}")
            if include_scores:
                lines.append(f"   Relevance Score: {action['rerank_score']:.4f}")
            lines.append(f"   Description: {action['description']}")
            lines.append(f"   Access Level: {action['access_level']}")
            
            if action['resource_types']:
                res_types = ", ".join(action['resource_types'])
                lines.append(f"   Resource Types: {res_types}")
            
            if action['condition_keys']:
                keys = ", ".join(action['condition_keys'][:5])
                if len(action['condition_keys']) > 5:
                    keys += ", ..."
                lines.append(f"   Condition Keys: {keys}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM"""
        return """You are an AWS IAM policy expert. Your task is to generate TWO outputs:
1. An accurate, minimal, and secure IAM policy.
2. A testing configuration for fuzzer-based validation.

Key principles for IAM policy generation:
1. CRITICALLY EVALUATE each action's description against the user's actual requirements.
2. ONLY include actions that directly fulfill the stated need.

3. **APPLY RESTRICTIVE CONDITIONS (MOST IMPORTANT):** Once an action is selected, determine if it can be restricted further. The goal is to limit HOW, WHEN, and WHERE an action can be used.
   - For actions that attach policies (`iam:AttachUserPolicy`), ALWAYS add a `Condition` to restrict which policies can be attached using the `iam:PolicyARN` key.
   - For actions that create resources, consider using `Condition` blocks to enforce tagging (`aws:RequestTag`) or other parameters.
   - For S3, use conditions like `s3:x-amz-acl` to control object ACLs on upload.

4. **USE ADVANCED PATTERNS FOR DELEGATION:** For requests involving one user/role managing another (e.g., "HR manages users," "CI/CD role manages EC2"), the most secure pattern is to enforce a `PermissionsBoundary`.
   - The managing entity should have permission to `iam:CreateUser` and `iam:PutUserPermissionsBoundary`.
   - The `iam:CreateUser` action MUST have a `Condition` forcing the attachment of a specific permissions boundary ARN.

5. **USE ORGANIZATIONAL PATHS IN ARNS:** Encourage the use of paths in ARNs to segregate resources (e.g., "arn:aws:iam::*:user/employees/*", "arn:aws:s3:::my-bucket/project-alpha/*").

6. Follow the principle of least privilege - exclude anything not explicitly needed.
7. Use appropriate resource ARNs (bucket-level vs object-level).
8. Generate valid JSON that conforms to IAM policy schema.

Resource ARN patterns:
- Bucket operations: "arn:aws:s3:::bucket-name" or "arn:aws:s3:::${BUCKET_NAME}"
- Object operations: "arn:aws:s3:::bucket-name/*" or "arn:aws:s3:::${BUCKET_NAME}/*"
- EC2 operations: "arn:aws:ec2:${REGION}:${ACCOUNT_ID}:instance/*"
- IAM operations: "arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
- IAM user paths: "arn:aws:iam::${ACCOUNT_ID}:user/path/${USER_NAME}"

PRINCIPAL RULES FOR TEST CONFIGURATION:
Be precise about principals based on what the user specifies:
1. If user explicitly says "all", "anyone", "everyone", "public":
   → Use: ["*"]
2. If user specifies a concrete principal (e.g., "developers", "admin role", "EC2 instances"):
   → Use appropriate ARN: ["arn:aws:iam::${ACCOUNT_ID}:role/developers"]
   → For service principals: ["ec2.amazonaws.com"]
3. If NO principal is mentioned at all in the query:
   → Use: ["${PRINCIPAL_PLACEHOLDER}"]
4. If user mentions generic "users" without specifics:
   → Use: ["arn:aws:iam::${ACCOUNT_ID}:user/${USER_PLACEHOLDER}"]
5. If user mentions generic "roles" without specifics:
   → Use: ["arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_PLACEHOLDER}"]

OUTPUT FORMAT:
You must output a JSON object with exactly two fields:
{
  "iam_policy": { ... the IAM policy ... },
  "test_config": { ... the testing configuration ... }
}

The test_config should follow this format:
{
  "service": "s3" or "ec2" or "iam",
  "rules": [
    {
      "id": "R1",
      "effect": "Allow" or "Deny",
      "principals": [...array following the PRINCIPAL RULES above...],
      "not_principals": [] if not used,
      "actions": [...list of service:Action...],
      "resources": [...list of resource ARNs with placeholders if needed...],
      "conditions": {} or specific conditions from the policy
    }
  ]
}

IMPORTANT: Use ${PLACEHOLDER} syntax for values that need substitution.
Each statement in the IAM policy should become a rule in test_config.
Output ONLY the JSON object, no explanations or markdown formatting."""
    
    def _build_raw_user_prompt(self, query: str) -> str:
        """Build a user prompt for RAW mode (no vector search context)"""
        return f"""Natural Language Requirement:
{query}

CRITICAL INSTRUCTIONS:
1. Generate a minimal, least-privilege IAM policy that directly satisfies the requirement.
2. Prefer specific actions over wildcards; include only what is necessary.
3. Use appropriate resource ARNs and placeholders (e.g., ${{BUCKET_NAME}}, ${{ACCOUNT_ID}}).
4. If multiple AWS services are implied, include only those needed.
5. Output both an IAM policy and a matching test configuration.

Generate the output JSON with both 'iam_policy' and 'test_config' fields:"""
    
    def _generate_filename(self, query: str) -> str:
        """Generate a descriptive filename from the query"""
        # Create a short hash of the query for uniqueness
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        # Clean up the query for filename
        clean_query = re.sub(r'[^\w\s-]', '', query.lower())
        clean_query = re.sub(r'[-\s]+', '_', clean_query)
        
        # Truncate if too long
        if len(clean_query) > 50:
            clean_query = clean_query[:50]
        
        # Add timestamp and hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{clean_query}_{timestamp}_{query_hash}"
    
    def _build_user_prompt(self, query: str, actions: List[Dict[str, Any]]) -> str:
        """Build the user prompt with query and action context"""
        action_context = self._format_action_context(actions)
        
        # Determine the primary service from actions
        services = set()
        for action in actions:
            services.add(action.get('service', 's3'))
        
        service_text = "/".join(sorted(services)).upper()
        
        return f"""Natural Language Requirement:
{query}

Available {service_text} Actions (from authoritative AWS documentation search):
{action_context}

CRITICAL INSTRUCTIONS:
1. You MUST ONLY use actions from the list above - DO NOT add any actions from your training data
2. The actions above are from the official AWS documentation and are the ONLY valid options
3. Select the minimum subset of actions that fulfill the requirement
4. If the requirement cannot be fully met with the provided actions, use what's available and note limitations
5. Generate both an IAM policy and test configuration using ONLY these actions

Generate the output JSON with both 'iam_policy' and 'test_config' fields:"""
    
    def generate_policy(self, query: str, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Generate an IAM policy and test configuration from natural language query
        
        Args:
            query: Natural language description of required permissions
            save_to_file: Whether to save outputs to files
            
        Returns:
            Dictionary containing the generated policy, test config, and metadata
        """
        # Step 1: Either search for relevant actions (vector mode) or run RAW mode
        if self.use_vector_search:
            actions, search_query = self.search_relevant_actions(query)
            if not actions:
                logger.warning("No relevant actions found for query")
                return {
                    "status": "error",
                    "message": "No relevant AWS actions found for the given query",
                    "query": query
                }
        else:
            actions = []
            search_query = query
        
        # Step 2: Generate policy and test config using LLM
        try:
            system_prompt = self._build_system_prompt()
            if self.use_vector_search:
                user_prompt = self._build_user_prompt(query, actions)
            else:
                user_prompt = self._build_raw_user_prompt(query)
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            if types:  # Using google.genai
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,  # Lower temperature for more consistent policies
                        max_output_tokens=4000,  # Increased for dual output
                    )
                )
            else:  # Using google.generativeai
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=4000,
                    )
                )
            
            output_text = response.text.strip()
            
            # Clean up any markdown formatting if present
            output_text = re.sub(r'^```json\s*', '', output_text)
            output_text = re.sub(r'\s*```$', '', output_text)
            
            # Parse the generated output
            try:
                output = json.loads(output_text)
                
                if "iam_policy" not in output or "test_config" not in output:
                    logger.error("Generated output missing required fields")
                    return {
                        "status": "error",
                        "message": "Generated output missing 'iam_policy' or 'test_config'",
                        "raw_output": output_text,
                        "query": query
                    }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse generated output: {e}")
                return {
                    "status": "error",
                    "message": "Generated output is not valid JSON",
                    "raw_output": output_text,
                    "query": query
                }
            
            # Step 3: Prepare response
            result = {
                "status": "success",
                "query": query,
                "iam_policy": output["iam_policy"],
                "test_config": output["test_config"],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model,
                    "original_query": query,
                    "search_query": (search_query if (self.use_vector_search and search_query != query) else None),
                    "actions_searched": (len(actions) if self.use_vector_search else 0),
                    "score_threshold": self.score_threshold,
                    "top_actions": ([
                        {
                            "action": f"{a.get('service', 's3')}:{a['action']}",
                            "score": a['rerank_score'],
                            "included": self._is_action_in_policy(f"{a.get('service', 's3')}:{a['action']}", output["iam_policy"])
                        }
                        for a in actions[:10]
                    ] if self.use_vector_search else [])
                }
            }

            # Add registry provenance for all included actions (all modes)
            try:
                if RegistryGuard is not None:
                    guard = RegistryGuard()
                    gres = guard.guard(output["iam_policy"])  # type: ignore
                    result["metadata"]["registry"] = {
                        "mismatches": gres.get("out_of_registry", []),
                        "replacements": gres.get("replacements", {}),
                        "provenance": gres.get("provenance", {}),
                    }
            except Exception as _e:
                # Non-fatal; provenance unavailable
                pass
            
            # Step 4: Save to files if requested
            if save_to_file:
                filename_base = self._generate_filename(query)
                
                # Save IAM policy
                policy_file = self.policies_dir / f"{filename_base}.json"
                with open(policy_file, "w") as f:
                    json.dump({
                        "query": query,
                        "timestamp": result["metadata"]["timestamp"],
                        "policy": output["iam_policy"]
                    }, f, indent=2)
                
                # Save test configuration
                test_file = self.tests_dir / f"{filename_base}.json"
                with open(test_file, "w") as f:
                    json.dump({
                        "query": query,
                        "timestamp": result["metadata"]["timestamp"],
                        "config": output["test_config"]
                    }, f, indent=2)
                
                result["files"] = {
                    "policy": str(policy_file),
                    "test": str(test_file)
                }
                
                logger.info(f"Saved policy to: {policy_file}")
                logger.info(f"Saved test config to: {test_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating policy: {e}")
            return {
                "status": "error",
                "message": str(e),
                "query": query
            }
    
    def _is_action_in_policy(self, action: str, policy: Dict) -> bool:
        """Check if an action is included in the generated policy"""
        # Handle both formats: "service:action" or just "action"
        if ":" in action:
            full_action = action
        else:
            # Default to s3 for backward compatibility
            full_action = f"s3:{action}"
        
        if "Statement" not in policy:
            return False
        
        for statement in policy["Statement"]:
            if "Action" in statement:
                actions = statement["Action"]
                if isinstance(actions, str):
                    actions = [actions]
                if full_action in actions:
                    return True
        
        return False
    
    def validate_policy(self, policy: Dict) -> Tuple[bool, List[str]]:
        """Basic validation of generated IAM policy structure"""
        errors = []
        
        # Check required top-level fields
        if "Version" not in policy:
            errors.append("Missing 'Version' field")
        elif policy["Version"] not in ["2012-10-17", "2008-10-17"]:
            errors.append(f"Invalid Version: {policy['Version']}")
        
        if "Statement" not in policy:
            errors.append("Missing 'Statement' field")
            return False, errors
        
        if not isinstance(policy["Statement"], list):
            errors.append("'Statement' must be a list")
            return False, errors
        
        # Validate each statement
        for i, stmt in enumerate(policy["Statement"]):
            if "Effect" not in stmt:
                errors.append(f"Statement {i}: Missing 'Effect' field")
            elif stmt["Effect"] not in ["Allow", "Deny"]:
                errors.append(f"Statement {i}: Invalid Effect: {stmt['Effect']}")
            
            if "Action" not in stmt:
                errors.append(f"Statement {i}: Missing 'Action' field")
            
            if "Resource" not in stmt:
                errors.append(f"Statement {i}: Missing 'Resource' field")
            
            # Validate actions start with s3:
            if "Action" in stmt:
                actions = stmt["Action"]
                if isinstance(actions, str):
                    actions = [actions]
                for action in actions:
                    if not action.startswith("s3:"):
                        errors.append(f"Statement {i}: Invalid action prefix: {action}")
        
        return len(errors) == 0, errors
    
    def validate_test_config(self, config: Dict) -> Tuple[bool, List[str]]:
        """Validate the test configuration structure"""
        errors = []
        
        if "service" not in config:
            errors.append("Missing 'service' field")
        elif config["service"] not in ["s3", "ec2", "iam"]:
            errors.append(f"Invalid service: {config['service']} (expected 's3', 'ec2', or 'iam')")
        
        if "rules" not in config:
            errors.append("Missing 'rules' field")
            return False, errors
        
        if not isinstance(config["rules"], list):
            errors.append("'rules' must be a list")
            return False, errors
        
        for i, rule in enumerate(config["rules"]):
            if "id" not in rule:
                errors.append(f"Rule {i}: Missing 'id' field")
            
            if "effect" not in rule:
                errors.append(f"Rule {i}: Missing 'effect' field")
            elif rule["effect"] not in ["Allow", "Deny"]:
                errors.append(f"Rule {i}: Invalid effect: {rule['effect']}")
            
            if "actions" not in rule:
                errors.append(f"Rule {i}: Missing 'actions' field")
            
            if "resources" not in rule:
                errors.append(f"Rule {i}: Missing 'resources' field")
        
        return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="Generate IAM policies and test configs from natural language")
    parser.add_argument("query", help="Natural language description of required permissions")
    parser.add_argument("--services", nargs="+", choices=["s3", "ec2", "iam"], 
                       help="Target specific AWS services (default: auto-detect)")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Minimum score threshold")
    parser.add_argument("--max-actions", type=int, default=15, help="Maximum actions to consider")
    parser.add_argument("--model", default="models/gemini-2.5-pro", help="Gemini model to use")
    parser.add_argument("--no-save", action="store_true", help="Don't save to files")
    parser.add_argument("--validate", action="store_true", help="Validate generated outputs")
    parser.add_argument("--experiments-dir", help="Directory for experiments")
    parser.add_argument("--no-expand", action="store_true", help="Disable query expansion")
    parser.add_argument("--raw", action="store_true", help="Raw LLM mode: disable vector search context")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = IAMPolicyGeneratorV2(
        data_path=args.data if hasattr(args, 'data') and args.data else None,
        score_threshold=args.threshold,
        max_actions=args.max_actions,
        model=args.model,
        experiments_dir=args.experiments_dir,
        use_query_expansion=not args.no_expand,
        target_services=args.services,
        use_vector_search=not args.raw
    )
    
    # Generate policy and test config
    result = generator.generate_policy(args.query, save_to_file=not args.no_save)
    
    if result["status"] == "error":
        logger.error(f"Generation failed: {result['message']}")
        print(json.dumps(result, indent=2))
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        policy_valid, policy_errors = generator.validate_policy(result["iam_policy"])
        test_valid, test_errors = generator.validate_test_config(result["test_config"])
        
        if not policy_valid:
            logger.warning(f"Policy validation failed: {policy_errors}")
        if not test_valid:
            logger.warning(f"Test config validation failed: {test_errors}")
        
        result["validation"] = {
            "policy_valid": policy_valid,
            "policy_errors": policy_errors if not policy_valid else [],
            "test_config_valid": test_valid,
            "test_config_errors": test_errors if not test_valid else []
        }
    
    # Output results
    print("\n" + "="*60)
    print("GENERATED IAM POLICY")
    print("="*60)
    print(json.dumps(result["iam_policy"], indent=2))
    
    print("\n" + "="*60)
    print("TEST CONFIGURATION")
    print("="*60)
    print(json.dumps(result["test_config"], indent=2))
    
    print("\n" + "="*60)
    print("METADATA")
    print("="*60)
    print(f"Model: {result['metadata']['model']}")
    print(f"Actions searched: {result['metadata']['actions_searched']}")
    print(f"Score threshold: {result['metadata']['score_threshold']}")
    
    if "files" in result:
        print(f"\nSaved files:")
        print(f"  Policy: {result['files']['policy']}")
        print(f"  Test config: {result['files']['test']}")
    
    print("\nTop actions considered:")
    for action in result["metadata"]["top_actions"]:
        status = "✓" if action["included"] else "✗"
        print(f"  {status} {action['action']} (score: {action['score']:.4f})")


if __name__ == "__main__":
    main()
