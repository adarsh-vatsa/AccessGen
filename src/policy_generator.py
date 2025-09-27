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

# Add pinecone directory to path for optional imports (vector mode only)
sys.path.insert(0, str(Path(__file__).parent.parent / "pinecone"))
try:
    from src.guards.registry_guard import RegistryGuard
except Exception:
    try:
        from guards.registry_guard import RegistryGuard  # fallback when running as module
    except Exception:
        RegistryGuard = None  # type: ignore
try:
    from src.guards.registry_validator import RegistryValidator
except Exception:
    try:
        from guards.registry_validator import RegistryValidator  # fallback
    except Exception:
        RegistryValidator = None  # type: ignore
from dotenv import load_dotenv

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
import os as _os
_FORCE_LEGACY = (_os.getenv("GENAI_FORCE_LEGACY", "0") == "1")
try:
    import google.generativeai as legacy_genai  # type: ignore
except Exception:
    legacy_genai = None  # type: ignore

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
            # Lazy import to avoid pulling pinecone when not needed
            try:
                from query_unified import UnifiedQueryEngine as _UnifiedQueryEngine  # type: ignore
                self.query_engine = _UnifiedQueryEngine(use_service_router=use_query_expansion)
            except Exception as _e:
                logger.error(f"Failed to initialize vector search engine: {_e}")
                self.query_engine = None
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
        
        # Prefer legacy when forced AND available; otherwise try new client and fall back
        self.genai_client = None
        if _FORCE_LEGACY and legacy_genai is not None:
            try:
                legacy_genai.configure(api_key=api_key)
            except Exception:
                pass
        else:
            if types is not None and genai is not None:
                try:
                    self.genai_client = genai.Client(api_key=api_key)  # type: ignore
                except Exception:
                    self.genai_client = None
            # If new client failed or unavailable, try legacy
            if self.genai_client is None and legacy_genai is not None:
                try:
                    legacy_genai.configure(api_key=api_key)
                except Exception:
                    pass
        # Final guard: if neither client is usable, raise explicit error
        if self.genai_client is None and legacy_genai is None:
            raise RuntimeError("No usable Gemini client: install google-generativeai or google-genai")
        
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
        if self.query_engine is None:
            raise RuntimeError(
                "Vector search is unavailable. Pinecone client could not be initialized. "
                "If you have 'pinecone-plugin-inference' installed, uninstall it: "
                "pip uninstall pinecone-plugin-inference. Or switch UI Mode to RAW."
            )
        
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
        """Legacy RAW-mode prompt (kept for compatibility)."""
        return f"""Natural Language Requirement:
{query}

CRITICAL INSTRUCTIONS:
1. Generate a minimal, least-privilege IAM policy that directly satisfies the requirement.
2. Prefer specific actions over wildcards; include only what is necessary.
3. Use appropriate resource ARNs and placeholders (e.g., ${{BUCKET_NAME}}, ${{ACCOUNT_ID}}).
4. If multiple AWS services are implied, include only those needed.
5. Output both an IAM policy and a matching test configuration.

Generate the output JSON with both 'iam_policy' and 'test_config' fields:"""

    def _build_raw_policy_only_prompt(self, query: str) -> str:
        """Build a short, policy-only prompt for RAW mode to improve reliability.

        The model must return exactly one valid JSON AWS policy object with keys
        Version and Statement, and no surrounding prose or code fences.
        """
        return f"""
Return only one valid AWS S3 bucket policy as pure JSON. No comments. No code fences. No explanations.

Rules:
- Use S3 actions only and least privilege.
- Use precise ARNs: bucket arn:aws:s3:::{{BUCKET_NAME}} and objects arn:aws:s3:::{{BUCKET_NAME}}/*.
- Apply required conditions and explicit Deny statements as described.

Input:
{query}

Output: a single JSON object with keys Version and Statement only (standard bucket policy structure).
""".strip()

    @staticmethod
    def _policy_to_test_config(policy: Dict[str, Any], default_service: Optional[str] = None) -> Dict[str, Any]:
        """Derive a deterministic test configuration from a bucket/IAM policy."""
        stmts = policy.get("Statement", []) if isinstance(policy, dict) else []
        service = default_service or "s3"
        rules: List[Dict[str, Any]] = []
        rid = 1
        for st in stmts:
            if not isinstance(st, dict):
                continue
            acts = st.get("Action")
            actions: List[str] = []
            if isinstance(acts, str):
                actions = [acts]
            elif isinstance(acts, list):
                actions = [a for a in acts if isinstance(a, str)]
            if actions and ":" in actions[0]:
                service = actions[0].split(":", 1)[0]
            res = st.get("Resource")
            resources: List[str] = []
            if isinstance(res, str):
                resources = [res]
            elif isinstance(res, list):
                resources = [r for r in res if isinstance(r, str)]
            rule = {
                "id": f"R{rid}",
                "effect": st.get("Effect", "Allow"),
                "principals": ["${PRINCIPAL_PLACEHOLDER}"],
                "not_principals": [],
                "actions": actions,
                "resources": resources or ["*"],
                "conditions": st.get("Condition", {}),
            }
            rules.append(rule)
            rid += 1
        return {"service": service, "rules": rules}
    
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

    def _build_vector_policy_only_prompt(self, query: str, actions: List[Dict[str, Any]]) -> str:
        ctx = self._format_action_context(actions)
        return f"""Return only one valid AWS policy JSON as pure JSON. No comments. No code fences. No explanations.

You MUST ONLY use actions from this allowed list:
{ctx}

Input:
{query}

Output: a single JSON object with keys Version and Statement only (standard policy structure)."""

    @staticmethod
    def _extract_text_from_response(resp) -> str:
        # Try standard .text first (both legacy and new SDKs)
        try:
            txt = getattr(resp, 'text', None)
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        except Exception:
            # Some SDKs raise when no parts are returned; ignore and continue
            pass
        # Try candidates -> content.parts[].text
        try:
            cand = getattr(resp, 'candidates', None)
            # Support proto-style repeated fields and lists
            if cand and len(cand) > 0:
                content = getattr(cand[0], 'content', None)
                parts = getattr(content, 'parts', None)
                if parts:
                    # parts may be a RepeatedComposite; iterate safely
                    texts = []
                    try:
                        for p in parts:
                            t = getattr(p, 'text', None)
                            if isinstance(t, str) and t.strip():
                                texts.append(t.strip())
                    except Exception:
                        pass
                    if texts:
                        return "\n".join(texts)
        except Exception:
            pass
        # Try to_dict() if available
        try:
            to_dict = getattr(resp, 'to_dict', None)
            if callable(to_dict):
                d = to_dict()
                # Attempt to harvest any text parts
                if isinstance(d, dict):
                    # Walk common paths
                    cand = d.get('candidates')
                    if isinstance(cand, list) and cand:
                        cont = cand[0].get('content', {})
                        parts = cont.get('parts') or []
                        texts = []
                        for p in parts:
                            t = p.get('text')
                            if isinstance(t, str):
                                texts.append(t)
                        if texts:
                            return "\n".join(t.strip() for t in texts if t and t.strip())
        except Exception:
            pass
        # Fallback to JSON dump if available (new SDK)
        try:
            dump_json = getattr(resp, 'model_dump_json', None)
            if callable(dump_json):
                s = dump_json()
                if s and isinstance(s, str):
                    return s
        except Exception:
            pass
        # Last resort: str(resp)
        try:
            s = str(resp)
            if s and '{' in s:
                return s
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of a JSON object from arbitrary model output."""
        # Fast path
        try:
            return json.loads(text)
        except Exception:
            pass
        # Strip common code fences
        t = re.sub(r"^[`\s]*json\s*", "", text.strip('`').strip(), flags=re.IGNORECASE)
        try:
            return json.loads(t)
        except Exception:
            pass
        # Bracket matching from the first '{'
        start = text.find('{')
        if start == -1:
            return None
        stack = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == '{':
                stack += 1
            elif c == '}':
                stack -= 1
                if stack == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        continue
        return None
    
    def generate_policy(self, query: str, save_to_file: bool = True) -> Dict[str, Any]:
        """
        Generate an IAM policy and test configuration from natural language query
        
        Args:
            query: Natural language description of required permissions
            save_to_file: Whether to save outputs to files
            
        Returns:
            Dictionary containing the generated policy, test config, and metadata
        """
        # Step 1: Either search for relevant actions (vector mode) or run RAW (policy-only) mode
        if self.use_vector_search:
            try:
                actions, search_query = self.search_relevant_actions(query)
            except Exception as e:
                logger.error(f"Vector search unavailable: {e}")
                return {
                    "status": "error",
                    "message": f"Vector mode unavailable: {e}",
                    "query": query
                }
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
            if self.use_vector_search:
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(query, actions)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                # RAW: ask for policy JSON only to improve reliability
                full_prompt = self._build_raw_policy_only_prompt(query)
            
            chosen_model = self.model
            output_text = ""
            err: Optional[Exception] = None
            # Try primary model
            try:
                if self.genai_client is not None and types is not None:
                    response = self.genai_client.models.generate_content(
                        model=chosen_model,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.3,
                            max_output_tokens=8000,
                        ),
                    )
                    output_text = self._extract_text_from_response(response)
                else:
                    model_client = legacy_genai.GenerativeModel(chosen_model)  # type: ignore
                    response = model_client.generate_content(
                        full_prompt,
                        generation_config=legacy_genai.GenerationConfig(  # type: ignore
                            temperature=0.3,
                            max_output_tokens=8000,
                        ),
                    )
                    output_text = self._extract_text_from_response(response)
            except Exception as e1:
                err = e1
                # Fallback to flash if pro is overloaded or failed
                try_model = "models/gemini-2.5-flash" if "pro" in (chosen_model or "") else chosen_model
                try:
                    if self.genai_client is not None and types is not None:
                        response = self.genai_client.models.generate_content(
                            model=try_model,
                            contents=full_prompt,
                            config=types.GenerateContentConfig(
                                temperature=0.3,
                                max_output_tokens=8000,
                            ),
                        )
                        output_text = self._extract_text_from_response(response)
                        chosen_model = try_model
                        err = None
                    else:
                        model_client = legacy_genai.GenerativeModel(try_model)  # type: ignore
                        response = model_client.generate_content(
                            full_prompt,
                            generation_config=legacy_genai.GenerationConfig(  # type: ignore
                                temperature=0.3,
                                max_output_tokens=8000,
                            ),
                        )
                        output_text = self._extract_text_from_response(response)
                        chosen_model = try_model
                        err = None
                except Exception as e2:
                    err = e2
            if err:
                raise err
            
            # Clean up any markdown formatting if present
            output_text = re.sub(r'^```json\s*', '', output_text.strip(), flags=re.IGNORECASE)
            output_text = re.sub(r'\s*```$', '', output_text.strip())

            # Parse model output
            parsed = self._extract_json_object(output_text)
            if not parsed or not isinstance(parsed, dict):
                logger.error("Failed to parse generated output (empty or invalid JSON)")
                if self.use_vector_search:
                    # Attempt policy-only fallback with allowed actions context
                    try:
                        po_prompt = self._build_vector_policy_only_prompt(query, actions)
                        text_po = ""
                        if self.genai_client is not None and types is not None:
                            resp_po = self.genai_client.models.generate_content(
                                model=self.model,
                                contents=po_prompt,
                                config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=4000),
                            )
                            text_po = self._extract_text_from_response(resp_po)
                        else:
                            model_client = legacy_genai.GenerativeModel(self.model)  # type: ignore
                            resp_po = model_client.generate_content(
                                po_prompt,
                                generation_config=legacy_genai.GenerationConfig(temperature=0.2, max_output_tokens=4000),  # type: ignore
                            )
                            text_po = self._extract_text_from_response(resp_po)
                        text_po = re.sub(r'^```json\s*', '', text_po.strip(), flags=re.IGNORECASE)
                        text_po = re.sub(r'\s*```$', '', text_po.strip())
                        obj_po = self._extract_json_object(text_po)
                        if isinstance(obj_po, dict) and "Version" in obj_po and "Statement" in obj_po:
                            parsed = {"iam_policy": obj_po, "test_config": self._policy_to_test_config(obj_po)}
                        else:
                            raise ValueError("policy-only generation failed")
                    except Exception:
                        return {
                            "status": "error",
                            "message": "Generated output is not valid JSON",
                            "raw_output": output_text,
                            "query": query
                        }
                else:
                    return {
                        "status": "error",
                        "message": "Generated output is not valid JSON",
                        "raw_output": output_text,
                        "query": query
                    }

            # Normalize to iam_policy + test_config
            if self.use_vector_search:
                output = parsed
                if "iam_policy" not in output or "test_config" not in output:
                    if "policy" in output and isinstance(output["policy"], dict):
                        output = {"iam_policy": output["policy"], "test_config": self._policy_to_test_config(output["policy"]) }
                    elif isinstance(output, dict) and "Version" in output and "Statement" in output:
                        output = {"iam_policy": output, "test_config": self._policy_to_test_config(output)}
                    else:
                        # Fallback: policy-only generation constrained to allowed actions
                        try:
                            po_prompt = self._build_vector_policy_only_prompt(query, actions)
                            # Reuse same model pipeline with a short prompt
                            chosen_model = self.model
                            text_po = ""
                            if self.genai_client is not None and types is not None:
                                resp_po = self.genai_client.models.generate_content(
                                    model=chosen_model,
                                    contents=po_prompt,
                                    config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=4000),
                                )
                                text_po = self._extract_text_from_response(resp_po)
                            else:
                                model_client = legacy_genai.GenerativeModel(chosen_model)  # type: ignore
                                resp_po = model_client.generate_content(
                                    po_prompt,
                                    generation_config=legacy_genai.GenerationConfig(temperature=0.2, max_output_tokens=4000),  # type: ignore
                                )
                                text_po = self._extract_text_from_response(resp_po)
                            text_po = re.sub(r'^```json\s*', '', text_po.strip(), flags=re.IGNORECASE)
                            text_po = re.sub(r'\s*```$', '', text_po.strip())
                            obj_po = self._extract_json_object(text_po)
                            if isinstance(obj_po, dict) and "Version" in obj_po and "Statement" in obj_po:
                                output = {"iam_policy": obj_po, "test_config": self._policy_to_test_config(obj_po)}
                            else:
                                raise ValueError("policy-only generation failed")
                        except Exception:
                            logger.error("Generated output missing required fields")
                            return {
                                "status": "error",
                                "message": "Generated output missing 'iam_policy' or 'test_config'",
                                "raw_output": output_text,
                                "query": query
                            }
            else:
                # RAW: expect a policy object only; derive test_config deterministically
                policy_obj = None
                if isinstance(parsed, dict):
                    if "Version" in parsed and "Statement" in parsed:
                        policy_obj = parsed
                    else:
                        # common wrapper keys
                        for k in ("iam_policy", "policy", "bucket_policy", "bucketPolicy"):
                            ip = parsed.get(k)
                            if isinstance(ip, dict) and "Version" in ip and "Statement" in ip:
                                policy_obj = ip
                                break
                elif isinstance(parsed, list) and parsed:
                    for item in parsed:
                        if isinstance(item, dict) and "Version" in item and "Statement" in item:
                            policy_obj = item
                            break
                if not policy_obj:
                    logger.error("RAW mode: model did not return a policy object")
                    return {
                        "status": "error",
                        "message": "RAW mode: model did not return a policy object",
                        "raw_output": output_text,
                        "query": query
                    }
                output = {"iam_policy": policy_obj, "test_config": self._policy_to_test_config(policy_obj)}
            
            # Step 3: Prepare response
            result = {
                "status": "success",
                "query": query,
                "iam_policy": output["iam_policy"],
                "test_config": output["test_config"],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model": chosen_model,
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
                    # Run registry validation (conditions/resources) and test_config principals
                    if 'provenance' in result['metadata']['registry'] and RegistryValidator is not None:
                        validator = RegistryValidator()
                        reg_checks = validator.validate_policy(output["iam_policy"], result["metadata"]["registry"]["provenance"])  # type: ignore
                        test_checks = validator.validate_test_config(output.get("test_config", {}))  # type: ignore
                        result["metadata"]["registry_validation"] = {
                            "policy": reg_checks,
                            "test_config": test_checks,
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
