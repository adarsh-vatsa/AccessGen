#!/usr/bin/env python3
"""
Service Router and Query Expander using Gemini 2.5 Flash
Identifies AWS services and expands queries for better search
"""
import os
import logging
from typing import List, Optional, Tuple
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


class ServiceRouter:
    """Routes queries to appropriate AWS services and expands them for better search"""
    
    def __init__(self, model: str = "models/gemini-2.5-flash"):
        """Initialize with Gemini Flash for fast, accurate service detection"""
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.model = model
        
        if types:  # Using google.genai
            self.genai_client = genai.Client(api_key=api_key)
        else:  # Using google.generativeai
            genai.configure(api_key=api_key)
            self.genai_client = None
    
    def identify_services(self, query: str) -> List[str]:
        """
        Identify which AWS services are referenced in the query
        
        Returns:
            List of service prefixes: ['s3'], ['ec2'], ['iam'], or combinations
        """
        prompt = f"""You are an AWS expert. Identify which AWS services are referenced in this query.

Query: {query}

Consider these patterns:
- S3: buckets, objects, files, storage, upload, download, versioning, lifecycle, CORS, ACL, multipart
- EC2: instances, servers, VMs, AMIs, volumes, snapshots, security groups, VPCs, subnets, elastic IPs
- IAM: roles, users, groups, policies, permissions, access keys, MFA, SAML, OIDC, credentials

Return ONLY the service prefixes that are clearly relevant, as a comma-separated list.
If no specific service is mentioned, return "all".
If multiple services are needed, list them all.

Examples:
- "upload files to bucket" -> s3
- "launch new server" -> ec2  
- "create role for lambda" -> iam
- "list all resources" -> all
- "instance profile with S3 access" -> ec2,iam,s3

Service prefixes (lowercase, comma-separated):"""

        try:
            if types:  # Using google.genai
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for consistency
                        max_output_tokens=50,
                    )
                )
                if response and hasattr(response, 'text') and response.text:
                    services_text = response.text.strip().lower()
                else:
                    logger.warning("No services identified, defaulting to all")
                    return ["all"]
            else:  # Using google.generativeai
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=50,
                    )
                )
                if hasattr(response, 'text'):
                    services_text = response.text.strip().lower()
                elif hasattr(response, 'candidates'):
                    services_text = response.candidates[0].content.parts[0].text.strip().lower()
                else:
                    services_text = str(response).strip().lower()
            
            # Parse the response
            if not services_text or services_text == "all":
                return ["all"]
            
            # Split and clean service names
            services = [s.strip() for s in services_text.split(',')]
            
            # Validate service names
            valid_services = {'s3', 'ec2', 'iam'}
            services = [s for s in services if s in valid_services]
            
            if not services:
                logger.warning(f"No valid services found in: {services_text}, defaulting to all")
                return ["all"]
            
            logger.info(f"Identified services: {services} for query: {query}")
            return services
            
        except Exception as e:
            logger.error(f"Service identification failed: {e}, defaulting to all")
            return ["all"]
    
    def expand_query(self, original_query: str, services: Optional[List[str]] = None) -> str:
        """
        Expand a query to include implicit requirements
        
        Args:
            original_query: The original natural language query
            services: Optional list of services to focus on
            
        Returns:
            Expanded query with implicit requirements made explicit
        """
        
        # If services not provided, identify them
        if services is None:
            services = self.identify_services(original_query)
        
        # Build service context for prompt
        if "all" in services:
            service_context = "all AWS services (S3, EC2, IAM)"
        else:
            service_context = ", ".join(services).upper()
        
        prompt = f"""You are an AWS expert helping to clarify search queries for IAM policy generation.

Given a natural language requirement, make it more specific while PRESERVING THE ORIGINAL INTENT.

Target services: {service_context}

CRITICAL RULES:
1. PRESERVE the main actions mentioned (if user says "create", keep "create" as primary)
2. Only add commonly paired operations (e.g., "create" often needs "list" to verify)
3. DO NOT shift focus away from the main request
4. Be concise - return a single sentence
5. If the query is already clear, return it unchanged

Examples of GOOD expansions:
- "create IAM users" → "create new IAM users and list existing users"
- "upload files to S3" → "upload files to S3 buckets and list bucket contents"
- "launch EC2 instances" → "launch EC2 instances and describe their status"

Examples of BAD expansions (DO NOT DO THIS):
- "create IAM users" → "attach policies to users" (WRONG - shifts focus from create)
- "manage users" → "delete and modify users" (WRONG - drops the create aspect)

Original query: {original_query}

Clarified query (preserve main intent):"""

        try:
            if types:  # Using google.genai
                response = self.genai_client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=150,
                    )
                )
                # Try multiple ways to get the text
                expanded = None
                if response and hasattr(response, 'text') and response.text:
                    expanded = response.text.strip()
                elif response and hasattr(response, 'candidates') and response.candidates:
                    if response.candidates[0].content.parts:
                        expanded = response.candidates[0].content.parts[0].text.strip()
                
                if not expanded:
                    logger.warning(f"No expansion generated, using original query")
                    return original_query
                # Continue to validation below
            else:  # Using google.generativeai
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=150,
                    )
                )
                if hasattr(response, 'text'):
                    expanded = response.text.strip()
                elif hasattr(response, 'candidates'):
                    expanded = response.candidates[0].content.parts[0].text.strip()
                else:
                    expanded = str(response).strip()
            
            # Validation
            if not expanded or len(expanded) > len(original_query) * 4:
                logger.warning("Query expansion produced unusual result, using original")
                return original_query
            
            logger.info(f"Expanded query: {original_query} → {expanded}")
            return expanded
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}, using original query")
            return original_query
    
    def route_and_expand(self, query: str) -> Tuple[List[str], str]:
        """
        Identify services and expand query in one call
        
        Returns:
            Tuple of (services_list, expanded_query)
        """
        services = self.identify_services(query)
        expanded = self.expand_query(query, services)
        return services, expanded


def main():
    """Test the service router"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test service routing and query expansion")
    parser.add_argument("query", help="Query to route and expand")
    parser.add_argument("--model", default="models/gemini-2.0-flash-exp", help="Model to use")
    
    args = parser.parse_args()
    
    router = ServiceRouter(model=args.model)
    
    # Test service identification
    services = router.identify_services(args.query)
    print(f"Identified services: {services}")
    
    # Test query expansion
    expanded = router.expand_query(args.query, services)
    print(f"Original: {args.query}")
    print(f"Expanded: {expanded}")
    
    # Test combined
    print("\nCombined routing:")
    services, expanded = router.route_and_expand(args.query)
    print(f"Services: {services}")
    print(f"Expanded: {expanded}")


if __name__ == "__main__":
    main()