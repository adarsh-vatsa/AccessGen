# Source Directory Structure

## Core Components

### Policy Generation
- **policy_generator.py**: Main IAM policy generator using unified search
  - Uses UnifiedQueryEngine for multi-service search
  - Generates both IAM policies and test configurations
  - Supports S3, EC2, and IAM services
  
### Service Routing
- **service_router.py**: Identifies AWS services and expands queries
  - Uses Gemini-2.5-flash for fast service detection
  - Expands queries to include implicit requirements
  - Returns service list and expanded query

### Data Fetching (`fetch/`)
- **s3_reference.py**: Fetches S3 service reference from AWS
- **ec2_reference.py**: Fetches EC2 service reference from AWS  
- **iam_reference.py**: Fetches IAM service reference from AWS

### Data Parsing (`parse/`)
- **build_s3_registry_from_reference.py**: Parses S3 reference into registry
- **build_ec2_registry_from_reference.py**: Parses EC2 reference into registry
- **build_iam_registry_from_reference.py**: Parses IAM reference into registry

## Usage

### Generate IAM Policy
```bash
python src/policy_generator.py "your natural language query"

# Target specific service
python src/policy_generator.py "list S3 buckets" --services s3

# Multi-service auto-detection
python src/policy_generator.py "EC2 instance with S3 access"
```

### Test Service Router
```bash
python src/service_router.py "upload files to S3"
```

## Architecture Flow
```
Natural Language Query
    ↓
Service Router (identifies services + expands query)
    ↓
Unified Query Engine (vector search across all services)
    ↓
Gemini 2.5 Pro (extracts components)
    ↓
Output: IAM Policy + Test Configuration
```

## Path Dependencies
- Imports `query_unified` from `../pinecone/`
- Uses enriched data from `../enriched_data/`
- Saves experiments to `../experiments/` (policies and tests)
