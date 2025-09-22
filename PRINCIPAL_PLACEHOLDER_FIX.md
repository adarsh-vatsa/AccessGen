# Principal Placeholder Fix - Documentation

## Problem Solved
Previously, when users didn't specify a principal in their query, the system defaulted to using `"*"` which means "all principals" in AWS IAM. This was semantically incorrect and dangerous for testing.

## Solution Implemented
Updated the system prompt to use semantic placeholders that clearly indicate when values need to be filled in by the user.

## Principal Handling Rules

### 1. Explicit "All" Access
**User says:** "allow everyone", "public access", "anyone can read"  
**Generated:** `["*"]`  
**Meaning:** Genuine public/universal access

### 2. No Principal Mentioned
**User says:** "allow reading S3 objects" (no who)  
**Generated:** `["${PRINCIPAL_PLACEHOLDER}"]`  
**Meaning:** Principal needs to be specified

### 3. Specific Role/Group
**User says:** "developers need to upload"  
**Generated:** `["arn:aws:iam::${ACCOUNT_ID}:role/developers"]`  
**Meaning:** Specific role with account ID to be filled

### 4. Generic Users
**User says:** "allow users to download"  
**Generated:** `["arn:aws:iam::${ACCOUNT_ID}:user/${USER_PLACEHOLDER}"]`  
**Meaning:** User identity needs specification

### 5. Service Principals
**User says:** "EC2 instances need S3 access"  
**Generated:** `["ec2.amazonaws.com"]`  
**Meaning:** AWS service principal

## Test Results

All scenarios tested successfully:

| Scenario | Input | Output Principal |
|----------|-------|-----------------|
| No principal | "allow reading S3 objects" | `${PRINCIPAL_PLACEHOLDER}` |
| Explicit all | "allow everyone to read" | `*` |
| Specific role | "developers need access" | `arn:aws:iam::${ACCOUNT_ID}:role/developers` |
| Service principal | "EC2 instances need access" | `ec2.amazonaws.com` |
| Generic users | "users can download" | `arn:aws:iam::${ACCOUNT_ID}:user/${USER_PLACEHOLDER}` |

## Benefits

1. **Semantic Clarity**: Placeholders clearly indicate "needs to be filled"
2. **Security**: No accidental public access when principal is unspecified  
3. **Testing Accuracy**: Test configurations accurately reflect intent
4. **Template Ready**: Standard `${VARIABLE}` syntax for substitution
5. **Explicit Intent**: Clear distinction between "all" and "unspecified"

## Usage

The test configuration now generates proper placeholders:

```json
{
  "service": "s3",
  "rules": [{
    "id": "R1",
    "effect": "Allow",
    "principals": ["${PRINCIPAL_PLACEHOLDER}"],  // Clear placeholder
    "actions": ["s3:GetObject"],
    "resources": ["arn:aws:s3:::${BUCKET_NAME}/*"],
    "conditions": {}
  }]
}
```

Users or testing frameworks can easily identify and substitute these placeholders with actual values before execution.