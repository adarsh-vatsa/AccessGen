#!/usr/bin/env python3
"""
Add extras fields (sparse_text, dense_text, query_hooks) to EC2 and IAM registries
for improved search quality in Pinecone indexes.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Set


class ExtrasFieldsGenerator:
    """Generate sparse_text, dense_text, and smart query hooks for AWS service actions"""
    
    def __init__(self):
        # Action verb mappings for intelligent query hook generation
        self.verb_mappings = {
            # S3-specific
            'abort': ['cancel', 'stop', 'terminate', 'halt'],
            'put': ['upload', 'write', 'store', 'save', 'add'],
            'get': ['download', 'read', 'retrieve', 'fetch', 'access'],
            'delete': ['remove', 'erase', 'purge', 'destroy', 'clear'],
            'create': ['new', 'make', 'generate', 'provision', 'setup'],
            'list': ['enumerate', 'show', 'display', 'view', 'browse'],
            
            # EC2-specific
            'run': ['launch', 'start', 'boot', 'spin up', 'initialize'],
            'terminate': ['destroy', 'delete permanently', 'remove', 'shutdown permanently'],
            'stop': ['halt', 'pause', 'suspend', 'shutdown'],
            'start': ['resume', 'restart', 'boot', 'turn on'],
            'reboot': ['restart', 'reset', 'cycle'],
            'describe': ['get details', 'show info', 'inspect', 'view details'],
            'modify': ['change', 'update', 'edit', 'alter'],
            'attach': ['connect', 'link', 'associate', 'bind'],
            'detach': ['disconnect', 'unlink', 'disassociate', 'unbind'],
            'allocate': ['reserve', 'assign', 'provision'],
            'release': ['free', 'deallocate', 'unassign'],
            'associate': ['link', 'connect', 'bind', 'map'],
            'disassociate': ['unlink', 'disconnect', 'unbind', 'unmap'],
            
            # IAM-specific
            'assume': ['take on', 'adopt', 'use'],
            'attach': ['grant', 'add', 'assign', 'apply'],
            'detach': ['revoke', 'remove', 'unassign'],
            'simulate': ['test', 'evaluate', 'check'],
            'tag': ['label', 'mark', 'annotate'],
            'untag': ['remove label', 'unmark', 'remove tag'],
            'enable': ['activate', 'turn on', 'allow'],
            'disable': ['deactivate', 'turn off', 'block'],
            'update': ['modify', 'change', 'edit'],
            'upload': ['add', 'import', 'load'],
            'pass': ['delegate', 'grant', 'allow assume'],
        }
        
        # Resource context mappings for better query hooks
        self.resource_contexts = {
            # S3
            'bucket': ['bucket', 's3 bucket', 'storage container'],
            'object': ['file', 'object', 'document', 'content', 'blob'],
            'multipart': ['large file', 'multipart upload', 'chunked upload'],
            'versioning': ['version', 'revision', 'version history'],
            'lifecycle': ['lifecycle', 'retention', 'expiration', 'transition'],
            'acl': ['permissions', 'access control', 'ACL'],
            'cors': ['cross-origin', 'CORS', 'browser access'],
            'website': ['static website', 'web hosting', 'public website'],
            
            # EC2  
            'instance': ['instance', 'server', 'VM', 'virtual machine', 'EC2', 'compute'],
            'image': ['AMI', 'image', 'machine image', 'template'],
            'volume': ['disk', 'storage', 'EBS', 'volume', 'drive'],
            'snapshot': ['backup', 'snapshot', 'point-in-time copy'],
            'securitygroup': ['firewall', 'security group', 'network rules'],
            'keypair': ['SSH key', 'key pair', 'access key', 'login key'],
            'vpc': ['network', 'VPC', 'virtual network'],
            'subnet': ['subnet', 'network segment', 'subnetwork'],
            'elasticip': ['static IP', 'elastic IP', 'public IP', 'fixed IP'],
            
            # IAM
            'user': ['user', 'IAM user', 'account'],
            'group': ['group', 'user group', 'IAM group'],
            'role': ['role', 'IAM role', 'service role', 'assumed role'],
            'policy': ['policy', 'permissions', 'IAM policy', 'access rules'],
            'accesskey': ['access key', 'API key', 'credentials', 'programmatic access'],
            'mfa': ['MFA', 'multi-factor', 'two-factor', '2FA'],
            'saml': ['SAML', 'SSO', 'single sign-on', 'federation'],
            'oidc': ['OIDC', 'OpenID', 'identity provider'],
        }
    
    def extract_action_verb(self, action: str) -> str:
        """Extract the primary verb from an action name"""
        # Handle special cases first
        if action.startswith('Put'):
            return 'put'
        if action.startswith('Get'):
            return 'get'
        if action.startswith('List'):
            return 'list'
        if action.startswith('Delete'):
            return 'delete'
        if action.startswith('Create'):
            return 'create'
        
        # Extract verb from CamelCase
        verb = ''
        for char in action:
            if char.isupper() and verb:
                break
            verb += char.lower()
        return verb
    
    def extract_resource_context(self, action: str, resource_types: List[Dict]) -> List[str]:
        """Extract resource context from action name and resource types"""
        contexts = []
        action_lower = action.lower()
        
        # Check resource types first
        for rt in resource_types:
            resource_type = rt.get('type', '').lower()
            if resource_type in self.resource_contexts:
                contexts.extend(self.resource_contexts[resource_type])
        
        # Fallback to parsing from action name
        if not contexts:
            for resource, aliases in self.resource_contexts.items():
                if resource in action_lower:
                    contexts.extend(aliases)
                    break
        
        return list(set(contexts)) if contexts else []
    
    def generate_query_hooks(self, action: str, description: str, resource_types: List[Dict], service: str) -> List[str]:
        """Generate smart, context-aware query hooks"""
        hooks = []
        action_lower = action.lower()
        verb = self.extract_action_verb(action)
        
        # Parse the description for better context
        desc_lower = description.lower() if description else ""
        
        # Special handling for List actions - CRITICAL FIX
        if verb == 'list' or 'list' in action_lower:
            # EC2 List actions
            if service == 'ec2':
                if 'ListImagesInRecycleBin' == action:
                    hooks.extend(['list deleted AMIs', 'show recycled images', 'view AMIs in recycle bin',
                                 'enumerate deleted machine images'])
                elif 'ListSnapshotsInRecycleBin' == action:
                    hooks.extend(['list deleted snapshots', 'show recycled backups', 'view snapshots in recycle bin',
                                 'enumerate deleted snapshots'])
                else:
                    # Generic EC2 list - avoid overly broad hooks
                    pass
            # IAM List actions
            elif service == 'iam':
                if 'ListRoles' == action:
                    hooks.extend(['list IAM roles', 'show all roles', 'enumerate service roles',
                                 'view IAM roles', 'get all roles'])
                elif 'ListUsers' == action:
                    hooks.extend(['list IAM users', 'show all users', 'enumerate users',
                                 'view IAM users', 'get all users'])
                elif 'ListPolicies' == action:
                    hooks.extend(['list IAM policies', 'show all policies', 'enumerate policies',
                                 'view IAM policies', 'get all policies'])
                elif 'ListGroups' == action:
                    hooks.extend(['list IAM groups', 'show all groups', 'enumerate groups',
                                 'view IAM groups', 'get all groups'])
                elif 'ListAccessKeys' == action:
                    hooks.extend(['list access keys', 'show API keys', 'enumerate credentials',
                                 'view access keys', 'get user credentials'])
            # S3 List actions (for reference)
            elif service == 's3':
                if 'ListBucket' == action:
                    hooks.extend(['list objects', 'list files', 'browse bucket', 'see contents', 
                                 'view files in bucket', 'enumerate objects', 'directory listing',
                                 'show bucket contents', 'ls bucket'])
                elif 'ListAllMyBuckets' == action:
                    hooks.extend(['list buckets', 'show all buckets', 'enumerate buckets', 
                                 'see my buckets', 'view storage containers'])
                elif 'ListBucketVersions' == action:
                    hooks.extend(['list versions', 'show object versions', 'version history', 
                                 'enumerate versions', 'see file versions'])
                elif 'ListMultipartUploads' in action:
                    hooks.extend(['list uploads', 'show ongoing uploads', 'incomplete uploads',
                                 'multipart status', 'upload progress'])
        
        # Handle Describe actions specially for EC2
        elif verb == 'describe' and service == 'ec2':
            if 'DescribeInstances' == action:
                hooks.extend(['list instances', 'show EC2 instances', 'get instance details',
                             'view servers', 'describe EC2', 'instance information'])
            elif 'DescribeImages' == action:
                hooks.extend(['list AMIs', 'show machine images', 'get AMI details',
                             'view images', 'describe AMIs', 'available images'])
            elif 'DescribeVolumes' == action:
                hooks.extend(['list EBS volumes', 'show disks', 'get volume details',
                             'view storage volumes', 'describe EBS', 'disk information'])
            elif 'DescribeSnapshots' == action:
                hooks.extend(['list snapshots', 'show backups', 'get snapshot details',
                             'view EBS snapshots', 'describe backups', 'snapshot information'])
            elif 'DescribeSecurityGroups' == action:
                hooks.extend(['list security groups', 'show firewall rules', 'get security group details',
                             'view network rules', 'describe firewalls', 'security group information'])
            elif 'DescribeVpcs' == action:
                hooks.extend(['list VPCs', 'show virtual networks', 'get VPC details',
                             'view networks', 'describe VPCs', 'network information'])
            elif 'DescribeSubnets' == action:
                hooks.extend(['list subnets', 'show network segments', 'get subnet details',
                             'view subnets', 'describe subnets', 'subnet information'])
            else:
                # Generic describe based on description
                if 'instance' in desc_lower:
                    hooks.extend(['get instance info', 'show server details'])
                elif 'volume' in desc_lower or 'ebs' in desc_lower:
                    hooks.extend(['get volume info', 'show disk details'])
                elif 'snapshot' in desc_lower:
                    hooks.extend(['get snapshot info', 'show backup details'])
        
        # Handle other verbs with context
        elif verb in self.verb_mappings:
            synonyms = self.verb_mappings[verb]
            resource_contexts = self.extract_resource_context(action, resource_types)
            
            if resource_contexts:
                # Generate combinations of verb synonyms + resource context
                for synonym in synonyms[:3]:  # Limit combinations
                    for ctx in resource_contexts[:2]:
                        hooks.append(f"{synonym} {ctx}")
            else:
                # Just use verb synonyms if no clear resource context
                hooks.extend(synonyms[:3])
        
        # Add action-specific hooks based on common patterns
        if service == 'ec2':
            if 'RunInstances' == action:
                hooks.extend(['launch EC2', 'start new server', 'create VM', 
                             'boot instance', 'spin up server', 'provision compute'])
            elif 'TerminateInstances' == action:
                hooks.extend(['destroy instance', 'delete server permanently', 
                             'terminate EC2', 'remove VM permanently'])
            elif 'StopInstances' == action:
                hooks.extend(['stop server', 'pause instance', 'halt EC2', 
                             'suspend VM', 'shutdown instance'])
            elif 'StartInstances' == action:
                hooks.extend(['start server', 'resume instance', 'boot EC2',
                             'turn on VM', 'restart stopped instance'])
            elif 'CreateImage' == action:
                hooks.extend(['create AMI', 'make image', 'snapshot instance',
                             'backup server', 'save instance template'])
            elif 'AllocateAddress' == action:
                hooks.extend(['get elastic IP', 'reserve IP address', 'allocate static IP',
                             'provision public IP'])
                
        elif service == 'iam':
            if 'CreateRole' == action:
                hooks.extend(['new IAM role', 'create service role', 'setup role',
                             'define role permissions', 'make assumed role'])
            elif 'AttachRolePolicy' == action:
                hooks.extend(['grant permissions to role', 'add policy to role',
                             'assign role permissions', 'apply policy to role'])
            elif 'CreateAccessKey' == action:
                hooks.extend(['generate API key', 'create credentials', 'new access key',
                             'programmatic access', 'API credentials'])
            elif 'AssumeRole' == action:
                hooks.extend(['switch role', 'assume IAM role', 'take on role',
                             'use service role', 'adopt role permissions'])
            elif 'CreatePolicy' == action:
                hooks.extend(['define permissions', 'create IAM policy', 'new policy',
                             'setup access rules', 'write policy document'])
            elif 'PassRole' == action:
                hooks.extend(['delegate role', 'allow role assumption', 'pass IAM role',
                             'grant role to service', 'enable service to assume role'])
        
        elif service == 's3':
            if 'PutObject' == action:
                hooks.extend(['upload file', 'store document', 'save to S3',
                             'write object', 'push content', 'add file to bucket'])
            elif 'GetObject' == action:
                hooks.extend(['download file', 'retrieve document', 'read from S3',
                             'fetch object', 'pull content', 'get file from bucket'])
            elif 'DeleteObject' == action:
                hooks.extend(['delete file', 'remove object', 'erase document',
                             'purge from S3', 'destroy file', 'clear from bucket'])
            elif 'CreateBucket' == action:
                hooks.extend(['new S3 bucket', 'create storage', 'make bucket',
                             'provision storage container', 'setup S3 storage'])
        
        # Parse description for additional context (but be selective)
        # Don't add generic "grant permission" - it's too broad and applies to everything
        if description and not hooks:  # Only if we have no hooks yet
            desc_lower = description.lower()
            if 'multipart' in desc_lower and 'multipart' not in ' '.join(hooks).lower():
                hooks.append('multipart operation')
        
        # Remove duplicates and filter out overly generic terms
        hooks = list(set(hooks))
        # Filter out single words that are too generic
        hooks = [h for h in hooks if len(h.split()) > 1 or h in ['upload', 'download', 'backup']]
        
        return hooks[:8]  # Limit to 8 most relevant hooks
    
    def generate_sparse_text(self, action: Dict[str, Any], service: str) -> str:
        """Generate sparse text for BM25 indexing"""
        parts = [
            f"service={service}",
            f"action={action['action']}",
            f"access={action.get('access_level', 'Unknown')}"
        ]
        
        # Add resource types
        resource_types = [rt.get('type', '') for rt in action.get('resource_types', [])]
        if resource_types:
            parts.append(f"resources=[{', '.join(resource_types)}]")
        
        # Add condition keys
        condition_keys = action.get('condition_keys', [])
        if condition_keys:
            parts.append(f"condition_keys=[{', '.join(condition_keys)}]")
        
        return '\n'.join(parts)
    
    def generate_dense_text(self, action: Dict[str, Any], service: str, query_hooks: List[str]) -> str:
        """Generate dense text for semantic embedding"""
        # Start with sparse text
        dense = self.generate_sparse_text(action, service)
        
        # Add description
        description = action.get('description', '')
        if description:
            dense += f"\ndescription={description}"
        
        # Add resource labels (more human-friendly than raw types)
        resource_types = action.get('resource_types', [])
        if resource_types:
            labels = []
            for rt in resource_types:
                rtype = rt.get('type', '')
                # Make resource types more human-readable
                if rtype == 'accesspointobject':
                    labels.append('object via access point')
                elif rtype == 'multiupload':
                    labels.append('multipart upload')
                elif rtype == 'accessgrantsinstance':
                    labels.append('access grants instance')
                else:
                    labels.append(rtype)
            dense += f"\nresource_labels=[{', '.join(labels)}]"
        
        # Add query hooks
        if query_hooks:
            dense += f"\nquery_hooks=[{', '.join(query_hooks)}]"
        
        return dense
    
    def generate_condition_hints(self, condition_keys: List[str], service: str) -> Dict[str, str]:
        """Generate human-readable hints for condition keys"""
        hints = {}
        
        common_hints = {
            'aws:RequestTag/${TagKey}': 'Restrict by tags in the request',
            'aws:ResourceTag/${TagKey}': 'Restrict by tags on the resource',
            'aws:TagKeys': 'Restrict by tag keys being used',
            'aws:SecureTransport': 'Require SSL/TLS connection',
            'aws:SourceIp': 'Restrict by source IP address',
            'aws:userid': 'Restrict by user ID',
            'aws:username': 'Restrict by username',
        }
        
        service_hints = {
            's3': {
                's3:x-amz-acl': 'Control ACL permissions on upload',
                's3:x-amz-server-side-encryption': 'Require server-side encryption',
                's3:x-amz-storage-class': 'Restrict storage class (STANDARD, GLACIER, etc.)',
                's3:ExistingObjectTag/${TagKey}': 'Restrict by existing object tags',
                's3:delimiter': 'Control list delimiter for object listing',
                's3:prefix': 'Restrict by object key prefix',
                's3:max-keys': 'Limit number of keys returned',
            },
            'ec2': {
                'ec2:InstanceType': 'Restrict by instance type (t2.micro, m5.large, etc.)',
                'ec2:Vpc': 'Restrict to specific VPC',
                'ec2:Subnet': 'Restrict to specific subnet',
                'ec2:Region': 'Restrict to specific AWS region',
                'ec2:ImageType': 'Restrict by AMI type',
                'ec2:RootDeviceType': 'Restrict by root device type (ebs, instance-store)',
                'ec2:Tenancy': 'Restrict by tenancy (default, dedicated, host)',
            },
            'iam': {
                'iam:PolicyARN': 'Restrict to specific policy ARN',
                'iam:PassedToService': 'Restrict which service can assume the role',
                'iam:PermissionsBoundary': 'Enforce permissions boundary',
                'iam:AWSServiceName': 'Restrict by AWS service name',
                'iam:OrganizationsPolicyId': 'Restrict by Organizations policy',
            }
        }
        
        for key in condition_keys:
            # Check common hints first
            if key in common_hints:
                hints[key] = common_hints[key]
            # Check service-specific hints
            elif service in service_hints and key in service_hints[service]:
                hints[key] = service_hints[service][key]
            # Generate a basic hint for unknown keys
            elif ':' in key:
                parts = key.split(':')
                if len(parts) >= 2:
                    hints[key] = f"Condition related to {parts[1]}"
        
        return hints
    
    def compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute a hash of the serialized data for versioning"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def process_registry(self, input_path: Path, output_path: Path, service: str) -> None:
        """Process a registry file and add extras fields"""
        print(f"Processing {service} registry from {input_path}")
        
        # Load the enriched registry
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        service_data = data[service]
        actions = service_data['actions']
        
        # Process each action
        processed_count = 0
        for action in actions:
            # Generate query hooks
            query_hooks = self.generate_query_hooks(
                action['action'],
                action.get('description', ''),
                action.get('resource_types', []),
                service
            )
            
            # Generate sparse text
            sparse_text = self.generate_sparse_text(action, service)
            
            # Generate dense text
            dense_text = self.generate_dense_text(action, service, query_hooks)
            
            # Generate condition hints
            condition_hints = self.generate_condition_hints(
                action.get('condition_keys', []),
                service
            )
            
            # Add the new fields
            action['query_hooks'] = query_hooks
            action['condition_hints'] = condition_hints
            action['sparse_text'] = sparse_text
            action['dense_text'] = dense_text
            action['serialization_hash'] = self.compute_hash(action)
            
            processed_count += 1
            
            # Show progress for first few
            if processed_count <= 3:
                print(f"  {action['action']}: {len(query_hooks)} hooks - {query_hooks[:3]}")
        
        # Save the enhanced data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Processed {processed_count} actions")
        print(f"Saved to {output_path}")


def main():
    """Process EC2 and IAM registries to add extras fields"""
    generator = ExtrasFieldsGenerator()
    
    # Define paths
    root = Path(__file__).resolve().parents[1]
    enriched_dir = root / "enriched_data"
    
    # Process EC2
    ec2_input = enriched_dir / "aws_iam_registry_ec2_enriched.json"
    ec2_output = enriched_dir / "aws_iam_registry_ec2_enriched_extras.json"
    if ec2_input.exists():
        generator.process_registry(ec2_input, ec2_output, 'ec2')
    else:
        print(f"EC2 registry not found at {ec2_input}")
    
    print()
    
    # Process IAM
    iam_input = enriched_dir / "aws_iam_registry_iam_enriched.json"
    iam_output = enriched_dir / "aws_iam_registry_iam_enriched_extras.json"
    if iam_input.exists():
        generator.process_registry(iam_input, iam_output, 'iam')
    else:
        print(f"IAM registry not found at {iam_input}")
    
    print("\nDone! The enriched extras files are ready for indexing.")


if __name__ == "__main__":
    main()