import json
import pytest
from pathlib import Path


@pytest.mark.data_quality
class TestRegistryDataQuality:
    """Data validation and quality tests for the S3 registry."""
    
    @pytest.fixture
    def registry_data(self):
        """Load the actual generated registry data."""
        registry_path = Path("data/aws_iam_registry_s3.json")
        if not registry_path.exists():
            pytest.skip("Registry file not found. Run build first.")
        
        with open(registry_path) as f:
            data = json.load(f)
        
        return data["s3"]
    
    def test_critical_actions_exist(self, registry_data):
        """Verify critical S3 actions are present."""
        critical_actions = [
            "PutObject",
            "GetObject",
            "DeleteObject",
            "ListBucket",
            "CreateBucket",
            "DeleteBucket",
            "PutBucketPolicy",
            "GetBucketPolicy",
            "PutObjectAcl",
            "GetObjectAcl"
        ]
        
        action_names = {a["action"] for a in registry_data["actions"]}
        
        for action in critical_actions:
            assert action in action_names, f"Critical action {action} not found"
    
    def test_common_resource_types_exist(self, registry_data):
        """Verify common resource types are present."""
        required_resources = ["bucket", "object"]
        
        resource_types = {r["type"] for r in registry_data["resource_types"]}
        
        for resource in required_resources:
            assert resource in resource_types, f"Resource type {resource} not found"
    
    def test_access_levels_distribution(self, registry_data):
        """Verify reasonable distribution of access levels."""
        access_levels = {}
        for action in registry_data["actions"]:
            level = action["access_level"]
            access_levels[level] = access_levels.get(level, 0) + 1
        
        # S3 should have actions of all types
        assert "Read" in access_levels
        assert "Write" in access_levels
        assert "List" in access_levels
        assert "Permissions management" in access_levels
        
        # Write actions should be common in S3
        assert access_levels.get("Write", 0) > 20
        assert access_levels.get("Read", 0) > 10
    
    def test_putobject_specifics(self, registry_data):
        """Detailed validation of PutObject action."""
        put_object = next((a for a in registry_data["actions"] 
                          if a["action"] == "PutObject"), None)
        
        assert put_object is not None, "PutObject action not found"
        assert put_object["access_level"] == "Write"
        
        # Should have object resource type
        resource_types = [r["type"] for r in put_object["resource_types"]]
        assert "object" in resource_types
        
        # Should have condition keys
        assert len(put_object["condition_keys"]) > 0
        
        # Provenance should be complete
        assert put_object["source"]["url"].startswith("https://")
        assert put_object["source"]["table"] == "actions"
        assert isinstance(put_object["source"]["row_index"], int)
    
    def test_bucket_resource_validation(self, registry_data):
        """Validate bucket resource type."""
        bucket = next((r for r in registry_data["resource_types"] 
                      if r["type"] == "bucket"), None)
        
        assert bucket is not None, "Bucket resource type not found"
        assert bucket["arn_template"], "Bucket ARN template is empty"
        assert "s3:::" in bucket["arn_template"]
        assert "${BucketName}" in bucket["arn_template"]
    
    def test_object_resource_validation(self, registry_data):
        """Validate object resource type."""
        obj = next((r for r in registry_data["resource_types"] 
                   if r["type"] == "object"), None)
        
        assert obj is not None, "Object resource type not found"
        assert obj["arn_template"], "Object ARN template is empty"
        assert "s3:::" in obj["arn_template"]
    
    def test_condition_keys_validation(self, registry_data):
        """Validate service condition keys."""
        condition_keys = registry_data["service_condition_keys"]
        
        assert len(condition_keys) > 30, "Too few condition keys"
        
        # Check for S3-specific keys
        s3_keys = [k for k in condition_keys if k.startswith("s3:")]
        assert len(s3_keys) > 25, "Too few S3-specific condition keys"
        
        # Check for common S3 condition keys
        common_keys = [
            "s3:x-amz-acl",
            "s3:x-amz-server-side-encryption",
            "s3:prefix"
        ]
        
        for key in common_keys:
            assert key in condition_keys, f"Common key {key} not found"
        
        # Also should have AWS global keys
        aws_keys = [k for k in condition_keys if k.startswith("aws:")]
        assert len(aws_keys) > 0, "No AWS global condition keys found"
    
    def test_schema_completeness(self, registry_data):
        """Verify all expected fields are present in the schema."""
        # Top-level fields
        assert "service_name" in registry_data
        assert "service_prefix" in registry_data
        assert "page_url" in registry_data
        assert "version" in registry_data
        assert "actions" in registry_data
        assert "resource_types" in registry_data
        assert "service_condition_keys" in registry_data
        
        # Check service metadata
        assert registry_data["service_name"] == "Amazon S3"
        assert registry_data["service_prefix"] == "s3"
        assert registry_data["page_url"].startswith("https://")
        assert registry_data["version"]
        
        # Check actions structure
        if registry_data["actions"]:
            first_action = registry_data["actions"][0]
            required_action_fields = [
                "action", "access_level", "resource_types",
                "condition_keys", "dependent_actions", "source"
            ]
            for field in required_action_fields:
                assert field in first_action, f"Action missing field: {field}"
        
        # Check resource types structure
        if registry_data["resource_types"]:
            first_resource = registry_data["resource_types"][0]
            required_resource_fields = [
                "type", "arn_template", "condition_keys", "source"
            ]
            for field in required_resource_fields:
                assert field in first_resource, f"Resource missing field: {field}"
    
    def test_no_duplicate_actions(self, registry_data):
        """Ensure no duplicate action names."""
        action_names = [a["action"] for a in registry_data["actions"]]
        assert len(action_names) == len(set(action_names)), "Duplicate actions found"
    
    def test_no_duplicate_resources(self, registry_data):
        """Ensure no duplicate resource type names."""
        resource_names = [r["type"] for r in registry_data["resource_types"]]
        assert len(resource_names) == len(set(resource_names)), "Duplicate resources found"
    
    def test_actions_are_sorted(self, registry_data):
        """Verify actions are sorted alphabetically."""
        action_names = [a["action"] for a in registry_data["actions"]]
        assert action_names == sorted(action_names, key=str.lower), "Actions not sorted"
    
    def test_resources_are_sorted(self, registry_data):
        """Verify resource types are sorted alphabetically."""
        resource_names = [r["type"] for r in registry_data["resource_types"]]
        assert resource_names == sorted(resource_names, key=str.lower), "Resources not sorted"
    
    def test_condition_keys_are_sorted(self, registry_data):
        """Verify condition keys are sorted."""
        keys = registry_data["service_condition_keys"]
        assert keys == sorted(keys), "Condition keys not sorted"
    
    def test_resource_references_valid(self, registry_data):
        """Verify all resource references in actions are valid."""
        valid_resource_types = {r["type"] for r in registry_data["resource_types"]}
        
        for action in registry_data["actions"]:
            for resource_ref in action["resource_types"]:
                assert resource_ref["type"] in valid_resource_types, \
                    f"Action {action['action']} references unknown resource {resource_ref['type']}"
    
    def test_provenance_completeness(self, registry_data):
        """Verify all entries have complete provenance information."""
        # Check actions
        for action in registry_data["actions"]:
            assert "source" in action
            assert action["source"]["url"]
            assert action["source"]["table"] in ["actions", "resource_types", "condition_keys"]
            assert isinstance(action["source"]["row_index"], int)
            assert action["source"]["row_index"] >= 0
            
            # Check resource references within actions
            for resource in action["resource_types"]:
                assert "source" in resource
                assert resource["source"]["url"]
                assert resource["source"]["table"]
                assert isinstance(resource["source"]["row_index"], int)
        
        # Check resource types
        for resource in registry_data["resource_types"]:
            assert "source" in resource
            assert resource["source"]["url"]
            assert resource["source"]["table"]
            assert isinstance(resource["source"]["row_index"], int)
    
    def test_no_empty_values(self, registry_data):
        """Verify no critical fields have empty values."""
        # Actions should have names
        for action in registry_data["actions"]:
            assert action["action"], "Empty action name found"
            assert action["access_level"], "Empty access level found"
            
            # Resource types and condition keys can be empty lists
            assert isinstance(action["resource_types"], list)
            assert isinstance(action["condition_keys"], list)
            assert isinstance(action["dependent_actions"], list)
        
        # Resource types should have names and templates
        for resource in registry_data["resource_types"]:
            assert resource["type"], "Empty resource type found"
            # ARN template can be empty in edge cases, but shouldn't be for S3
            if resource["type"] in ["bucket", "object"]:
                assert resource["arn_template"], f"Empty ARN template for {resource['type']}"
    
    def test_access_level_values(self, registry_data):
        """Verify all access levels are valid."""
        valid_levels = {
            "Read", "Write", "List", "Tagging", "Permissions management"
        }
        
        for action in registry_data["actions"]:
            assert action["access_level"] in valid_levels, \
                f"Invalid access level '{action['access_level']}' for {action['action']}"
    
    def test_arn_template_format(self, registry_data):
        """Verify ARN templates follow expected format."""
        for resource in registry_data["resource_types"]:
            if resource["arn_template"]:
                arn = resource["arn_template"]
                
                # Should start with arn:
                assert arn.startswith("arn:"), f"Invalid ARN format for {resource['type']}"
                
                # Should contain partition placeholder
                assert "${Partition}" in arn or "${" in arn, \
                    f"ARN missing placeholders for {resource['type']}"
                
                # S3 ARNs should contain s3
                assert ":s3:" in arn or ":s3-" in arn, \
                    f"S3 ARN should contain service identifier for {resource['type']}"