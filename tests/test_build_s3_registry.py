import json
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.parse.build_s3_registry_from_reference import (
    build_s3_registry_from_reference,
    _access_level_from_properties,
    _first_or_empty,
    to_jsonable,
    SourceRef,
    ResourceTypeRef,
    ActionRow,
    ResourceTypeDef,
    S3Registry
)


class TestAccessLevelMapping:
    """Tests for access level determination from properties."""
    
    def test_permissions_management_precedence(self):
        """Permissions management should have highest precedence."""
        props = {
            "IsPermissionManagement": True,
            "IsWrite": True,
            "IsTaggingOnly": True,
            "IsList": True
        }
        assert _access_level_from_properties(props) == "Permissions management"
    
    def test_write_precedence(self):
        """Write should have second precedence."""
        props = {
            "IsPermissionManagement": False,
            "IsWrite": True,
            "IsTaggingOnly": True,
            "IsList": True
        }
        assert _access_level_from_properties(props) == "Write"
    
    def test_tagging_precedence(self):
        """Tagging should have third precedence."""
        props = {
            "IsPermissionManagement": False,
            "IsWrite": False,
            "IsTaggingOnly": True,
            "IsList": True
        }
        assert _access_level_from_properties(props) == "Tagging"
    
    def test_list_precedence(self):
        """List should have fourth precedence."""
        props = {
            "IsPermissionManagement": False,
            "IsWrite": False,
            "IsTaggingOnly": False,
            "IsList": True
        }
        assert _access_level_from_properties(props) == "List"
    
    def test_default_read(self):
        """Default to Read when no flags are true."""
        props = {
            "IsPermissionManagement": False,
            "IsWrite": False,
            "IsTaggingOnly": False,
            "IsList": False
        }
        assert _access_level_from_properties(props) == "Read"
    
    def test_empty_properties(self):
        """Empty properties should default to Read."""
        assert _access_level_from_properties({}) == "Read"
    
    @pytest.mark.parametrize("props,expected", [
        ({"IsWrite": True}, "Write"),
        ({"IsList": True}, "List"),
        ({"IsTaggingOnly": True}, "Tagging"),
        ({"IsPermissionManagement": True}, "Permissions management"),
    ])
    def test_individual_flags(self, props, expected):
        """Test individual flags work correctly."""
        assert _access_level_from_properties(props) == expected


class TestHelperFunctions:
    """Tests for helper functions."""
    
    def test_first_or_empty_with_items(self):
        """Should return first item from non-empty list."""
        assert _first_or_empty(["first", "second", "third"]) == "first"
    
    def test_first_or_empty_with_empty_list(self):
        """Should return empty string for empty list."""
        assert _first_or_empty([]) == ""
    
    def test_first_or_empty_with_single_item(self):
        """Should work with single-item list."""
        assert _first_or_empty(["only"]) == "only"


class TestDataclassConversion:
    """Tests for dataclass to JSON conversion."""
    
    def test_source_ref_to_jsonable(self):
        """SourceRef should convert to dict correctly."""
        source = SourceRef(url="http://example.com", table="actions", row_index=5)
        result = to_jsonable(source)
        assert result == {
            "url": "http://example.com",
            "table": "actions",
            "row_index": 5
        }
    
    def test_nested_dataclass_conversion(self):
        """Nested dataclasses should convert correctly."""
        source = SourceRef(url="http://test.com", table="test", row_index=0)
        resource = ResourceTypeRef(type="bucket", required=False, source=source)
        result = to_jsonable(resource)
        assert result == {
            "type": "bucket",
            "required": False,
            "source": {
                "url": "http://test.com",
                "table": "test",
                "row_index": 0
            }
        }
    
    def test_list_conversion(self):
        """Lists of dataclasses should convert correctly."""
        sources = [
            SourceRef(url="url1", table="t1", row_index=1),
            SourceRef(url="url2", table="t2", row_index=2)
        ]
        result = to_jsonable(sources)
        assert len(result) == 2
        assert result[0]["url"] == "url1"
        assert result[1]["row_index"] == 2


class TestBuildS3Registry:
    """Tests for main build function."""
    
    def test_build_with_mock_data(self, isolated_parse_module, mock_s3_reference_data):
        """Test building registry with mock data."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            report = build_s3_registry_from_reference()
            
            # Verify report structure
            assert report["service"] == "s3"
            assert report["source"] == "aws-service-reference"
            assert report["version"] == "v1.3"
            assert report["actions_count"] == 6  # Based on mock data
            assert report["resource_types_count"] == 3
            assert report["condition_keys_count"] == 6
            
            # Verify output file was created
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            assert output_path.exists()
            
            # Load and verify output content
            with open(output_path) as f:
                registry_data = json.load(f)
            
            assert "s3" in registry_data
            s3_data = registry_data["s3"]
            assert s3_data["service_name"] == "Amazon S3"
            assert s3_data["service_prefix"] == "s3"
            assert s3_data["version"] == "v1.3"
    
    def test_action_parsing(self, isolated_parse_module, mock_s3_reference_data):
        """Test correct parsing of actions."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            actions = registry_data["s3"]["actions"]
            
            # Find specific actions to verify
            put_object = next(a for a in actions if a["action"] == "PutObject")
            assert put_object["access_level"] == "Write"
            assert len(put_object["resource_types"]) == 2
            assert any(r["type"] == "object" for r in put_object["resource_types"])
            assert "s3:x-amz-acl" in put_object["condition_keys"]
            
            get_object = next(a for a in actions if a["action"] == "GetObject")
            assert get_object["access_level"] == "Read"
            
            list_bucket = next(a for a in actions if a["action"] == "ListBucket")
            assert list_bucket["access_level"] == "List"
            
            put_bucket_policy = next(a for a in actions if a["action"] == "PutBucketPolicy")
            assert put_bucket_policy["access_level"] == "Permissions management"
            
            put_bucket_tagging = next(a for a in actions if a["action"] == "PutBucketTagging")
            assert put_bucket_tagging["access_level"] == "Tagging"
    
    def test_resource_type_parsing(self, isolated_parse_module, mock_s3_reference_data):
        """Test correct parsing of resource types."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            resource_types = registry_data["s3"]["resource_types"]
            
            # Verify bucket resource
            bucket = next(r for r in resource_types if r["type"] == "bucket")
            assert bucket["arn_template"] == "arn:${Partition}:s3:::${BucketName}"
            
            # Verify object resource (should use first ARN format)
            obj = next(r for r in resource_types if r["type"] == "object")
            assert obj["arn_template"] == "arn:${Partition}:s3:::${BucketName}/${ObjectName}"
    
    def test_condition_keys_parsing(self, isolated_parse_module, mock_s3_reference_data):
        """Test correct parsing of condition keys."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            condition_keys = registry_data["s3"]["service_condition_keys"]
            
            # Verify both S3 and AWS keys are included
            assert "s3:x-amz-acl" in condition_keys
            assert "s3:prefix" in condition_keys
            assert "aws:RequestTag/${TagKey}" in condition_keys
            assert "aws:TagKeys" in condition_keys
    
    def test_sorting(self, isolated_parse_module, mock_s3_reference_data):
        """Test that outputs are sorted alphabetically."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            actions = registry_data["s3"]["actions"]
            action_names = [a["action"] for a in actions]
            assert action_names == sorted(action_names, key=str.lower)
            
            resource_types = registry_data["s3"]["resource_types"]
            resource_names = [r["type"] for r in resource_types]
            assert resource_names == sorted(resource_names, key=str.lower)
            
            condition_keys = registry_data["s3"]["service_condition_keys"]
            assert condition_keys == sorted(condition_keys)
    
    def test_provenance_tracking(self, isolated_parse_module, mock_s3_reference_data):
        """Test that provenance information is correctly included."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            # Check action provenance
            first_action = registry_data["s3"]["actions"][0]
            assert "source" in first_action
            assert first_action["source"]["url"] == "https://servicereference.us-east-1.amazonaws.com/v1/s3/s3.json"
            assert first_action["source"]["table"] == "actions"
            assert isinstance(first_action["source"]["row_index"], int)
            
            # Check resource provenance
            if first_action["resource_types"]:
                first_resource = first_action["resource_types"][0]
                assert "source" in first_resource
                assert first_resource["source"]["table"] == "actions"
    
    def test_force_flag(self, isolated_parse_module, mock_s3_reference_data):
        """Test that force flag is passed through to fetch."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            build_s3_registry_from_reference(force=True)
            
            mock_fetch.assert_called_once_with(force=True)
    
    def test_report_file_creation(self, isolated_parse_module, mock_s3_reference_data):
        """Test that build report is created correctly."""
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = mock_s3_reference_data
            
            report = build_s3_registry_from_reference()
            
            report_path = isolated_parse_module / "data" / "build_reports" / "s3_registry_report.json"
            assert report_path.exists()
            
            with open(report_path) as f:
                saved_report = json.load(f)
            
            assert saved_report == report
    
    def test_empty_resources_handling(self, isolated_parse_module):
        """Test handling when action has no resources."""
        test_data = {
            "Version": "v1.0",
            "Actions": [
                {
                    "Name": "TestAction",
                    "ActionConditionKeys": [],
                    "Resources": [],  # Empty resources
                    "Annotations": {"Properties": {}}
                }
            ],
            "Resources": [],
            "ConditionKeys": []
        }
        
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = test_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            action = registry_data["s3"]["actions"][0]
            assert action["resource_types"] == []
    
    def test_missing_annotations_handling(self, isolated_parse_module):
        """Test handling when action has no annotations."""
        test_data = {
            "Version": "v1.0",
            "Actions": [
                {
                    "Name": "TestAction",
                    "ActionConditionKeys": [],
                    "Resources": [],
                    # Missing Annotations
                }
            ],
            "Resources": [],
            "ConditionKeys": []
        }
        
        with patch('src.parse.build_s3_registry_from_reference.fetch_s3_reference_json') as mock_fetch:
            mock_fetch.return_value = test_data
            
            build_s3_registry_from_reference()
            
            output_path = isolated_parse_module / "data" / "aws_iam_registry_s3.json"
            with open(output_path) as f:
                registry_data = json.load(f)
            
            action = registry_data["s3"]["actions"][0]
            assert action["access_level"] == "Read"  # Default