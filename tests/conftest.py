import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test isolation."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_s3_reference_data():
    """Sample S3 reference JSON data matching AWS format."""
    return {
        "Version": "v1.3",
        "Name": "s3",
        "Actions": [
            {
                "Name": "PutObject",
                "ActionConditionKeys": ["s3:x-amz-acl", "s3:x-amz-content-sha256"],
                "Resources": [
                    {"Name": "object"},
                    {"Name": "accesspointobject"}
                ],
                "Annotations": {
                    "Properties": {
                        "IsWrite": True,
                        "IsList": False,
                        "IsPermissionManagement": False,
                        "IsTaggingOnly": False
                    }
                }
            },
            {
                "Name": "GetObject",
                "ActionConditionKeys": ["s3:ExistingObjectTag/<key>"],
                "Resources": [
                    {"Name": "object"},
                    {"Name": "accesspointobject"}
                ],
                "Annotations": {
                    "Properties": {
                        "IsWrite": False,
                        "IsList": False,
                        "IsPermissionManagement": False,
                        "IsTaggingOnly": False
                    }
                }
            },
            {
                "Name": "DeleteBucket",
                "ActionConditionKeys": [],
                "Resources": [
                    {"Name": "bucket"}
                ],
                "Annotations": {
                    "Properties": {
                        "IsWrite": True,
                        "IsList": False,
                        "IsPermissionManagement": False,
                        "IsTaggingOnly": False
                    }
                }
            },
            {
                "Name": "ListBucket",
                "ActionConditionKeys": ["s3:prefix"],
                "Resources": [
                    {"Name": "bucket"}
                ],
                "Annotations": {
                    "Properties": {
                        "IsWrite": False,
                        "IsList": True,
                        "IsPermissionManagement": False,
                        "IsTaggingOnly": False
                    }
                }
            },
            {
                "Name": "PutBucketTagging",
                "ActionConditionKeys": [],
                "Resources": [
                    {"Name": "bucket"}
                ],
                "Annotations": {
                    "Properties": {
                        "IsWrite": False,
                        "IsList": False,
                        "IsPermissionManagement": False,
                        "IsTaggingOnly": True
                    }
                }
            },
            {
                "Name": "PutBucketPolicy",
                "ActionConditionKeys": [],
                "Resources": [
                    {"Name": "bucket"}
                ],
                "Annotations": {
                    "Properties": {
                        "IsWrite": False,
                        "IsList": False,
                        "IsPermissionManagement": True,
                        "IsTaggingOnly": False
                    }
                }
            }
        ],
        "Resources": [
            {
                "Name": "bucket",
                "ARNFormats": [
                    "arn:${Partition}:s3:::${BucketName}"
                ]
            },
            {
                "Name": "object",
                "ARNFormats": [
                    "arn:${Partition}:s3:::${BucketName}/${ObjectName}",
                    "arn:${Partition}:s3:::${BucketName}/*"
                ]
            },
            {
                "Name": "accesspointobject",
                "ARNFormats": [
                    "arn:${Partition}:s3:${Region}:${Account}:accesspoint/${AccessPointName}/object/${ObjectName}"
                ]
            }
        ],
        "ConditionKeys": [
            {"Name": "s3:x-amz-acl"},
            {"Name": "s3:x-amz-content-sha256"},
            {"Name": "s3:ExistingObjectTag/<key>"},
            {"Name": "s3:prefix"},
            {"Name": "aws:RequestTag/${TagKey}"},
            {"Name": "aws:TagKeys"}
        ]
    }


@pytest.fixture
def malformed_json_data():
    """Malformed JSON data for error testing."""
    return '{"invalid": json, "data": }'


@pytest.fixture
def mock_requests_get(mock_s3_reference_data):
    """Mock requests.get for network tests."""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_s3_reference_data
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture
def isolated_fetch_module(temp_dir, monkeypatch):
    """Isolate fetch module to use temporary directory."""
    monkeypatch.setattr('src.fetch.s3_reference.RAW_DIR', temp_dir / 'data' / 'raw')
    (temp_dir / 'data' / 'raw').mkdir(parents=True, exist_ok=True)
    return temp_dir


@pytest.fixture
def isolated_parse_module(temp_dir, monkeypatch):
    """Isolate parse module to use temporary directory."""
    monkeypatch.setattr('src.parse.build_s3_registry_from_reference.OUT', 
                       temp_dir / 'data' / 'aws_iam_registry_s3.json')
    monkeypatch.setattr('src.parse.build_s3_registry_from_reference.REPORT',
                       temp_dir / 'data' / 'build_reports' / 's3_registry_report.json')
    (temp_dir / 'data').mkdir(exist_ok=True)
    (temp_dir / 'data' / 'build_reports').mkdir(exist_ok=True)
    return temp_dir