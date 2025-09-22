import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests

from src.fetch.s3_reference import (
    fetch_s3_reference_json,
    _cache_path,
    S3_REF_URL,
    HEADERS
)


class TestCachePath:
    """Tests for cache path generation."""
    
    def test_cache_path_deterministic(self):
        """Cache path should be deterministic for same URL."""
        path1 = _cache_path(S3_REF_URL)
        path2 = _cache_path(S3_REF_URL)
        assert path1 == path2
    
    def test_cache_path_different_urls(self):
        """Different URLs should produce different cache paths."""
        path1 = _cache_path("https://example.com/api1")
        path2 = _cache_path("https://example.com/api2")
        assert path1 != path2
    
    def test_cache_path_format(self):
        """Cache path should follow expected format."""
        path = _cache_path(S3_REF_URL)
        assert path.suffix == '.json'
        assert path.parent.name == 'raw'


class TestFetchS3Reference:
    """Tests for fetch_s3_reference_json function."""
    
    def test_fetch_creates_cache(self, isolated_fetch_module, mock_requests_get, mock_s3_reference_data):
        """First fetch should create cache file."""
        result = fetch_s3_reference_json()
        
        # Check result matches mock data
        assert result == mock_s3_reference_data
        
        # Check cache file was created
        cache_file = _cache_path(S3_REF_URL)
        assert cache_file.exists()
        
        # Verify cache content
        cached_data = json.loads(cache_file.read_text())
        assert cached_data == mock_s3_reference_data
    
    def test_fetch_uses_cache(self, isolated_fetch_module, mock_requests_get, mock_s3_reference_data):
        """Second fetch should use cache, not network."""
        # First call creates cache
        fetch_s3_reference_json()
        mock_requests_get.reset_mock()
        
        # Second call should use cache
        result = fetch_s3_reference_json()
        
        # Verify network was not called
        mock_requests_get.assert_not_called()
        assert result == mock_s3_reference_data
    
    def test_force_refresh_bypasses_cache(self, isolated_fetch_module, mock_requests_get, mock_s3_reference_data):
        """Force flag should bypass cache and fetch fresh data."""
        # Create initial cache
        fetch_s3_reference_json()
        mock_requests_get.reset_mock()
        
        # Force refresh
        result = fetch_s3_reference_json(force=True)
        
        # Verify network was called despite cache
        mock_requests_get.assert_called_once()
        assert result == mock_s3_reference_data
    
    def test_correct_headers_sent(self, isolated_fetch_module):
        """Verify correct headers are sent with request."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            fetch_s3_reference_json()
            
            mock_get.assert_called_with(
                S3_REF_URL,
                headers=HEADERS,
                timeout=30
            )
    
    def test_custom_timeout(self, isolated_fetch_module, mock_requests_get):
        """Test custom timeout parameter."""
        fetch_s3_reference_json(timeout=60)
        
        mock_requests_get.assert_called_with(
            S3_REF_URL,
            headers=HEADERS,
            timeout=60
        )
    
    def test_http_error_handling(self, isolated_fetch_module):
        """Test handling of HTTP errors."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
            mock_get.return_value = mock_response
            
            with pytest.raises(requests.HTTPError):
                fetch_s3_reference_json()
    
    def test_json_decode_error(self, isolated_fetch_module):
        """Test handling of invalid JSON response."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
            mock_get.return_value = mock_response
            
            with pytest.raises(json.JSONDecodeError):
                fetch_s3_reference_json()
    
    def test_network_timeout(self, isolated_fetch_module):
        """Test handling of network timeout."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.Timeout("Connection timed out")
            
            with pytest.raises(requests.Timeout):
                fetch_s3_reference_json(timeout=1)
    
    def test_cache_with_unicode_handling(self, isolated_fetch_module):
        """Test cache handles unicode data correctly."""
        test_data = {
            "Version": "v1.0",
            "unicode_field": "日本語テスト",
            "emoji": "🚀"
        }
        
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = test_data
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # Fetch and cache
            result1 = fetch_s3_reference_json()
            
            # Read from cache
            result2 = fetch_s3_reference_json()
            
            assert result1 == test_data
            assert result2 == test_data
            assert result1["unicode_field"] == "日本語テスト"
    
    def test_empty_response_handling(self, isolated_fetch_module):
        """Test handling of empty response."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            result = fetch_s3_reference_json()
            assert result == {}


class TestDirectoryCreation:
    """Tests for automatic directory creation."""
    
    def test_raw_dir_created_on_import(self):
        """RAW_DIR should be created when module is imported."""
        # This is already tested implicitly, but let's be explicit
        from src.fetch import s3_reference
        assert s3_reference.RAW_DIR.exists()
        assert s3_reference.RAW_DIR.is_dir()