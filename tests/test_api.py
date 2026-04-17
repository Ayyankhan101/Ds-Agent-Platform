import pandas as pd
from unittest.mock import patch, MagicMock
from ds_platform.agents.api import APIFetcher


class TestAPIFetcher:
    def setup_method(self):
        self.agent = APIFetcher()

    @patch("ds_platform.agents.api.requests.get")
    def test_fetch_returns_dataframe_from_list(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.agent.fetch("http://api.example.com/data")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "name" in result.columns
        mock_get.assert_called_once()

    @patch("ds_platform.agents.api.requests.get")
    def test_fetch_handles_dict_with_data_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": 1, "value": 100}, {"id": 2, "value": 200}]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.agent.fetch("http://api.example.com/data")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("ds_platform.agents.api.requests.get")
    def test_fetch_handles_dict_with_results_key(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"id": 1}, {"id": 2}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.agent.fetch("http://api.example.com/data")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    @patch("ds_platform.agents.api.requests.get")
    def test_fetch_handles_single_dict(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "ok", "count": 1}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = self.agent.fetch("http://api.example.com/status")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "status" in result.columns

    @patch("ds_platform.agents.api.requests.get")
    def test_fetch_with_headers(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        headers = {"Authorization": "Bearer token123"}
        self.agent.fetch("http://api.example.com/data", headers=headers)

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["headers"] == headers

    @patch("ds_platform.agents.api.requests.get")
    def test_fetch_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        result = self.agent.fetch("http://api.example.com/data")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
