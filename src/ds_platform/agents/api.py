import time
import requests
import pandas as pd
from typing import Dict, Any, Optional


class APIFetcher:
    """
    Agent for fetching data from external APIs with retry logic.
    """

    def fetch(
        self,
        base_url: str,
        headers: Optional[Dict[str, Any]] = None,
        retry_count: int = 2,
    ) -> pd.DataFrame:
        last_error = None

        for attempt in range(retry_count):
            try:
                response = requests.get(base_url, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()

                df = self._normalize_data(data)
                return df

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < retry_count - 1:
                    wait_time = (attempt + 1) * 1
                    time.sleep(wait_time)
                continue
            except Exception as e:
                print(f"Error fetching data: {e}")
                return pd.DataFrame()

        print(f"Failed after {retry_count} attempts: {last_error}")
        return pd.DataFrame()

    def _normalize_data(self, data: Any) -> pd.DataFrame:
        """Normalize API response to DataFrame."""
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ["data", "results", "items", "records"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
        return pd.DataFrame()
