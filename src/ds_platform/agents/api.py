import requests
import pandas as pd
from typing import Dict, Any, Optional

class APIFetcher:
    """
    Agent for fetching data from external APIs and normalizing to DataFrames.
    """
    def fetch(self, base_url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Simple normalization (assumes list of dicts)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Look for a common data key
                for key in ['data', 'results', 'items']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame()
                
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()
