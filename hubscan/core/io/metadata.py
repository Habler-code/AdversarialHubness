# Copyright 2025 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Metadata handling."""

from typing import Dict, List, Optional, Any
import json
import pandas as pd
import numpy as np
from pathlib import Path


class Metadata:
    """Document metadata container."""
    
    def __init__(self, data: Dict[str, List[Any]]):
        """
        Initialize metadata.
        
        Args:
            data: Dictionary mapping field names to lists of values
        """
        self.data = data
        self.num_docs = len(next(iter(data.values()))) if data else 0
        
        # Validate all fields have same length
        for field, values in data.items():
            if len(values) != self.num_docs:
                raise ValueError(f"Field {field} has length {len(values)}, expected {self.num_docs}")
    
    def get(self, field: str, default: Optional[Any] = None) -> List[Any]:
        """Get field values."""
        return self.data.get(field, [default] * self.num_docs)
    
    def get_field(self, field: str, idx: int) -> Any:
        """Get value for specific document index."""
        if field not in self.data:
            return None
        return self.data[field][idx]
    
    def has_field(self, field: str) -> bool:
        """Check if field exists."""
        return field in self.data
    
    def to_dict(self, idx: Optional[int] = None) -> Dict[str, Any]:
        """Convert to dictionary, optionally for specific index."""
        if idx is not None:
            return {field: values[idx] for field, values in self.data.items()}
        return self.data.copy()
    
    def filter(self, indices: np.ndarray) -> "Metadata":
        """Create filtered metadata for given indices."""
        filtered_data = {
            field: [values[i] for i in indices]
            for field, values in self.data.items()
        }
        return Metadata(filtered_data)


def load_metadata(path: str) -> Metadata:
    """Load metadata from file."""
    path_obj = Path(path)
    
    if path_obj.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
        # If it's a list of dicts, convert to dict of lists
        if isinstance(data, list):
            if not data:
                return Metadata({})
            keys = data[0].keys()
            metadata_dict = {key: [item[key] for item in data] for key in keys}
            return Metadata(metadata_dict)
        elif isinstance(data, dict):
            # Check if it's already dict of lists or dict of dicts
            if data and isinstance(next(iter(data.values())), list):
                return Metadata(data)
            else:
                # Single dict, wrap it
                return Metadata({k: [v] for k, v in data.items()})
    
    elif path_obj.suffix == ".jsonl":
        records = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        if not records:
            return Metadata({})
        keys = records[0].keys()
        metadata_dict = {key: [record[key] for record in records] for key in keys}
        return Metadata(metadata_dict)
    
    elif path_obj.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
        metadata_dict = {col: df[col].tolist() for col in df.columns}
        return Metadata(metadata_dict)
    
    else:
        raise ValueError(f"Unsupported metadata format: {path_obj.suffix}")

