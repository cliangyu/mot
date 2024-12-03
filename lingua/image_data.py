import json
from typing import Iterator, Dict, Any, List
import numpy as np
import linecache

def image_token_iterator(file_path: str, position: int = 0, block_size: int = 1, offset: int = 0) -> Iterator[Dict[str, Any]]:
    """
    Iterator over image tokens from a JSONL file.
    Each line contains {"image": "image_name.jpg", "tokens": [...]}
    Uses linecache to handle file positions correctly.
    """
    line_number = position + 1  # linecache is 1-indexed
    
    while True:
        # Skip lines according to block size and offset
        if offset > 0:
            line_number += offset
            offset = 0
        
        try:
            for _ in range(block_size):
                line = linecache.getline(file_path, line_number)
                if not line:  # End of file
                    line_number = 1  # Reset to start
                    line = linecache.getline(file_path, line_number)
                    if not line:  # Empty file
                        raise StopIteration
                
                data = json.loads(line)
                # Pass tokens directly - the ImageTokenizer will handle them
                yield {"tokens": data["tokens"]}, {
                    "file_path": file_path,
                    "position": line_number - 1,  # Convert back to 0-indexed
                    "block_size": block_size,
                    "offset": offset,
                }
                line_number += block_size
            
        except StopIteration:
            line_number = 1
            continue