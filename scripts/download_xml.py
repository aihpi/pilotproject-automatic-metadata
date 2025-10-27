#!/usr/bin/env python3
"""
Download XML files from Brandenburg parliament documentation.

This script downloads XML export files for specified Wahlperioden (election periods)
from the Brandenburg parliament documentation website.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Set
import requests
from urllib.parse import urljoin


def parse_wahlperiode_range(wp_string: str) -> Set[int]:
    """
    Parse wahlperiode string into a set of integers.
    
    Supports formats like:
    - "1-8" (range from 1 to 8)
    - "1,2,5-8" (individual numbers and ranges)
    - "3-5" (range from 3 to 5)
    - "1,3" (individual numbers)
    
    Args:
        wp_string: String representing wahlperiode selection
        
    Returns:
        Set of wahlperiode numbers
        
    Raises:
        ValueError: If the format is invalid
    """
    wp_set = set()
    
    # Split by comma to handle multiple parts
    parts = wp_string.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range (e.g., "1-8" or "5-7")
            try:
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                
                if start_num > end_num:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                
                wp_set.update(range(start_num, end_num + 1))
            except ValueError as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid range format: {part}")
                raise
        else:
            # Handle single number
            try:
                wp_set.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid wahlperiode number: {part}")
    
    return wp_set


def download_file(url: str, output_path: Path, filename: str) -> bool:
    """
    Download a file from URL to the specified path.
    
    Args:
        url: URL to download from
        output_path: Directory to save the file
        filename: Name of the file to save
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write file
        file_path = output_path / filename
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded {filename} ({len(response.content)} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error saving {filename}: {e}")
        return False


def main():
    """Main function to handle command line arguments and download files."""
    parser = argparse.ArgumentParser(
        description="Download XML files from Brandenburg parliament documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Download WP 1-8 to data/xml
  %(prog)s --wp 1,2,5-8             # Download WP 1, 2, 5, 6, 7, 8
  %(prog)s --wp 3-5 --output /tmp   # Download WP 3, 4, 5 to /tmp
  %(prog)s --wp 1,3                 # Download only WP 1 and 3
        """
    )
    
    parser.add_argument(
        '--wp', '--wahlperiode',
        type=str,
        default='1-8',
        help='Wahlperiode selection (default: 1-8). Examples: "1-8", "1,2,5-8", "3-5", "1,3"'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/xml',
        help='Output directory for downloaded XML files (default: data/xml)'
    )
    
    args = parser.parse_args()
    
    # Parse wahlperiode selection
    try:
        wp_numbers = parse_wahlperiode_range(args.wp)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate wahlperiode numbers (1-8 are available)
    invalid_wp = [wp for wp in wp_numbers if wp < 1 or wp > 8]
    if invalid_wp:
        print(f"Error: Invalid Wahlperiode numbers: {sorted(invalid_wp)}", file=sys.stderr)
        print("Available Wahlperioden: 1-8", file=sys.stderr)
        sys.exit(1)
    
    # Setup paths
    output_path = Path(args.output)
    base_url = "https://www.parlamentsdokumentation.brandenburg.de/opendata/"
    
    print(f"Downloading XML files for Wahlperioden: {sorted(wp_numbers)}")
    print(f"Output directory: {output_path.absolute()}")
    print()
    
    # Download files
    successful_downloads = 0
    total_downloads = len(wp_numbers)
    
    for wp in sorted(wp_numbers):
        filename = f"exportWP{wp}.xml"
        url = urljoin(base_url, filename)
        
        if download_file(url, output_path, filename):
            successful_downloads += 1
    
    print()
    print(f"Download complete: {successful_downloads}/{total_downloads} files downloaded successfully")
    
    if successful_downloads == 0:
        sys.exit(1)
    elif successful_downloads < total_downloads:
        sys.exit(2)  # Partial success
    else:
        print("All files downloaded successfully!")


if __name__ == "__main__":
    main()