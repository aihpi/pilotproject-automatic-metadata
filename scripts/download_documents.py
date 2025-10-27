#!/usr/bin/env python3
"""
Script to download all PDF and DOCX documents from the parliamentary XML export.
Extracts URLs from LokURL elements and downloads files to data/documents folder.
"""

import xml.etree.ElementTree as ET
import requests
import os
import time
from urllib.parse import urlparse
from pathlib import Path
import re

def extract_urls_from_xml(xml_file_path):
    """Extract all unique URLs from LokURL elements in the XML file."""
    print(f"Parsing XML file: {xml_file_path}")
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        urls = set()
        for lok_url in root.iter('LokURL'):
            if lok_url.text:
                url = lok_url.text.strip()
                # Only include URLs that point to actual files
                if url.endswith(('.pdf', '.docx', '.doc', '.html', '.htm')):
                    urls.add(url)
        
        print(f"Found {len(urls)} unique valid URLs")
        return sorted(urls)
    
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []
    except FileNotFoundError:
        print(f"XML file not found: {xml_file_path}")
        return []

def create_filename_from_url(url):
    """Create a safe filename from the URL, preserving the document structure."""
    parsed = urlparse(url)
    
    # Extract the path without the domain
    path = parsed.path
    
    # Remove leading slash and replace remaining slashes with underscores
    # This preserves the directory structure in the filename
    filename = path.lstrip('/').replace('/', '_')
    
    # Ensure we have a valid filename
    if not filename:
        filename = "unknown_document"
    
    return filename

def download_file(url, output_dir, max_retries=3, delay=1):
    """Download a single file with retry logic."""
    # Skip invalid URLs (no actual file)
    if not url.endswith(('.pdf', '.docx', '.doc', '.html', '.htm')):
        print(f"✗ Skipping invalid URL (not a file): {url}")
        return False
        
    filename = create_filename_from_url(url)
    output_path = output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        print(f"✓ Already exists: {filename}")
        return True
    
    print(f"Downloading: {filename}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the file using streaming
            total_bytes = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
            
            print(f"✓ Downloaded: {filename} ({total_bytes} bytes)")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"✗ Failed to download: {filename}")
                return False
        
        except Exception as e:
            print(f"✗ Unexpected error downloading {filename}: {e}")
            return False
    
    return False

def main():
    """Main function to orchestrate the download process."""
    # Find all XML files in data/xml directory
    xml_dir = Path("data/xml")
    xml_files = sorted(xml_dir.glob("exportWP*.xml"))
    output_dir = Path("data/documents")
    
    if not xml_files:
        print(f"No XML files found in {xml_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files to process:")
    for xml_file in xml_files:
        print(f"  - {xml_file.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process all XML files
    all_urls = set()
    for xml_file in xml_files:
        print(f"\nProcessing {xml_file.name}...")
        urls = extract_urls_from_xml(xml_file)
        if urls:
            all_urls.update(urls)
            print(f"  Added {len(urls)} URLs from {xml_file.name}")
    
    if not all_urls:
        print("No URLs found in any XML files")
        return
    
    print(f"\nTotal unique URLs across all files: {len(all_urls)}")
    
    # Download files
    print(f"\nStarting download of {len(all_urls)} unique files...")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, url in enumerate(sorted(all_urls), 1):
        print(f"\n[{i}/{len(all_urls)}] ", end="")
        
        if download_file(url, output_dir):
            successful += 1
        else:
            failed += 1
        
        # Small delay to be respectful to the server
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"📁 Files saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()