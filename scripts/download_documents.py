#!/usr/bin/env python3
"""
Script to download all PDF and DOCX documents from the parliamentary XML export.
Extracts URLs from LokURL elements and downloads files to data/documents folder.

Optimized version with parallel processing, progress tracking, and resume capability.
"""

import xml.etree.ElementTree as ET
import requests
import os
import time
import argparse
import sys
from urllib.parse import urlparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import logging
from typing import List, Set, Optional

def extract_urls_from_xml(xml_file_path: Path) -> List[str]:
    """Extract all unique URLs from LokURL elements in the XML file."""
    logging.info(f"Parsing XML file: {xml_file_path}")
    
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
        
        logging.info(f"Found {len(urls)} unique valid URLs in {xml_file_path.name}")
        return sorted(urls)
    
    except ET.ParseError as e:
        logging.error(f"Error parsing XML {xml_file_path}: {e}")
        return []
    except FileNotFoundError:
        logging.error(f"XML file not found: {xml_file_path}")
        return []


def create_filename_from_url(url: str) -> str:
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

def download_file(url: str, output_dir: Path, max_retries: int = 3, delay: float = 1.0, 
                 progress_lock: Optional[Lock] = None, progress_counter: Optional[dict] = None) -> bool:
    """Download a single file with retry logic and thread-safe progress tracking."""
    # Skip invalid URLs (no actual file)
    if not url.endswith(('.pdf', '.docx', '.doc', '.html', '.htm')):
        logging.warning(f"Skipping invalid URL (not a file): {url}")
        return False
        
    filename = create_filename_from_url(url)
    output_path = output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        logging.debug(f"Already exists: {filename}")
        if progress_lock and progress_counter is not None:
            with progress_lock:
                progress_counter['skipped'] += 1
        return True
    
    logging.debug(f"Downloading: {filename}")
    
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
            
            logging.info(f"Downloaded: {filename} ({total_bytes:,} bytes)")
            if progress_lock and progress_counter is not None:
                with progress_lock:
                    progress_counter['successful'] += 1
                    progress_counter['total_bytes'] += total_bytes
            return True
            
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if attempt < max_retries - 1:
                logging.debug(f"Retrying {filename} in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed to download: {filename}")
                if progress_lock and progress_counter is not None:
                    with progress_lock:
                        progress_counter['failed'] += 1
                return False
        
        except Exception as e:
            logging.error(f"Unexpected error downloading {filename}: {e}")
            if progress_lock and progress_counter is not None:
                with progress_lock:
                    progress_counter['failed'] += 1
            return False
    
    return False


def download_files_parallel(urls: List[str], output_dir: Path, max_workers: int = 8, 
                          max_retries: int = 3, delay: float = 0.5) -> dict:
    """Download files in parallel with progress tracking."""
    progress_lock = Lock()
    progress_counter = {
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'total_bytes': 0
    }
    
    logging.info(f"Starting parallel download of {len(urls)} files with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_url = {
            executor.submit(download_file, url, output_dir, max_retries, delay, 
                          progress_lock, progress_counter): url 
            for url in urls
        }
        
        # Process completed downloads
        for i, future in enumerate(as_completed(future_to_url), 1):
            url = future_to_url[future]
            try:
                success = future.result()
                
                # Progress logging every 50 downloads
                if i % 50 == 0:
                    with progress_lock:
                        total_processed = progress_counter['successful'] + progress_counter['failed'] + progress_counter['skipped']
                        logging.info(f"Progress: {total_processed}/{len(urls)} files processed "
                                   f"(✓{progress_counter['successful']} ✗{progress_counter['failed']} "
                                   f"↷{progress_counter['skipped']})")
                        
            except Exception as e:
                logging.error(f"Error processing download for {url}: {e}")
                with progress_lock:
                    progress_counter['failed'] += 1
    
    return progress_counter


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup file and console logging
    log_file = log_dir / "download_documents.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download documents from Brandenburg parliament XML exports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Download all with defaults
  %(prog)s --xml-dir custom/xml --workers 16  # Custom XML dir, 16 workers
  %(prog)s --output-dir /tmp/docs --dry-run   # Dry run to different location
  %(prog)s --log-level DEBUG --retries 5     # Debug logging, 5 retries
        """
    )
    
    parser.add_argument(
        '--xml-dir',
        type=str,
        default='data/xml',
        help='Directory containing XML files (default: data/xml)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/documents',
        help='Output directory for downloaded documents (default: data/documents)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of parallel download workers (default: 8)'
    )
    
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='Maximum number of download retries (default: 3)'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between downloads in seconds (default: 0.5)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be downloaded without actually downloading'
    )
    
    return parser.parse_args()


def main():
    """Main function to orchestrate the download process."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Convert paths to Path objects
    xml_dir = Path(args.xml_dir)
    output_dir = Path(args.output_dir)
    
    logging.info("=" * 60)
    logging.info("Parliamentary Documents Downloader")
    logging.info("=" * 60)
    
    # Find all XML files
    xml_files = sorted(xml_dir.glob("exportWP*.xml"))
    
    if not xml_files:
        logging.error(f"No XML files found in {xml_dir}")
        sys.exit(1)
    
    logging.info(f"Found {len(xml_files)} XML files to process:")
    for xml_file in xml_files:
        logging.info(f"  - {xml_file.name}")
    
    # Create output directory
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Process all XML files
    all_urls = set()
    for xml_file in xml_files:
        logging.info(f"Processing {xml_file.name}...")
        urls = extract_urls_from_xml(xml_file)
        if urls:
            all_urls.update(urls)
    
    if not all_urls:
        logging.error("No URLs found in any XML files")
        sys.exit(1)
    
    logging.info(f"Total unique URLs across all files: {len(all_urls)}")
    
    # Show configuration
    logging.info("Configuration:")
    logging.info(f"  XML directory: {xml_dir}")
    logging.info(f"  Output directory: {output_dir}")
    logging.info(f"  Workers: {args.workers}")
    logging.info(f"  Max retries: {args.retries}")
    logging.info(f"  Delay: {args.delay}s")
    logging.info(f"  Log level: {args.log_level}")
    logging.info(f"  Dry run: {args.dry_run}")
    
    if args.dry_run:
        logging.info("DRY RUN: Would download the following files:")
        for i, url in enumerate(sorted(all_urls), 1):
            filename = create_filename_from_url(url)
            logging.info(f"  {i:4d}. {filename}")
        logging.info(f"Total files to download: {len(all_urls)}")
        return
    
    # Download files
    logging.info("=" * 60)
    logging.info(f"Starting download of {len(all_urls)} unique files...")
    start_time = time.time()
    
    progress = download_files_parallel(
        sorted(all_urls), 
        output_dir, 
        max_workers=args.workers,
        max_retries=args.retries,
        delay=args.delay
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary
    logging.info("=" * 60)
    logging.info("Download Summary:")
    logging.info(f"✓ Successful: {progress['successful']}")
    logging.info(f"✗ Failed: {progress['failed']}")
    logging.info(f"↷ Skipped (already exists): {progress['skipped']}")
    logging.info(f"📁 Total data downloaded: {progress['total_bytes']:,} bytes ({progress['total_bytes']/1024/1024:.1f} MB)")
    logging.info(f"⏱️  Total time: {duration:.1f} seconds")
    
    if progress['successful'] > 0:
        avg_speed = progress['successful'] / duration
        logging.info(f"🚀 Average speed: {avg_speed:.2f} files/second")
    
    logging.info(f"� Files saved to: {output_dir.absolute()}")
    
    # Exit with appropriate code
    if progress['failed'] > 0:
        if progress['successful'] == 0:
            logging.error("All downloads failed!")
            sys.exit(1)
        else:
            logging.warning("Some downloads failed!")
            sys.exit(2)  # Partial success
    else:
        logging.info("All downloads completed successfully!")


if __name__ == "__main__":
    main()