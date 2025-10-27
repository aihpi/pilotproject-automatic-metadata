#!/usr/bin/env python3
"""
Parse XML files from Brandenburg parliament documentation and extract all metadata fields.

This script processes parliamentary XML export files and extracts structured metadata
for each document, including handling of multiple document references and nested elements.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup file and console logging
    log_file = log_dir / "parse_xml.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def extract_document_metadata(dokument_element: ET.Element) -> Dict[str, Any]:
    """
    Extract all metadata fields from a Dokument XML element.
    
    Args:
        dokument_element: XML element representing a document
        
    Returns:
        Dictionary containing all extracted metadata fields
    """
    metadata = {}
    
    # Define all possible metadata fields we want to extract
    simple_fields = [
        'ReihNr', 'DHerk', 'DHerkL', 'Wp', 'DokArt', 'DokArtL', 
        'DokTyp', 'DokTypL', 'Desk', 'Titel', 'DokNr', 'NrInTyp',
        'DokDat', 'Sb', 'Su', 'BText', 'Urheber', 'FundSt'
    ]
    
    # Extract simple text fields
    for field in simple_fields:
        element = dokument_element.find(field)
        metadata[field] = element.text.strip() if element is not None and element.text else None
    
    # Handle multiple LokURL elements (documents can have PDF + DOCX versions)
    lokurl_elements = dokument_element.findall('LokURL')
    if lokurl_elements:
        metadata['LokURLs'] = [url.text.strip() for url in lokurl_elements if url.text]
        # Keep primary URL for backward compatibility
        metadata['LokURL'] = metadata['LokURLs'][0] if metadata['LokURLs'] else None
    else:
        metadata['LokURLs'] = []
        metadata['LokURL'] = None
    
    # Handle multiple Redner elements (speakers)
    redner_elements = dokument_element.findall('Redner')
    if redner_elements:
        metadata['Redner'] = [redner.text.strip() for redner in redner_elements if redner.text]
    else:
        # Check for single Redner element
        redner_element = dokument_element.find('Redner')
        if redner_element is not None and redner_element.text:
            metadata['Redner'] = [redner_element.text.strip()]
        else:
            metadata['Redner'] = []
    
    # Extract additional structured information that might be present
    # (These fields may not be present in all documents)
    optional_fields = [
        'Betreff', 'Status', 'Fraktion', 'Typ', 'Kategorie', 
        'Sachgebiet', 'Schlagwort', 'Bemerkung'
    ]
    
    for field in optional_fields:
        element = dokument_element.find(field)
        metadata[field] = element.text.strip() if element is not None and element.text else None
    
    return metadata


def extract_vorgang_metadata(vorgang_element: ET.Element) -> Dict[str, Any]:
    """
    Extract metadata from a Vorgang (procedure) XML element.
    
    Args:
        vorgang_element: XML element representing a parliamentary procedure
        
    Returns:
        Dictionary containing procedure metadata
    """
    vorgang_metadata = {}
    
    # Extract Vorgang-level fields
    vorgang_fields = [
        'VNr', 'ReihNr', 'VTyp', 'VTypL', 'VSys', 'VSysL', 'VIR', 'VFunktion'
    ]
    
    for field in vorgang_fields:
        element = vorgang_element.find(field)
        vorgang_metadata[field] = element.text.strip() if element is not None and element.text else None
    
    # Extract Nebeneintrag elements (related entries)
    nebeneintrag_elements = vorgang_element.findall('Nebeneintrag')
    nebeneintrage = []
    for ne in nebeneintrag_elements:
        ne_data = {}
        reihnr_elem = ne.find('ReihNr')
        desk_elem = ne.find('Desk')
        
        if reihnr_elem is not None and reihnr_elem.text:
            ne_data['ReihNr'] = reihnr_elem.text.strip()
        if desk_elem is not None and desk_elem.text:
            ne_data['Desk'] = desk_elem.text.strip()
        
        if ne_data:  # Only add if we found some data
            nebeneintrage.append(ne_data)
    
    vorgang_metadata['Nebeneintrage'] = nebeneintrage
    
    return vorgang_metadata


def parse_xml_file(xml_file_path: Path) -> List[Dict[str, Any]]:
    """
    Parse a single XML file and extract all document metadata.
    
    Args:
        xml_file_path: Path to the XML file to parse
        
    Returns:
        List of dictionaries containing document metadata
    """
    logging.info(f"Parsing XML file: {xml_file_path}")
    
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        documents = []
        
        # Process each Vorgang (procedure)
        for vorgang in root.findall('Vorgang'):
            # Skip deletion entries
            vfunktion = vorgang.find('VFunktion')
            if vfunktion is not None and vfunktion.text == 'delete':
                continue
            
            # Extract Vorgang-level metadata
            vorgang_metadata = extract_vorgang_metadata(vorgang)
            
            # Process each Dokument within this Vorgang
            for dokument in vorgang.findall('Dokument'):
                doc_metadata = extract_document_metadata(dokument)
                
                # Combine Vorgang and Dokument metadata
                combined_metadata = {
                    'source_file': xml_file_path.name,
                    'extraction_timestamp': datetime.now().isoformat(),
                    'vorgang': vorgang_metadata,
                    'dokument': doc_metadata
                }
                
                # Only include documents that have at least a LokURL
                if doc_metadata.get('LokURL'):
                    documents.append(combined_metadata)
        
        logging.info(f"Extracted {len(documents)} documents from {xml_file_path.name}")
        return documents
        
    except ET.ParseError as e:
        logging.error(f"Error parsing XML file {xml_file_path}: {e}")
        return []
    except Exception as e:
        logging.error(f"Unexpected error processing {xml_file_path}: {e}")
        return []


def save_metadata(documents: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Save extracted metadata to a JSON file.
    
    Args:
        documents: List of document metadata dictionaries
        output_file: Path where to save the JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(documents)} documents to {output_file}")
    except Exception as e:
        logging.error(f"Error saving metadata to {output_file}: {e}")
        raise


def generate_statistics(all_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about the extracted metadata.
    
    Args:
        all_documents: List of all extracted document metadata
        
    Returns:
        Dictionary containing various statistics
    """
    stats = {
        'total_documents': len(all_documents),
        'documents_with_urls': 0,
        'unique_urls': set(),
        'dokart_distribution': {},
        'wahlperiode_distribution': {},
        'source_file_distribution': {},
        'documents_with_speakers': 0,
        'unique_speakers': set(),
        'documents_with_multiple_urls': 0
    }
    
    for doc in all_documents:
        dok_metadata = doc.get('dokument', {})
        
        # Count documents with URLs
        if dok_metadata.get('LokURL'):
            stats['documents_with_urls'] += 1
            
        # Track unique URLs
        if dok_metadata.get('LokURLs'):
            stats['unique_urls'].update(dok_metadata['LokURLs'])
            if len(dok_metadata['LokURLs']) > 1:
                stats['documents_with_multiple_urls'] += 1
        
        # DokArt distribution
        dokart = dok_metadata.get('DokArt')
        if dokart:
            stats['dokart_distribution'][dokart] = stats['dokart_distribution'].get(dokart, 0) + 1
        
        # Wahlperiode distribution
        wp = dok_metadata.get('Wp')
        if wp:
            stats['wahlperiode_distribution'][wp] = stats['wahlperiode_distribution'].get(wp, 0) + 1
        
        # Source file distribution
        source = doc.get('source_file')
        if source:
            stats['source_file_distribution'][source] = stats['source_file_distribution'].get(source, 0) + 1
        
        # Speaker statistics
        speakers = dok_metadata.get('Redner', [])
        if speakers:
            stats['documents_with_speakers'] += 1
            stats['unique_speakers'].update(speakers)
    
    # Convert sets to counts for JSON serialization
    stats['unique_urls_count'] = len(stats['unique_urls'])
    stats['unique_speakers_count'] = len(stats['unique_speakers'])
    del stats['unique_urls']
    del stats['unique_speakers']
    
    return stats


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract metadata from Brandenburg parliament XML files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Process data/xml/ → data/metadata_complete/
  %(prog)s --input custom/xml --output custom/metadata  # Custom directories
  %(prog)s --log-level DEBUG                  # Debug logging
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/xml',
        help='Directory containing XML files (default: data/xml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/metadata_complete',
        help='Output directory for extracted metadata (default: data/metadata_complete)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main function to orchestrate the metadata extraction process."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Convert paths to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    logging.info("=" * 60)
    logging.info("Parliamentary XML Metadata Extractor")
    logging.info("=" * 60)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Find XML files
    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        logging.error(f"No XML files found in {input_dir}")
        sys.exit(1)
    
    logging.info(f"Found {len(xml_files)} XML files:")
    for xml_file in xml_files:
        logging.info(f"  - {xml_file.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Process all XML files
    all_documents = []
    
    for xml_file in xml_files:
        documents = parse_xml_file(xml_file)
        all_documents.extend(documents)
        
        # Save individual file metadata
        output_file = output_dir / f"{xml_file.stem}_metadata.json"
        save_metadata(documents, output_file)
    
    # Save combined metadata
    combined_output = output_dir / "all_metadata.json"
    save_metadata(all_documents, combined_output)
    
    # Generate and save statistics
    stats = generate_statistics(all_documents)
    stats_output = output_dir / "metadata_statistics.json"
    
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Log summary
    logging.info("=" * 60)
    logging.info("Extraction Summary:")
    logging.info(f"Total documents processed: {stats['total_documents']}")
    logging.info(f"Documents with URLs: {stats['documents_with_urls']}")
    logging.info(f"Unique URLs: {stats['unique_urls_count']}")
    logging.info(f"Documents with multiple URLs: {stats['documents_with_multiple_urls']}")
    logging.info(f"Documents with speakers: {stats['documents_with_speakers']}")
    logging.info(f"Unique speakers: {stats['unique_speakers_count']}")
    
    logging.info("\nDocument types (DokArt) distribution:")
    for dokart, count in sorted(stats['dokart_distribution'].items()):
        logging.info(f"  {dokart}: {count}")
    
    logging.info(f"\nFiles saved to: {output_dir.absolute()}")
    logging.info("  - Individual metadata files: *_metadata.json")
    logging.info("  - Combined metadata: all_metadata.json")
    logging.info("  - Statistics: metadata_statistics.json")


if __name__ == "__main__":
    main()