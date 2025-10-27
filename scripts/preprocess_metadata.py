#!/usr/bin/env python3
"""
Preprocess extracted metadata to handle duplicate document references and implement 
document-level metadata aggregation strategies.

This script processes the raw metadata extracted from XML files and creates clean,
document-level datasets suitable for machine learning training.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import re


def setup_logging(log_level: str        # Set the target based on target_field
        if target_field.startswith('derived_'):
            # Handle derived targets like 'derived_vtyp_simplified', 'derived_size_category'
            derived_field = target_field[8:]  # Remove 'derived_' prefix
            target_value = doc.get('derived_targets', {}).get(derived_field)
        else:
            # Handle original targets like 'primary_dokart'
            target_value = doc.get(target_field)
        
        # Only include if target value exists and is not None
        if target_value is not None and target_value != '':
            training_entry['target'] = target_value
            training_data.append(training_entry)
            target_distribution[target_value] += 1None:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Setup file and console logging
    log_file = log_dir / "preprocess_metadata.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_metadata(input_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all metadata from the input directory.
    
    Args:
        input_dir: Directory containing metadata JSON files
        
    Returns:
        List of all document metadata entries
    """
    # Try to load the combined metadata file first
    combined_file = input_dir / "all_metadata.json"
    
    if combined_file.exists():
        logging.info(f"Loading combined metadata from {combined_file}")
        try:
            with open(combined_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading combined metadata: {e}")
    
    # Fallback: load individual metadata files
    logging.info("Loading individual metadata files...")
    all_metadata = []
    
    metadata_files = sorted(input_dir.glob("*_metadata.json"))
    if not metadata_files:
        logging.error(f"No metadata files found in {input_dir}")
        return []
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_metadata.extend(data)
                logging.info(f"Loaded {len(data)} entries from {metadata_file.name}")
        except Exception as e:
            logging.error(f"Error loading {metadata_file}: {e}")
    
    return all_metadata


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent comparison.
    
    Args:
        url: Raw URL string
        
    Returns:
        Normalized URL string
    """
    if not url:
        return ""
    
    # Remove trailing whitespace and convert to lowercase
    url = url.strip().lower()
    
    # Normalize protocol
    if url.startswith('https://'):
        url = url[8:]
    elif url.startswith('http://'):
        url = url[7:]
    
    return url


def extract_file_extension(url: str) -> Optional[str]:
    """
    Extract file extension from URL.
    
    Args:
        url: Document URL
        
    Returns:
        File extension (without dot) or None
    """
    if not url:
        return None
    
    # Extract extension from URL
    match = re.search(r'\.([a-zA-Z0-9]+)$', url.lower())
    return match.group(1) if match else None


def select_primary_dokart(documents: List[Dict[str, Any]], dokart_frequencies: Dict[str, int]) -> str:
    """
    Select primary DokArt using hierarchical strategy:
    1. Majority vote among references
    2. If tie, use hierarchical priority based on global frequencies 
    3. If still tied, use first occurrence
    
    Args:
        documents: List of document metadata entries for the same URL
        dokart_frequencies: Global frequency distribution of DokArt types
        
    Returns:
        Selected primary DokArt
    """
    if not documents:
        return None
    
    # Extract DokArt values
    dokarts = [doc.get('dokument', {}).get('DokArt') for doc in documents]
    dokarts = [d for d in dokarts if d]  # Remove None values
    
    if not dokarts:
        return None
    
    # Strategy 1: Majority vote
    dokart_counts = Counter(dokarts)
    most_common = dokart_counts.most_common()
    
    # Check if there's a clear majority (single most frequent)
    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        return most_common[0][0]
    
    # Strategy 2: Hierarchical priority based on global frequencies
    # In case of tie, prefer the DokArt that's globally more common
    tied_dokarts = [dokart for dokart, count in most_common if count == most_common[0][1]]
    
    if len(tied_dokarts) > 1:
        # Sort by global frequency (higher frequency = higher priority)
        tied_dokarts.sort(key=lambda x: dokart_frequencies.get(x, 0), reverse=True)
        return tied_dokarts[0]
    
    # Strategy 3: First occurrence (fallback)
    return dokarts[0]


def calculate_dokart_frequencies(metadata_entries: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate global frequency distribution of DokArt types.
    
    Args:
        metadata_entries: All metadata entries
        
    Returns:
        Dictionary with DokArt frequencies
    """
    dokart_counts = Counter()
    
    for entry in metadata_entries:
        dokart = entry.get('dokument', {}).get('DokArt')
        if dokart:
            dokart_counts[dokart] += 1
    
    return dict(dokart_counts)


def create_derived_targets(doc_metadata: Dict[str, Any], vorgang_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create derived classification targets with better class balance.
    
    Args:
        doc_metadata: Document-level metadata
        vorgang_metadata: Vorgang-level metadata
        
    Returns:
        Dictionary with derived targets
    """
    derived = {}
    
    # 1. Wahlperiode (8 classes, good balance)
    derived['wahlperiode'] = doc_metadata.get('Wp')
    
    # 2. Document size categories (4 classes, excellent balance)
    su = doc_metadata.get('Su')  # Page count
    if su and su.replace(' ', '').replace(',', '').isdigit():
        try:
            # Handle formats like "3, 6" or "119" 
            pages = int(su.replace(' ', '').replace(',', '').split()[0])
            if pages <= 2:
                derived['size_category'] = 'Short'
            elif pages <= 10:
                derived['size_category'] = 'Medium'  
            elif pages <= 30:
                derived['size_category'] = 'Long'
            else:
                derived['size_category'] = 'Very Long'
        except (ValueError, IndexError):
            derived['size_category'] = None
    else:
        derived['size_category'] = None
    
    # 3. VTyp simplified (6 classes, remove tiny categories)
    vtyp = vorgang_metadata.get('VTyp')
    if vtyp in ['Beschlussempfehlung', 'Vorschrift']:  # <1% classes
        derived['vtyp_simplified'] = 'Other'
    else:
        derived['vtyp_simplified'] = vtyp
    
    # 4. Top subject areas (6 classes from VSysL)
    vsysl = vorgang_metadata.get('VSysL')
    top_subjects = {
        'Öffentlicher Haushalt', 'Parlament', 'Schulen', 
        'Kommunale Angelegenheiten', 'Wirtschaft'
    }
    if vsysl in top_subjects:
        derived['subject_area'] = vsysl
    elif vsysl:
        derived['subject_area'] = 'Other'
    else:
        derived['subject_area'] = None
    
    # 5. Document vs Procedure focus (binary classification)
    dokart = doc_metadata.get('DokArt') 
    if dokart in ['Drs', 'Inf', 'GVBl', 'Gutachten', 'Einl']:
        derived['focus_type'] = 'Document'
    elif dokart in ['PlPr', 'APr']:
        derived['focus_type'] = 'Procedure'
    else:
        derived['focus_type'] = 'Other'
        
    return derived


def aggregate_document_metadata(documents: List[Dict[str, Any]], dokart_frequencies: Dict[str, int]) -> Dict[str, Any]:
    """
    Aggregate metadata for multiple references to the same document.
    
    Args:
        documents: List of document metadata entries for the same URL
        
    Returns:
        Aggregated metadata for the document
    """
    if not documents:
        return {}
    
    # Start with the first document as base
    base_doc = documents[0]['dokument']
    base_vorgang = documents[0]['vorgang']
    
    # Select primary DokArt using improved strategy
    primary_dokart = select_primary_dokart(documents, dokart_frequencies)
    
    aggregated = {
        'source_files': list(set(doc.get('source_file', '') for doc in documents)),
        'extraction_timestamp': max(doc.get('extraction_timestamp', '') for doc in documents),
        'reference_count': len(documents),
        
        # Document-level metadata (using selected primary)
        'primary_dokart': primary_dokart,
        'primary_dokart_label': None,  # Will be set based on primary_dokart
        'primary_titel': base_doc.get('Titel'),
        'wahlperiode': base_doc.get('Wp'),
        'document_number': base_doc.get('DokNr'),
        'document_date': base_doc.get('DokDat'),
        'document_herkunft': base_doc.get('DHerk'),
        'document_typ': base_doc.get('DokTyp'),
        'pages': base_doc.get('Sb'),  # Seitenbereich (page range)
        'size': base_doc.get('Su'),   # Seitenumfang (page count)
        
        # URLs (all versions)
        'urls': [],
        'primary_url': None,
        'url_types': set(),
        
        # Aggregated fields (for analysis, not training features)
        'all_dokart': set(),
        'all_dokart_labels': set(),
        'all_titel': set(),
        'all_speakers': set(),
        'all_desk_categories': set(),
        'all_urheber': set(),
        'all_btext': set(),
        
        # Procedural context
        'vorgang_types': set(),
        'vorgang_numbers': set(),
        'nebeneintrage': [],
    }
    
    # Find the document entry that has the primary_dokart to get its label
    for doc in documents:
        dok_data = doc.get('dokument', {})
        if dok_data.get('DokArt') == primary_dokart and dok_data.get('DokArtL'):
            aggregated['primary_dokart_label'] = dok_data['DokArtL']
            break
    
    # Collect all URLs
    all_urls = set()
    for doc in documents:
        dok_data = doc.get('dokument', {})
        if dok_data.get('LokURLs'):
            all_urls.update(dok_data['LokURLs'])
        elif dok_data.get('LokURL'):
            all_urls.add(dok_data['LokURL'])
    
    aggregated['urls'] = sorted(all_urls)
    if aggregated['urls']:
        aggregated['primary_url'] = aggregated['urls'][0]
        aggregated['url_types'] = set(extract_file_extension(url) for url in aggregated['urls'])
        aggregated['url_types'].discard(None)  # Remove None values
    
    # Aggregate all metadata fields
    for doc in documents:
        dok_data = doc.get('dokument', {})
        vorgang_data = doc.get('vorgang', {})
        
        # Document metadata
        if dok_data.get('DokArt'):
            aggregated['all_dokart'].add(dok_data['DokArt'])
        if dok_data.get('DokArtL'):
            aggregated['all_dokart_labels'].add(dok_data['DokArtL'])
        if dok_data.get('Titel'):
            aggregated['all_titel'].add(dok_data['Titel'])
        if dok_data.get('Redner'):
            if isinstance(dok_data['Redner'], list):
                aggregated['all_speakers'].update(dok_data['Redner'])
            else:
                aggregated['all_speakers'].add(dok_data['Redner'])
        if dok_data.get('Desk'):
            aggregated['all_desk_categories'].add(dok_data['Desk'])
        if dok_data.get('Urheber'):
            aggregated['all_urheber'].add(dok_data['Urheber'])
        if dok_data.get('BText'):
            aggregated['all_btext'].add(dok_data['BText'])
        
        # Vorgang metadata
        if vorgang_data.get('VTyp'):
            aggregated['vorgang_types'].add(vorgang_data['VTyp'])
        if vorgang_data.get('VNr'):
            aggregated['vorgang_numbers'].add(vorgang_data['VNr'])
        if vorgang_data.get('Nebeneintrage'):
            aggregated['nebeneintrage'].extend(vorgang_data['Nebeneintrage'])
    
    # Convert sets to lists for JSON serialization
    for key in ['all_dokart', 'all_dokart_labels', 'all_titel', 'all_speakers', 
                'all_desk_categories', 'all_urheber', 'all_btext', 'vorgang_types', 
                'vorgang_numbers', 'url_types']:
        aggregated[key] = sorted(list(aggregated[key]))
    
    # Remove duplicates from nebeneintrage
    seen_nebeneintrage = set()
    unique_nebeneintrage = []
    for ne in aggregated['nebeneintrage']:
        ne_key = f"{ne.get('ReihNr', '')}_{ne.get('Desk', '')}"
        if ne_key not in seen_nebeneintrage:
            seen_nebeneintrage.add(ne_key)
            unique_nebeneintrage.append(ne)
    aggregated['nebeneintrage'] = unique_nebeneintrage
    
    # Add derived targets for better class balance
    derived_targets = create_derived_targets(base_doc, base_vorgang)
    aggregated['derived_targets'] = derived_targets
    
    return aggregated


def create_document_level_dataset(metadata_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create document-level dataset by aggregating multiple references to same documents.
    
    Args:
        metadata_entries: List of all metadata entries from XML parsing
        
    Returns:
        List of aggregated document-level entries
    """
    logging.info("Creating document-level dataset...")
    
    # Calculate global DokArt frequencies for hierarchical priority
    dokart_frequencies = calculate_dokart_frequencies(metadata_entries)
    logging.info(f"Global DokArt frequencies: {dokart_frequencies}")
    
    # Group entries by normalized URL
    url_to_entries = defaultdict(list)
    
    for entry in metadata_entries:
        dok_data = entry.get('dokument', {})
        
        # Get all URLs for this document
        urls = []
        if dok_data.get('LokURLs'):
            urls = dok_data['LokURLs']
        elif dok_data.get('LokURL'):
            urls = [dok_data['LokURL']]
        
        # Group by normalized primary URL
        if urls:
            primary_url = urls[0]
            normalized_url = normalize_url(primary_url)
            if normalized_url:
                url_to_entries[normalized_url].append(entry)
    
    logging.info(f"Found {len(url_to_entries)} unique documents (by URL)")
    
    # Create aggregated dataset
    document_dataset = []
    
    for normalized_url, entries in url_to_entries.items():
        aggregated = aggregate_document_metadata(entries, dokart_frequencies)
        aggregated['normalized_url'] = normalized_url
        document_dataset.append(aggregated)
    
    # Sort by wahlperiode and document number for consistent ordering
    document_dataset.sort(key=lambda x: (
        x.get('wahlperiode', '') or '',  # Handle None values
        x.get('document_number', '') or '',
        x.get('primary_url', '') or ''
    ))
    
    logging.info(f"Created document-level dataset with {len(document_dataset)} entries")
    
    return document_dataset


def analyze_duplicates(metadata_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze duplicate patterns in the metadata.
    
    Args:
        metadata_entries: List of all metadata entries
        
    Returns:
        Dictionary containing duplicate analysis
    """
    logging.info("Analyzing duplicate patterns...")
    
    url_to_count = defaultdict(int)
    url_to_dokart = defaultdict(set)
    url_to_speakers = defaultdict(set)
    
    for entry in metadata_entries:
        dok_data = entry.get('dokument', {})
        
        # Count URL occurrences
        if dok_data.get('LokURL'):
            url = normalize_url(dok_data['LokURL'])
            url_to_count[url] += 1
            
            if dok_data.get('DokArt'):
                url_to_dokart[url].add(dok_data['DokArt'])
            
            if dok_data.get('Redner'):
                if isinstance(dok_data['Redner'], list):
                    url_to_speakers[url].update(dok_data['Redner'])
                else:
                    url_to_speakers[url].add(dok_data['Redner'])
    
    # Find most duplicated documents
    most_duplicated = sorted(url_to_count.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Analyze consistency
    inconsistent_dokart = []
    for url, dokart_set in url_to_dokart.items():
        if len(dokart_set) > 1:
            inconsistent_dokart.append((url, list(dokart_set), url_to_count[url]))
    
    analysis = {
        'total_entries': len(metadata_entries),
        'unique_urls': len(url_to_count),
        'duplicate_ratio': 1 - (len(url_to_count) / len(metadata_entries)),
        'most_duplicated': most_duplicated[:10],
        'inconsistent_dokart_count': len(inconsistent_dokart),
        'inconsistent_dokart_examples': inconsistent_dokart[:5],
        'average_references_per_document': sum(url_to_count.values()) / len(url_to_count),
        'documents_with_multiple_speakers': len([url for url, speakers in url_to_speakers.items() if len(speakers) > 1])
    }
    
    return analysis


def generate_training_dataset(document_dataset: List[Dict[str, Any]], 
                            target_field: str = 'primary_dokart') -> Dict[str, Any]:
    """
    Generate a clean dataset suitable for machine learning training.
    
    This creates a text-to-metadata prediction dataset where:
    - Input: Document text content (extracted later from URLs)
    - Output: Target metadata field (e.g., DokArt)
    
    No metadata features are included since the model will learn to predict them.
    
    Args:
        document_dataset: Document-level aggregated dataset
        target_field: Target field for prediction (default: 'primary_dokart')
        
    Returns:
        Dictionary containing training dataset and metadata
    """
    logging.info(f"Generating training dataset for target: {target_field}")
    
    training_data = []
    target_distribution = Counter()
    
    for doc in document_dataset:
        # Determine target value based on target_field type
        if target_field.startswith('derived_'):
            # Handle derived targets like 'derived_vtyp_simplified', 'derived_size_category'
            derived_field = target_field[8:]  # Remove 'derived_' prefix
            target_value = doc.get('derived_targets', {}).get(derived_field)
        else:
            # Handle original targets like 'primary_dokart'
            target_value = doc.get(target_field)
        
        # Only include documents with URLs and target labels
        if not doc.get('primary_url') or target_value is None or target_value == '':
            continue
        
        # Create training entry - minimal, focused on text→target mapping
        training_entry = {
            # Document identification
            'document_id': doc.get('normalized_url', ''),
            'primary_url': doc.get('primary_url', ''),
            'urls': doc.get('urls', []),  # All format versions (PDF, DOCX, etc.)
            
            # Target variable (what we want to predict)
            'target': target_value,
            
            # Document source information (for text extraction pipeline)
            'wahlperiode': doc.get('wahlperiode'),
            'document_date': doc.get('document_date'),
            'source_files': doc.get('source_files', []),
            
            # Analysis metadata (not for training, for dataset analysis)
            'reference_count': doc.get('reference_count', 1),
            'url_types': doc.get('url_types', []),
            
            # Ground truth metadata (for future multi-task learning)
            # These are stored but NOT used as input features
            'ground_truth': {
                'dokart': doc.get('primary_dokart'),
                'dokart_label': doc.get('primary_dokart_label'),
                'titel': doc.get('primary_titel'),
                'speakers': doc.get('all_speakers', []),
                'desk_categories': doc.get('all_desk_categories', []),
                'procedural_texts': doc.get('all_btext', []),
            },
            
            # Derived targets with better class balance
            'derived_targets': doc.get('derived_targets', {})
        }
        
        training_data.append(training_entry)
        target_distribution[target_value] += 1
    
    # Generate dataset metadata
    dataset_metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'target_field': target_field,
        'total_documents': len(training_data),
        'target_distribution': dict(target_distribution),
        'unique_wahlperioden': len(set(doc.get('wahlperiode') for doc in training_data)),
        'documents_by_wahlperiode': dict(Counter(doc.get('wahlperiode') for doc in training_data)),
        'url_type_distribution': dict(Counter(
            url_type for doc in training_data 
            for url_type in doc.get('url_types', [])
        )),
        'average_references_per_document': sum(doc.get('reference_count', 1) for doc in training_data) / len(training_data) if training_data else 0,
        
        # Information about the training paradigm
        'training_paradigm': 'text_to_metadata',
        'input_description': 'Document text content (to be extracted from URLs)',
        'output_description': f'Target metadata field: {target_field}',
        'note': 'No metadata features included as inputs - model learns text→metadata mapping'
    }
    
    return {
        'training_data': training_data,
        'metadata': dataset_metadata
    }


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess extracted metadata for machine learning training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Process data/metadata_complete/ → data/metadata_preprocessed/
  %(prog)s --input custom/metadata --output custom/processed  # Custom directories
  %(prog)s --target-field primary_dokart     # Specify target field for training
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/metadata_complete',
        help='Directory containing extracted metadata (default: data/metadata_complete)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/metadata_preprocessed',
        help='Output directory for preprocessed metadata (default: data/metadata_preprocessed)'
    )
    
    parser.add_argument(
        '--target-field',
        type=str,
        default='derived_vtyp_simplified',
        choices=[
            'primary_dokart',           # Original (heavily imbalanced)
            'derived_vtyp_simplified',  # 6 classes, moderate balance (RECOMMENDED)
            'derived_size_category',    # 4 classes, excellent balance  
            'derived_subject_area',     # 6 classes, moderate balance
            'derived_focus_type',       # 3 classes, binary-like balance
            'derived_wahlperiode',      # 8 classes, temporal (not content-based)
        ],
        help='Target field for machine learning (default: derived_vtyp_simplified)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """Main function to orchestrate the metadata preprocessing."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Convert paths to Path objects
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    logging.info("=" * 60)
    logging.info("Parliamentary Metadata Preprocessor")
    logging.info("=" * 60)
    
    # Validate input directory
    if not input_dir.exists():
        logging.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    
    # Load metadata
    metadata_entries = load_metadata(input_dir)
    if not metadata_entries:
        logging.error("No metadata loaded. Exiting.")
        sys.exit(1)
    
    logging.info(f"Loaded {len(metadata_entries)} metadata entries")
    
    # Analyze duplicates
    duplicate_analysis = analyze_duplicates(metadata_entries)
    
    # Save duplicate analysis
    with open(output_dir / "duplicate_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(duplicate_analysis, f, ensure_ascii=False, indent=2)
    
    # Create document-level dataset
    document_dataset = create_document_level_dataset(metadata_entries)
    
    # Save document-level dataset
    with open(output_dir / "document_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(document_dataset, f, ensure_ascii=False, indent=2)
    
    # Generate training dataset
    training_dataset = generate_training_dataset(document_dataset, args.target_field)
    
    # Save training dataset
    with open(output_dir / "training_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(training_dataset, f, ensure_ascii=False, indent=2)
    
    # Log summary
    logging.info("=" * 60)
    logging.info("Preprocessing Summary:")
    logging.info(f"Input entries: {len(metadata_entries)}")
    logging.info(f"Unique documents: {len(document_dataset)}")
    logging.info(f"Training documents: {len(training_dataset['training_data'])}")
    logging.info(f"Duplicate ratio: {duplicate_analysis['duplicate_ratio']:.2%}")
    logging.info(f"Average references per document: {duplicate_analysis['average_references_per_document']:.1f}")
    
    logging.info(f"\nTarget field: {args.target_field}")
    logging.info("Target distribution:")
    for target, count in training_dataset['metadata']['target_distribution'].items():
        logging.info(f"  {target}: {count}")
    
    logging.info(f"\nFiles saved to: {output_dir.absolute()}")
    logging.info("  - duplicate_analysis.json: Analysis of duplicate patterns")
    logging.info("  - document_dataset.json: Aggregated document-level dataset")  
    logging.info("  - training_dataset.json: Clean dataset for ML training")


if __name__ == "__main__":
    main()