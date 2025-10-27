# Automatic Metadata Extraction from Parliamentary Documents

## Table of Contents

1. [Requirements](#1-requirements)
   - 1.1. [Virtual Environment](#11-virtual-environment)
   - 1.2. [Working Environment](#12-working-environment)
2. [Scripts](#2-scripts)
3. [Data](#3-data)
   - 3.1. [Download XML](#31-download-xml)
   - 3.2. [Download Documents](#32-download-documents)
   - 3.3. [Parse XML to Extract Metadata](#33-parse-xml-metadata)
   - 3.4. [Preprocess Metadata](#34-preprocess-metadata)
5. [Roadmap](#5-project-roadmap)

## 1. Requirements

### 1.1. Virtual Environment

To ensure that the same requirements are met across different operating systems
and machines, it is recommended to create a virtual environment. This can be set
up with UV.

```bash
which uv || echo "UV not found" # checks the UV installation
```

If UV is not installed, it can be installed as follows.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Afterwards, the virtual environment can be created and activated.

```bash
uv venv .venv # creates a virtual environment with the name ".venv"
source .venv/bin/activate # activates the virtual environment
```

Then the required packages are installed. UV ensures that the exact versions are
installed.

```bash
uv sync --all-extras  # installs exact versions
```

### 1.2. Working Environment

Before running any scripts on an HPC cluster, you need to configure your personal working
directory:

1. Copy the environment template:
   ```bash
   cp .env.local.template .env.local
   ```

2. Edit `.env.local` to set your working directory:
   ```bash
   # Open in your preferred editor
   nano .env.local
   # or
   vim .env.local
   ```

3. Update the `PROJECT_ROOT` variable to point to your personal working directory:
   ```bash
   # Example for user "john.doe":
   PROJECT_ROOT=/sc/home/john.doe/pilotproject-automatic-metadata
   
   # Example for different mount point:
   PROJECT_ROOT=/home/username/projects/pilotproject-automatic-metadata
   ```

4. Verify your configuration:
   ```bash
   source .env.local
   echo "Project root: $PROJECT_ROOT"
   ```

Note: The `.env.local` file is ignored by git, so your personal configuration won't be committed to
the repository.

## 2. Scripts

All scripts are located in the scripts folder.

## 3. Data

### 3.1. Download XML

The `scripts/download_xml.py` script downloads XML export files from the Brandenburg parliament documentation website.

#### Usage

```bash
# Basic usage - downloads all WP 1-8 to data/xml/
python scripts/download_xml.py

# Custom wahlperiode selection
python scripts/download_xml.py --wp 1,2,5-8

# Custom output directory
python scripts/download_xml.py --wp 3-5 --output /tmp/xml
```

The script will:
- Download specified `exportWP*.xml` files from https://www.parlamentsdokumentation.brandenburg.de/portal/opendata.tt.html
- Save XMLs to specified output directory (default: `data/xml/`)

### 3.2. Download Documents

The `scripts/download_documents.py` script extracts URLs from parliamentary XML exports and downloads all referenced PDF and DOCX documents. The script has been optimized with parallel processing for faster downloads.

#### Usage

```bash
# Basic usage - downloads all documents from XML files in data/xml/
python scripts/download_documents.py

# With custom parameters for faster processing
python scripts/download_documents.py --workers 16 --retries 5

# Dry run to see what would be downloaded
python scripts/download_documents.py --dry-run

# Custom directories and logging
python scripts/download_documents.py --xml-dir custom/xml --output-dir custom/docs --log-level DEBUG
```

For large-scale downloads on HPC clusters, use the SLURM batch script:

```bash
# Submit job with default settings
sbatch scripts/download_documents.sbatch
```

#### Performance Features

The script will:
- Process all `exportWP*.xml` files in the specified XML directory
- Extract unique document URLs from `<LokURL>` elements
- Download PDF, DOCX, DOC, and HTML files in parallel
- Save files to the specified output directory (default: `data/documents/`)
- Generate detailed logs in `logs/download_documents.log`

### 3.3. Parse XML Metadata

The `scripts/parse_xml.py` script extracts comprehensive metadata from parliamentary XML exports.

#### Usage

```bash
# Parse all XML files and extract metadata
python scripts/parse_xml.py

# Custom input/output directories
python scripts/parse_xml.py --input data/xml --output data/metadata_complete
```

The script extracts:
- Document details (`DokArt`, `Titel`, `Datum`, `Nummer`)
- URLs for different file formats (PDF, DOCX, HTML)
- Speaker information (`Redner`)
- Procedural categories (`Desk`)
- Procedural text (`BText`)
- Complete Vorgang-Dokument hierarchy

### 3.4. Preprocess Metadata

The `scripts/preprocess_metadata.py` script handles duplicate references and creates ML-ready datasets.

#### Usage

```bash
# Process extracted metadata with default VTyp target
python scripts/preprocess_metadata.py

# Use different classification targets
python scripts/preprocess_metadata.py --target-field primary_dokart          # (imbalanced)
python scripts/preprocess_metadata.py --target-field derived_subject_area    # Subject-based classes
```

## 5. Project Roadmap

### Phase 1: Document Processing 🔄 **IN PROGRESS**
- [ ] **Text Extraction Pipeline**: PDF/DOCX/HTML text extraction
- [ ] **Text Preprocessing**: Clean and normalize German parliamentary text
- [ ] **Long Document Handling**: Chunking strategy for Llama 2 context limits
- [ ] **Train/Validation/Test Splits**: Stratified sampling by VTyp and Wahlperiode
- [ ] **Final Training Format**: Text-target pairs ready for model training

### Phase 2: Base Model Training 📋 **PLANNED**
- [ ] **Llama 2 7B + LoRA Setup**: Configure for VTyp classification
- [ ] **Distributed Training**: Utilize up to 8 H100 GPUs with SLURM
- [ ] **Model Training**: Fine-tune for German parliamentary document classification
- [ ] **Evaluation**: Accuracy, F1-score, confusion matrix analysis
- [ ] **Error Analysis**: Identify challenging document types and failure modes

### Phase 3: Multi-Metadata Prediction 📋 **PLANNED**
- [ ] **Multi-Task Architecture**: Predict VTyp + Titel + Desk + Redner simultaneously
- [ ] **Comparative Analysis**: Single-task vs. multi-task performance
- [ ] **Advanced Targets**: Subject area classification, document size prediction
- [ ] **Ensemble Methods**: Combine multiple models for improved accuracy

### Phase 4: Advanced Segmentation 📋 **FUTURE**
- [ ] **Document Segmentation**: Speaker-based and topic-based chunking
- [ ] **Hierarchical Prediction**: Segment-level → document-level aggregation
- [ ] **Attention Mechanisms**: Learn segment importance for document classification
- [ ] **Production Pipeline**: End-to-end metadata prediction system
