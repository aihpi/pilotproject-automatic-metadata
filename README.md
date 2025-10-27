# Automatic Metadata Extraction from Parliamentary Documents

## Table of Contents

1. [Requirements](#1-requirements)
   - 1.1. [Virtual Environment](#11-virtual-environment)
   - 1.2. [Working Environment](#12-working-environment)
2. [Scripts](#2-scripts)

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

### Document Download

The `scripts/download_documents.py` script extracts URLs from parliamentary XML exports and downloads all referenced PDF and DOCX documents.

#### Usage

```bash
# Basic usage - downloads all documents from XML files in data/xml/
python scripts/download_documents.py
```

The script will:
- Process all `exportWP*.xml` files in `data/xml/`
- Extract unique document URLs from `<LokURL>` elements
- Download PDF and DOCX files to `data/documents/`
- Skip files that already exist locally
- Provide progress tracking and error handling

#### Features

- **Multi-file processing**: Handles all electoral periods (WP1-WP8) automatically
- **Resume capability**: Skips already downloaded files
- **Error resilience**: Retries failed downloads and continues with the next file
- **Progress tracking**: Shows current progress and download status
- **Respectful downloading**: Includes delays to avoid overwhelming the server