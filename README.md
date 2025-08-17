# Bootcamp Repository
## Folder Structure
- **homework/** → All homework contributions will be submitted here.
- **project/** → All project contributions will be submitted here.
- **class_materials/** → Local storage for class materials. Never pushed to
GitHub.

## Homework Folder Rules
- Each homework will be in its own subfolder (`homework0`, `homework1`, etc.)
- Include all required files for grading.
## Project Folder Rules
- Keep project files organized and clearly named.

## Data Storage

This notebook demonstrates a reproducible data storage workflow using CSV and Parquet formats, managed via environment variables.

### Folder Structure

- `data/raw/` — Raw CSV files
- `data/processed/` — Optimized Parquet files

### Formats Used

- **CSV**: Human-readable and easy to share
- **Parquet**: Efficient for analytics, preserves data types

### Environment Variables

Paths are loaded from `.env`:

