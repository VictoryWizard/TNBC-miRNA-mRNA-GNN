#!/usr/bin/env python3
"""TCGA miRNA validation feasibility check (notebook 18).

Scans local data for TCGA miRNA expression files or miRNA-like columns in existing
TCGA tables. Does not download data or train models.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

warnings.filterwarnings('ignore')

TABLES_DIR = Path('results/tables')
SEARCH_DIRS = [
    Path('data'),
    Path('data/raw'),
    Path('data/processed'),
    Path('results/tables'),
]
TCGA_EXPRESSION_FILES = [
    Path('data/tcga_brca_basal_mrna_expression.csv'),
    Path('data/tcga_brca_clinical_tnbc_mrna_expression.csv'),
    Path('data/tcga_brca_normal_mrna_expression.csv'),
]

FILENAME_KEYWORDS = ('mirna', 'mirna', 'microrna', 'tcga', 'brca')
MIRNA_COLUMN_PATTERNS = (
    re.compile(r'^hsa-mir', re.I),
    re.compile(r'^hsa-let', re.I),
    re.compile(r'^mir-', re.I),
    re.compile(r'^let-', re.I),
)

TARGET_MIRNAS = [
    'hsa-miR-490-3p',
    'hsa-miR-1979',
    'hsa-let-7f',
    'hsa-let-7g',
    'hsa-miR-130b',
    'hsa-miR-425',
    'hsa-miR-421',
    'hsa-miR-106b',
    'hsa-miR-200a',
    'hsa-miR-660',
    'hsa-miR-10b',
    'hsa-miR-21',
    'hsa-miR-26b',
]

TEXT_EXTENSIONS = {'.csv', '.tsv', '.txt', '.json', '.gmt'}
SKIP_EXTENSIONS = {'.gz', '.zip', '.pkl', '.pt', '.pth', '.graphml', '.png', '.pdf'}


def normalize_mirna_name(name: str) -> str:
    """Normalize miRNA identifiers for matching."""
    cleaned = str(name).strip()
    cleaned = re.sub(r'\s*\(\+\+\+.*$', '', cleaned)
    cleaned = re.sub(r'\s*\(.*\)$', '', cleaned)
    return cleaned.strip()


def normalize_mirna_key(name: str) -> str:
    return normalize_mirna_name(name).lower()


def filename_matches_keywords(path: Path) -> bool:
    text = path.as_posix().lower()
    return any(keyword in text for keyword in FILENAME_KEYWORDS)


def is_tcga_path(path: Path) -> bool:
    return 'tcga' in path.as_posix().lower()


def is_mirna_like_column(column: str) -> bool:
    col = str(column).strip()
    if not col or col.lower() in {'sample_id', 'sample', 'patient_id', 'barcode'}:
        return False
    return any(pattern.search(col) for pattern in MIRNA_COLUMN_PATTERNS)


def count_rows(path: Path) -> int:
    try:
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return 0


def read_columns(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == '.tsv':
            return pd.read_csv(path, sep='\t', nrows=0).columns.tolist()
        if suffix in {'.csv', '.txt'}:
            return pd.read_csv(path, nrows=0).columns.tolist()
    except Exception:
        pass

    try:
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            header = handle.readline().strip()
        if not header:
            return []
        delimiter = '\t' if suffix == '.tsv' or '\t' in header else ','
        return [col.strip() for col in header.split(delimiter)]
    except OSError:
        return []


def discover_candidate_files() -> List[Path]:
    seen: Set[Path] = set()
    files: List[Path] = []

    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if not path.is_file():
                continue
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS and not filename_matches_keywords(path):
                continue
            if not filename_matches_keywords(path):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)

    for path in TCGA_EXPRESSION_FILES:
        if path.exists() and path.resolve() not in seen:
            seen.add(path.resolve())
            files.append(path)

    return sorted(files, key=lambda p: p.as_posix().lower())


def inspect_file(path: Path) -> Dict:
    columns = read_columns(path) if path.suffix.lower() in {'.csv', '.tsv', '.txt'} else []
    mirna_like = [col for col in columns if is_mirna_like_column(col)]
    notes: List[str] = []

    if path.suffix.lower() not in {'.csv', '.tsv', '.txt'}:
        notes.append('Non-tabular file; column inspection skipped')
    elif not columns:
        notes.append('Could not read column names')
    elif mirna_like:
        notes.append(f'Found {len(mirna_like)} miRNA-like column(s)')
    elif is_tcga_path(path) and 'mrna' in path.name.lower():
        notes.append('TCGA mRNA expression table; columns appear to be gene symbols')
    elif 'mirna' in path.name.lower() and path.suffix.lower() == '.csv':
        notes.append('Filename suggests miRNA content; inspect column semantics manually')

    return {
        'file_path': path.as_posix(),
        'n_rows': count_rows(path) if path.suffix.lower() in {'.csv', '.tsv', '.txt'} else '',
        'n_columns': len(columns) if columns else '',
        'appears_to_contain_mirna': bool(mirna_like),
        'example_mirna_like_columns': '; '.join(mirna_like[:10]),
        'notes': '; '.join(notes) if notes else '',
        '_columns': columns,
        '_mirna_like': mirna_like,
    }


def build_column_index(
    inventory_rows: List[Dict],
    tcga_only: bool,
) -> Dict[str, List[Tuple[str, str]]]:
    index: Dict[str, List[Tuple[str, str]]] = {}
    for row in inventory_rows:
        path = Path(row['file_path'])
        if tcga_only and not is_tcga_path(path):
            continue
        for col in row.get('_mirna_like', []):
            key = normalize_mirna_key(col)
            index.setdefault(key, []).append((row['file_path'], col))
    return index


def match_target_mirna(
    target: str,
    column_index: Dict[str, List[Tuple[str, str]]],
) -> Tuple[bool, str, str, str]:
    target_key = normalize_mirna_key(target)
    exact = column_index.get(target_key, [])
    if exact:
        file_path, column = exact[0]
        return True, file_path, column, 'Exact normalized column match'

    prefix_matches: List[Tuple[str, str]] = []
    for key, entries in column_index.items():
        if key.startswith(target_key) or target_key.startswith(key):
            prefix_matches.extend(entries)

    if prefix_matches:
        file_path, column = prefix_matches[0]
        return True, file_path, column, 'Prefix/normalized partial column match'

    return False, '', '', 'Not found in inspected TCGA miRNA-like columns'


def tcga_mirna_available(inventory_df: pd.DataFrame) -> bool:
    tcga_rows = inventory_df[inventory_df['file_path'].str.contains('tcga', case=False, na=False)]
    dedicated = tcga_rows['file_path'].str.contains('mirna|microrna', case=False, na=False)
    expression_like = tcga_rows['appears_to_contain_mirna']
    return bool(dedicated.any() or expression_like.any())


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 72)
    print('TCGA miRNA VALIDATION FEASIBILITY CHECK')
    print('=' * 72)

    candidate_files = discover_candidate_files()
    print(f'Candidate files scanned: {len(candidate_files)}', flush=True)

    inventory_rows = [inspect_file(path) for path in candidate_files]
    inventory_df = pd.DataFrame([
        {k: v for k, v in row.items() if not k.startswith('_')}
        for row in inventory_rows
    ])

    tcga_column_index = build_column_index(inventory_rows, tcga_only=True)
    feasibility_rows = []
    for mirna in TARGET_MIRNAS:
        present, matching_file, matching_column, notes = match_target_mirna(
            mirna, tcga_column_index
        )
        feasibility_rows.append({
            'miRNA': mirna,
            'present_in_any_tcga_file': present,
            'matching_file': matching_file,
            'matching_column': matching_column,
            'notes': notes,
        })

    feasibility_df = pd.DataFrame(feasibility_rows)
    inventory_path = TABLES_DIR / 'tcga_mirna_file_inventory.csv'
    feasibility_path = TABLES_DIR / 'tcga_mirna_feasibility.csv'
    inventory_df.to_csv(inventory_path, index=False)
    feasibility_df.to_csv(feasibility_path, index=False)

    found = feasibility_df.loc[feasibility_df['present_in_any_tcga_file'], 'miRNA'].tolist()
    missing = feasibility_df.loc[~feasibility_df['present_in_any_tcga_file'], 'miRNA'].tolist()
    available = tcga_mirna_available(inventory_df)

    if available:
        feasibility_text = (
            'Partial or full TCGA miRNA validation may be possible using local TCGA miRNA-like files.'
        )
    else:
        feasibility_text = (
            'Full TCGA miRNA validation is not feasible without downloading new TCGA miRNA '
            'expression data; existing TCGA files in this repo are mRNA/gene-level only.'
        )

    print('\n1. TCGA miRNA expression available locally:', 'yes' if available else 'no')
    print(f'2. Target miRNAs found in TCGA files: {len(found)} / {len(TARGET_MIRNAS)}')
    print(f'3. Found: {", ".join(found) if found else "none"}')
    print(f'4. Missing: {", ".join(missing) if missing else "none"}')
    print(f'5. Feasibility: {feasibility_text}')
    print('\nSaved files:')
    print(f'  {feasibility_path}')
    print(f'  {inventory_path}')


if __name__ == '__main__':
    main()
