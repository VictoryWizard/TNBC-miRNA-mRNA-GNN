#!/usr/bin/env python3
"""Download clinical TNBC TCGA-BRCA expression from cBioPortal."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path('data')
PANEL_PATH = DATA_DIR / 'tcga_gene_panel.txt'
OUTPUT_PATH = DATA_DIR / 'tcga_brca_clinical_tnbc_mrna_expression.csv'
ENTREZ_CACHE_PATH = DATA_DIR / 'tcga_gene_entrez_map.json'
MANIFEST_PATH = DATA_DIR / 'tcga_clinical_tnbc_manifest.txt'

CBIO_BASE = 'https://www.cbioportal.org/api'
CLINICAL_STUDY = 'brca_tcga'
EXPRESSION_STUDY = 'brca_tcga_pan_can_atlas_2018'
MOLECULAR_PROFILE = f'{EXPRESSION_STUDY}_rna_seq_v2_mrna'

IHC_ATTRIBUTES = [
    'ER_STATUS_BY_IHC',
    'PR_STATUS_BY_IHC',
    'HER2_FISH_STATUS',
    'HER2_IHC_SCORE',
]


def _api_get(url: str, timeout: int = 60) -> object:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _api_post(url: str, body: object, timeout: int = 180) -> object:
    payload = json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def fetch_clinical_ihc_wide() -> pd.DataFrame:
    """Patient-level ER/PR/HER2 IHC and FISH from cBioPortal brca_tcga."""
    url = f'{CBIO_BASE}/studies/{CLINICAL_STUDY}/clinical-data?clinicalDataType=PATIENT'
    records = _api_get(url, timeout=120)
    frame = pd.DataFrame(records)
    subset = frame[frame['clinicalAttributeId'].isin(IHC_ATTRIBUTES)]
    wide = subset.pivot(index='patientId', columns='clinicalAttributeId', values='value')
    return wide


def is_receptor_negative(value: object) -> bool:
    if pd.isna(value):
        return False
    token = str(value).strip().lower()
    return token in {'negative', 'neg', 'not detected'}


def is_her2_negative(row: pd.Series) -> bool:
    if is_receptor_negative(row.get('HER2_FISH_STATUS')):
        return True
    score = str(row.get('HER2_IHC_SCORE', '')).strip()
    return score in {'0', '1+'}


def filter_clinical_tnbc_patients(ihc: pd.DataFrame) -> pd.Index:
    """ER-/PR-/HER2- by IHC, with HER2 IHC 0/1+ when FISH is missing."""
    mask = (
        ihc['ER_STATUS_BY_IHC'].map(is_receptor_negative)
        & ihc['PR_STATUS_BY_IHC'].map(is_receptor_negative)
        & ihc.apply(is_her2_negative, axis=1)
    )
    return ihc.index[mask]


def fetch_primary_sample_ids(tnbc_patients: pd.Index) -> list[str]:
    """Primary tumor samples in PanCan Atlas for clinical TNBC patients."""
    samples = _api_get(f'{CBIO_BASE}/studies/{EXPRESSION_STUDY}/samples', timeout=120)
    sample_df = pd.DataFrame(samples)
    primary = sample_df[
        sample_df['sampleType'].astype(str).str.contains('Primary', case=False, na=False)
    ]
    primary_tnbc = primary[primary['patientId'].isin(tnbc_patients)]
    # One primary sample per patient (prefer *-01).
    primary_tnbc = primary_tnbc.sort_values('sampleId')
    return primary_tnbc.drop_duplicates('patientId', keep='first')['sampleId'].tolist()


def load_gene_panel() -> list[str]:
    return [g.strip() for g in PANEL_PATH.read_text().splitlines() if g.strip()]


def load_entrez_map(symbols: list[str]) -> dict[str, int]:
    if ENTREZ_CACHE_PATH.exists():
        cached = json.loads(ENTREZ_CACHE_PATH.read_text())
        missing = [s for s in symbols if s not in cached]
    else:
        cached = {}
        missing = list(symbols)

    for symbol in missing:
        for attempt in range(3):
            try:
                gene = _api_get(f'{CBIO_BASE}/genes/{symbol}', timeout=30)
                cached[symbol] = int(gene['entrezGeneId'])
                break
            except urllib.error.HTTPError:
                break
            except urllib.error.URLError:
                time.sleep(2 * (attempt + 1))
        time.sleep(0.05)

    ENTREZ_CACHE_PATH.write_text(json.dumps(cached, indent=2, sort_keys=True))
    return {symbol: cached[symbol] for symbol in symbols if symbol in cached}


def fetch_expression_matrix(
    sample_ids: list[str], symbols: list[str], sample_chunk: int = 25, gene_chunk: int = 80
) -> pd.DataFrame:
    """RSEM expression (same profile as tcga_brca_basal_mrna_expression.csv)."""
    entrez_map = load_entrez_map(symbols)
    entrez_ids = [entrez_map[s] for s in symbols if s in entrez_map]
    id_to_symbol = {entrez_map[s]: s for s in symbols if s in entrez_map}

    records: list[dict] = []
    url = f'{CBIO_BASE}/molecular-profiles/{MOLECULAR_PROFILE}/molecular-data/fetch'
    for sample_start in range(0, len(sample_ids), sample_chunk):
        sample_batch = sample_ids[sample_start : sample_start + sample_chunk]
        for gene_start in range(0, len(entrez_ids), gene_chunk):
            gene_batch = entrez_ids[gene_start : gene_start + gene_chunk]
            body = {'sampleIds': sample_batch, 'entrezGeneIds': gene_batch}
            batch = _api_post(url, body, timeout=300)
            records.extend(batch)

    if not records:
        raise RuntimeError('No expression values returned from cBioPortal.')

    long = pd.DataFrame(records)
    matrix = long.pivot(index='sampleId', columns='entrezGeneId', values='value')
    matrix.columns = [id_to_symbol[int(col)] for col in matrix.columns]
    matrix = matrix.reindex(columns=[s for s in symbols if s in matrix.columns])
    matrix.index.name = 'sample_id'
    return matrix


def write_manifest(
    ihc: pd.DataFrame,
    tnbc_patients: pd.Index,
    sample_ids: list[str],
    expression: pd.DataFrame,
) -> None:
    lines = [
        'Clinical TNBC TCGA-BRCA download manifest',
        '=' * 72,
        f'IHC source study:        {CLINICAL_STUDY} (cBioPortal patient clinical)',
        f'Expression source study: {EXPRESSION_STUDY}',
        f'Molecular profile:       {MOLECULAR_PROFILE}',
        f'Filter:                  ER- and PR- by IHC; HER2- by FISH or IHC 0/1+',
        f'Clinical TNBC patients:  {len(tnbc_patients)}',
        f'Primary tumor samples:   {len(sample_ids)}',
        f'Genes in matrix:         {expression.shape[1]}',
        '',
        'Sample IDs:',
        *sample_ids,
        '',
    ]
    MANIFEST_PATH.write_text('\n'.join(lines))


def build_clinical_tnbc_matrix() -> pd.DataFrame:
    ihc = fetch_clinical_ihc_wide()
    tnbc_patients = filter_clinical_tnbc_patients(ihc)
    if len(tnbc_patients) == 0:
        raise RuntimeError('No clinical TNBC patients matched IHC filters.')

    sample_ids = fetch_primary_sample_ids(tnbc_patients)
    if len(sample_ids) == 0:
        raise RuntimeError('No primary tumor samples found for clinical TNBC patients.')

    symbols = load_gene_panel()
    expression = fetch_expression_matrix(sample_ids, symbols)
    expression = expression.sort_index()
    write_manifest(ihc, tnbc_patients, sample_ids, expression)
    return expression


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f'Missing gene panel: {PANEL_PATH}')

    print('Fetching ER/PR/HER2 IHC from cBioPortal (brca_tcga)...')
    matrix = build_clinical_tnbc_matrix()
    matrix.to_csv(OUTPUT_PATH)
    print(f'Saved {matrix.shape[0]} samples x {matrix.shape[1]} genes to {OUTPUT_PATH}')
    print(f'Manifest: {MANIFEST_PATH}')


if __name__ == '__main__':
    main()
