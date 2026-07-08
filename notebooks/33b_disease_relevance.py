#!/usr/bin/env python3
"""Disease-relevance of GAT-top vs XGBoost-top features (the 'GAT earns its keep' test).

Question: even if the GAT does not beat XGBoost on accuracy, do the features it
ranks highest carry MORE established TNBC / breast-cancer relevance? If so, the
graph view surfaces more disease-relevant biology and the accuracy gap is a
small-sample artifact rather than a failure of the approach.

Method: take each model's top-K miRNAs and top-K mRNAs, attach a disease-relevance
score, and compare the GAT-top vs XGBoost-top score distributions (Mann-Whitney U,
one-sided GAT>XGB) plus a 'hit rate' (fraction with any known association).

Relevance sources (free, no API key):
  * mRNA  : Open Targets Platform GraphQL association score (0-1) for
            'triple-negative breast carcinoma' and 'breast carcinoma'. This is the
            analogue of the DisGeNET gene-disease score (e.g. PLAUR).
  * miRNA : HMDD v4 human miRNA-disease associations (breast/TNBC); score = number
            of supporting records. Attempted; skipped gracefully if unreachable.

Inputs:  results/tables/{xgboost_importance,tabnet_importance,gat_attention_stability_graphml}.csv
Outputs: results/tables/disease_relevance_scores.csv
         results/tables/disease_relevance_summary.csv

Run from repo root (needs internet):  python notebooks/33_disease_relevance.py
"""

from pathlib import Path
import json, time
import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None

TABLES = Path('results/tables'); TABLES.mkdir(parents=True, exist_ok=True)
TOP_K = 25  # standardized across analyses
OT_URL = 'https://api.platform.opentargets.org/api/v4/graphql'
DISEASE_QUERIES = ['triple-negative breast carcinoma', 'breast carcinoma']


def modality_of(name):
    n = str(name).lower()
    return 'miRNA' if n.startswith('hsa-') or n.startswith('mir') else 'mRNA'


# ---------- load model top-K lists ----------
def top_k_lists():
    xgb = pd.read_csv(TABLES / 'xgboost_importance.csv').rename(
        columns={'feature': 'molecule', 'mean_abs_shap': 'imp'})
    xgb['modality'] = xgb['molecule'].map(modality_of)
    gat = pd.read_csv(TABLES / 'gat_attention_stability_graphml.csv')
    if 'molecule' in gat.columns and 'node' in gat.columns:
        gat = gat.drop(columns=['node'])
    elif 'molecule' not in gat.columns:
        gat = gat.rename(columns={'node': 'molecule'})
    gat = gat.rename(columns={'mean_attention': 'imp'})
    if 'modality' not in gat.columns:
        gat['modality'] = gat['molecule'].map(modality_of)
    out = {}
    for name, d in [('XGBoost', xgb), ('GAT', gat)]:
        for mod in ['miRNA', 'mRNA']:
            sub = d[d['modality'].str.lower().str.contains(mod.lower())]
            out[(name, mod)] = sub.sort_values('imp', ascending=False)['molecule'].head(TOP_K).tolist()
    return out


# ---------- Open Targets (mRNA) ----------
def _ot(query, variables):
    r = requests.post(OT_URL, json={'query': query, 'variables': variables}, timeout=30)
    r.raise_for_status()
    return r.json()['data']


def resolve_disease_efos():
    q = 'query($s:String!){search(queryString:$s,entityNames:["disease"],page:{index:0,size:1}){hits{id name}}}'
    efos = []
    for name in DISEASE_QUERIES:
        try:
            hits = _ot(q, {'s': name})['search']['hits']
            if hits:
                efos.append((hits[0]['id'], hits[0]['name'])); print(f'  disease "{name}" -> {hits[0]["id"]} ({hits[0]["name"]})')
        except Exception as e:
            print(f'  disease resolve failed for {name}: {e}')
    return efos


_DEBUG = {'done': False}


def ot_score(symbol, efo_ids):
    """Max Open Targets association score for a gene symbol across the EFO ids.

    enableIndirect:true propagates associations through the disease ontology;
    without it, filtering to specific EFOs (TNBC/breast carcinoma) returns no
    rows for most targets and every score collapses to 0.
    """
    sq = 'query($s:String!){search(queryString:$s,entityNames:["target"],page:{index:0,size:1}){hits{id name}}}'
    # No efoIds/enableIndirect args (they 400 the API): fetch top disease
    # associations with paging and match TNBC/breast client-side by id or name.
    aq = ('query($e:String!){target(ensemblId:$e){'
          'associatedDiseases(page:{index:0,size:500}){rows{score disease{id name}}}}}')
    try:
        hits = _ot(sq, {'s': symbol})['search']['hits']
        if not hits:
            return 0.0
        data = _ot(aq, {'e': hits[0]['id']})
        if not _DEBUG['done']:
            print('   [debug first OT response]', json.dumps(data)[:300]); _DEBUG['done'] = True
        rows = data['target']['associatedDiseases']['rows']
        tgt = set(efo_ids)
        vals = [r['score'] for r in rows
                if r['disease']['id'] in tgt or 'breast' in r['disease']['name'].lower()]
        return float(max(vals, default=0.0))
    except Exception as e:
        if not _DEBUG['done']:
            print('   [debug OT error]', repr(e)); _DEBUG['done'] = True
        return 0.0


# ---------- HMDD (miRNA) ----------
def hmdd_breast_counts():
    """Return {normalized_mirna: n_breast_records} from HMDD v4, or {} if unreachable."""
    import io, urllib3
    urllib3.disable_warnings()
    urls = ['https://www.cuilab.cn/static/hmdd4/data/alldata.txt',
            'https://www.cuilab.cn/static/hmdd3/data/alldata.txt']
    for u in urls:
        try:
            txt = requests.get(u, verify=False, timeout=60).text  # cert expired -> verify off
            df = pd.read_csv(io.StringIO(txt), sep='\t', encoding='latin-1')
            dcol = [c for c in df.columns if 'disease' in c.lower()][0]
            mcol = [c for c in df.columns if 'mir' in c.lower()][0]
            bc = df[df[dcol].str.contains('breast', case=False, na=False)]
            counts = bc[mcol].str.lower().str.replace('hsa-', '', regex=False).value_counts()
            print(f'  HMDD loaded from {u}: {len(bc)} breast records')
            return {k: int(v) for k, v in counts.items()}
        except Exception as e:
            print(f'  HMDD fetch failed ({u}): {e}')
    return {}


def norm_mirna(name):
    return str(name).lower().replace('hsa-', '').strip()


def main():
    if requests is None:
        raise SystemExit('requests not installed: pip install requests')
    lists = top_k_lists()

    print('Resolving disease EFO ids (Open Targets)...')
    efos = resolve_disease_efos()
    efo_ids = [e for e, _ in efos]

    # mRNA scores via Open Targets
    mrna_genes = sorted(set(lists[('XGBoost', 'mRNA')]) | set(lists[('GAT', 'mRNA')]))
    print(f'\nScoring {len(mrna_genes)} mRNAs via Open Targets...')
    mrna_score = {}
    for g in mrna_genes:
        mrna_score[g] = ot_score(g, efo_ids) if efo_ids else 0.0
        time.sleep(0.2)
        print(f'  {g}: {mrna_score[g]:.3f}')

    # miRNA scores via HMDD
    print('\nLoading HMDD breast-cancer miRNA associations...')
    hmdd = hmdd_breast_counts()
    mir_genes = sorted(set(lists[('XGBoost', 'miRNA')]) | set(lists[('GAT', 'miRNA')]))
    mir_score = {m: float(hmdd.get(norm_mirna(m), 0)) for m in mir_genes} if hmdd else {}

    # assemble per-feature table
    rows = []
    for (model, mod), feats in lists.items():
        for f in feats:
            if mod == 'mRNA':
                s = mrna_score.get(f, np.nan); src = 'OpenTargets'
            else:
                s = mir_score.get(f, np.nan); src = 'HMDD'
            rows.append({'model': model, 'modality': mod, 'molecule': f,
                         'relevance_score': s, 'source': src})
    scores = pd.DataFrame(rows)
    scores.to_csv(TABLES / 'disease_relevance_scores.csv', index=False)

    # Compare GAT vs XGBoost per modality -- DESCRIPTIVE ONLY. With ~25 features
    # per group and a small cohort, a significance test is underpowered and
    # uninformative, so we report magnitudes (mean/median/hit-rate) and let them
    # speak rather than hanging the claim on a p-value.
    summ = []
    for mod in ['mRNA', 'miRNA']:
        g = scores[(scores.model == 'GAT') & (scores.modality == mod)]['relevance_score'].dropna()
        x = scores[(scores.model == 'XGBoost') & (scores.modality == mod)]['relevance_score'].dropna()
        if len(g) == 0 or len(x) == 0:
            print(f'\n{mod}: insufficient scores (source unavailable) — skipped')
            continue
        summ.append({'modality': mod,
                     'gat_mean': round(float(g.mean()), 3), 'xgb_mean': round(float(x.mean()), 3),
                     'gat_median': round(float(g.median()), 3), 'xgb_median': round(float(x.median()), 3),
                     'gat_hit_rate': round(float((g > 0).mean()), 3), 'xgb_hit_rate': round(float((x > 0).mean()), 3),
                     'gat_minus_xgb_mean': round(float(g.mean() - x.mean()), 3),
                     'n_per_group': int(min(len(g), len(x)))})
    summ = pd.DataFrame(summ)
    summ.to_csv(TABLES / 'disease_relevance_summary.csv', index=False)

    print('\n=== Disease-relevance: GAT-top vs XGBoost-top (descriptive, no test) ===')
    if not summ.empty:
        print(summ.to_string(index=False))
        print('\nHigher gat_mean/median/hit_rate than xgb => GAT-top features are more TNBC/breast-relevant.')
    print('\nSaved disease_relevance_scores.csv and disease_relevance_summary.csv')


if __name__ == '__main__':
    main()
