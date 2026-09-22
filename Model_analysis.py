

# ------------------------------------------------------------------------------------------------ #

### Title:  ## Models pipeline ##
### Author: Patron-Rivero, Carlos ####
### Project: "Morphological change occurred by phylogenetic context, not by ecological divergence in Neotropical hognose pitvipers (Viperidae: Crotalinae)" ###

# ------------------------------------------------------------------------------------------------ #

## Setup ##

# ------------------------------------------------------------------------------------------------ #

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.figure
from scipy import optimize

SEED = 42
rng = np.random.default_rng(SEED)

mpl.rcParams['font.family'] = ['Liberation Sans', 'Arimo', 'DejaVu Sans']
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

import os
os.makedirs('/mnt/results/figures', exist_ok=True)
os.makedirs('/mnt/results/data', exist_ok=True)

# S3-backed mounts can refuse to overwrite existing files after a machine
# hibernation cycle; remove-then-write is always safe. Idempotent patch.
if not getattr(pd.DataFrame.to_csv, '_safe_patched', False):
    _orig_to_csv = pd.DataFrame.to_csv
    def _safe_to_csv(self, path, *a, **k):
        if os.path.exists(path):
            os.remove(path)
        return _orig_to_csv(self, path, *a, **k)
    _safe_to_csv._safe_patched = True
    pd.DataFrame.to_csv = _safe_to_csv

if not getattr(matplotlib.figure.Figure.savefig, '_safe_patched', False):
    _orig_savefig = matplotlib.figure.Figure.savefig
    def _safe_savefig(self, path, *a, **k):
        if os.path.exists(path):
            os.remove(path)
        return _orig_savefig(self, path, *a, **k)
    _safe_savefig._safe_patched = True
    matplotlib.figure.Figure.savefig = _safe_savefig

print('numpy', np.__version__, '| pandas', pd.__version__)

# ------------------------------------------------------------------------------------------------ #

## TPS ##

# ------------------------------------------------------------------------------------------------ #

TPS_PATH = '/mnt/user-uploads/P_lm.TPS'

def parse_tps(path):
    specimens, cur = [], None
    with open(path) as f:
        lines = [l.strip() for l in f]
    for line in lines:
        if line.startswith('LM='):
            if cur is not None and len(cur['coords']) == cur['lm']:
                specimens.append(cur)
            cur = {'lm': int(line.split('=')[1]), 'coords': [],
                   'id': None, 'scale': None, 'image': None}
        elif cur is None:
            continue
        elif line.startswith('ID='):
            cur['id'] = line[3:].strip()
        elif line.startswith('SCALE='):
            cur['scale'] = float(line.split('=')[1])
        elif line.startswith('IMAGE='):
            cur['image'] = line[6:].strip()
        elif line and len(cur['coords']) < cur['lm']:
            parts = line.split()
            if len(parts) == 2:
                try:
                    cur['coords'].append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
    if cur is not None and len(cur['coords']) == cur['lm']:
        specimens.append(cur)
    return specimens

specs = parse_tps(TPS_PATH)
print('Parsed specimens:', len(specs))
assert all(len(s['coords']) == 55 for s in specs)
assert all(s['id'] for s in specs)

# species mapping: TPS prefix -> tree/eco labels
PREFIX_MAP = {'P_arc': 'arcosae', 'P_dun': 'dunni', 'P_hes': 'hespere',
              'P_lan': 'lansbergii', 'P_nas': 'nasutum', 'P_oph': 'ophryomegas',
              'P_por': 'porrasi', 'P_yuc': 'yucatanicum', 'P_vol': 'volcanicum'}

import re as re2
def sp_of(sid):
    m = re2.match(r'(P_[a-z]{3})', sid)
    return PREFIX_MAP.get(m.group(1), 'UNKNOWN')

for s in specs:
    s['species'] = sp_of(s['id'])

from collections import Counter
counts = Counter(s['species'] for s in specs)
print('Per-species specimen counts:', dict(counts))

# --- exclusions ------------------------------------------------------------------
no_scale = [s for s in specs if s['scale'] is None]
print('\nExcluded: %d specimen(s) without a SCALE line (cannot be placed on the absolute size scale): %s'
      % (len(no_scale), ', '.join(s['id'] for s in no_scale)))
specs = [s for s in specs if s['scale'] is not None]

print('Excluded: %d specimen(s) of P. volcanicum (absent from phylogeny and ecology table)' % counts.get('volcanicum', 0))
specs = [s for s in specs if s['species'] != 'volcanicum']

SP8 = sorted({s['species'] for s in specs})
print('\nSpecies in analysis (n=%d):' % len(SP8), SP8)
print('Total specimens retained:', len(specs))
final_counts = Counter(s['species'] for s in specs)
print('Final per-species counts:', dict(final_counts))
print('\nSMALL-N SPECIES (flagged for limitations): ' +
      ', '.join(f'{sp} (n={final_counts[sp]})' for sp in ['arcosae', 'porrasi', 'hespere']))
	  
# ------------------------------------------------------------------------------------------------ #

### GPA, size and shape

# ------------------------------------------------------------------------------------------------ #

X_raw = np.array([np.array(s['coords']) * s['scale'] for s in specs])   # (441, 55, 2) in mm

def centroid_size(X):
    c = X.mean(axis=0)
    return np.sqrt(((X - c) ** 2).sum())

cent_sizes = np.array([centroid_size(X) for X in X_raw])
logCS = np.log(cent_sizes)

# --- Full GPA ---
def procrustes_rotate(X, ref):
    """Rotate shape X onto reference (both centered, unit size); no reflection."""
    M = X.T @ ref
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:                      # disallow reflection
        Vt[-1] *= -1
        R = U @ Vt
    return X @ R

def gpa(shapes, tol=1e-10, max_iter=200):
    shapes = [X - X.mean(axis=0) for X in shapes]
    shapes = [X / centroid_size(X) for X in shapes]
    ref = np.mean(shapes, axis=0)
    ref /= centroid_size(ref)
    for it in range(max_iter):
        aligned = np.array([procrustes_rotate(X, ref) for X in shapes])
        new_ref = aligned.mean(axis=0)
        new_ref /= centroid_size(new_ref)
        shift = np.sqrt(((new_ref - ref) ** 2).sum())
        ref = new_ref
        if shift < tol:
            break
    return aligned, it + 1, shift

aligned, n_iter, final_shift = gpa(list(X_raw))
print(f'GPA converged in {n_iter} iterations (final reference shift {final_shift:.2e})')

# --- PCA of aligned coordinates ---
X_vec = aligned.reshape(len(aligned), -1)
X_c = X_vec - X_vec.mean(axis=0)
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
var_expl = S ** 2 / (S ** 2).sum()
pc_scores = U * S                    # scores on principal axes

n95 = int(np.searchsorted(np.cumsum(var_expl), 0.95) + 1)
print(f'{len(var_expl)} non-zero PCs; first {n95} explain 95% of shape variance')
print('Variance explained by PC1-PC5:', np.round(var_expl[:5] * 100, 2))

# --- Specimen table ---
datR = pd.DataFrame({
    'specimen_id': [s['id'] for s in specs],
    'species': [s['species'] for s in specs],
    'CS_mm': cent_sizes, 'logCS': logCS})
for k in range(n95):
    datR[f'PC{k+1}'] = pc_scores[:, k]

print('\nCS (mm) summary by species:')
print(datR.groupby('species')['CS_mm'].agg(['count', 'min', 'mean', 'max']).round(2))

import os, sys
print('os module:', os.__file__ if hasattr(os, '__file__') else 'builtin')
print('os.stat:', os.stat)
print('type:', type(os.stat))
print('is builtin:', 'built-in' in repr(os.stat))
print('exists:', os.path.exists)
print('genericpath:', __import__('genericpath').__file__)
import genericpath
import inspect
try:
    print(inspect.getsource(genericpath.exists))
except Exception as e:
    print('no source:', e)
# check for shadowing names in builtins
candidates = [n for n in dir(os) if 'stat' in n.lower()]
print('os stat-related attrs:', candidates)
print('os.stat is posix.stat:', os.stat is __import__('posix').stat)


import sys, traceback
sys.setrecursionlimit(60)
try:
    pd.DataFrame({'a': [1]}).to_csv('/mnt/results/data/_diag.csv')
    print('no recursion at limit 60')
except RecursionError:
    tb = traceback.format_exc().splitlines()
    print('TOTAL TRACEBACK LINES:', len(tb))
    print('\n'.join(tb[:14]))
    print('   ...')
    print('\n'.join(tb[-14:]))
finally:
    sys.setrecursionlimit(1000)

scales = np.array([s['scale'] for s in specs])
print('SCALE value counts:', dict(Counter(scales)))

raw_cs = np.array([centroid_size(np.array(s['coords'])) for s in specs])
print('\nRaw CS (px) summary: min %.0f, median %.0f, max %.0f' % (raw_cs.min(), np.median(raw_cs), raw_cs.max()))

scaled_cs = raw_cs * scales
print('Scaled CS summary: min %.1f, median %.1f, max %.1f' % (scaled_cs.min(), np.median(scaled_cs), scaled_cs.max()))

# which specimens are extreme?
order = np.argsort(scaled_cs)
print('\n5 smallest:')
for i in order[:5]:
    print('  %s (%s) scale=%s rawCS=%.0f scaledCS=%.1f' % (specs[i]['id'], specs[i]['species'], specs[i]['scale'], raw_cs[i], scaled_cs[i]))
print('5 largest:')
for i in order[-5:]:
    print('  %s (%s) scale=%s rawCS=%.0f scaledCS=%.1f' % (specs[i]['id'], specs[i]['species'], specs[i]['scale'], raw_cs[i], scaled_cs[i]))

# relationship between raw CS and scale: do different scales correspond to different image sizes?
import collections
sc_by_sp = collections.defaultdict(set)
for s in specs:
    sc_by_sp[s['species']].add(s['scale'])
print('\nScales per species:', {k: sorted(v) for k, v in sc_by_sp.items()})

print('Raw CS (px): min %.0f | median %.0f | max %.0f' % (raw_cs.min(), np.median(raw_cs), raw_cs.max()))
print('Scaled CS (mm): min %.1f | median %.1f | max %.1f' % (scaled_cs.min(), np.median(scaled_cs), scaled_cs.max()))
print('Scale range: %.5f - %.5f' % (scales.min(), scales.max()))
order = np.argsort(scaled_cs)
print('\n3 smallest scaled CS:')
for i in order[:3]:
    print('  %-22s %-12s scale=%.5f rawCS=%6.0f -> %7.1f' % (specs[i]['id'], specs[i]['species'], specs[i]['scale'], raw_cs[i], scaled_cs[i]))
print('3 largest scaled CS:')
for i in order[-3:]:
    print('  %-22s %-12s scale=%.5f rawCS=%6.0f -> %7.1f' % (specs[i]['id'], specs[i]['species'], specs[i]['scale'], raw_cs[i], scaled_cs[i]))
# per-species median scaled CS
import pandas as pd2
cs_df = pd.DataFrame({'sp': [s['species'] for s in specs], 'cs': scaled_cs})
print('\nMedian scaled CS by species:')
print(cs_df.groupby('sp')['cs'].agg(['count', 'median']).round(1))

# logCS distribution check (scale-error sanity)
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True)
for ax, sp in zip(axes.ravel(), ['lansbergii', 'nasutum', 'ophryomegas', 'yucatanicum', 'dunni', 'hespere']):
    sub = datR.loc[datR['species'] == sp, 'logCS']
    ax.hist(sub, bins=20, color='#0279EE', edgecolor='white')
    ax.set_title(f'{sp} (n={len(sub)})', fontsize=10)
    ax.set_xlabel('logCS')
fig.tight_layout()
plt.show()

print(datR.groupby('species')['logCS'].describe()[['count', 'min', '25%', '50%', '75%', 'max']].round(2))

# ------------------------------------------------------------------------------------------------ #

## Species-level distance matrices ##

# ------------------------------------------------------------------------------------------------ #

# species means
sp_means_R = datR.groupby('species').mean(numeric_only=True)
SP8 = sorted(sp_means_R.index)

# phylogeny: parse real tree
REAL_NEWICK = open('/mnt/user-uploads/P_tree.newick').read().strip()
rt = parse_newick(REAL_NEWICK)
rt_height = assign_heights(rt)
add_parents(rt)
r_leaves = get_leaves(rt)

# map tree tip names (Porthidium_x) -> short labels
TIP_MAP = {'Porthidium_arcosae': 'arcosae', 'Porthidium_dunni': 'dunni',
           'Porthidium_hespere': 'hespere', 'Porthidium_lansbergii': 'lansbergii',
           'Porthidium_nasutum': 'nasutum', 'Porthidium_ophryomegas': 'ophryomegas',
           'Porthidium_porrasi': 'porrasi', 'Porthidium_yucatanicum': 'yucatanicum'}
tree_species = [TIP_MAP[l['name']] for l in r_leaves]
assert set(tree_species) == set(SP8), (tree_species, SP8)

r_paths = {}
for l in r_leaves:
    anc, cur = [], l
    while cur is not None:
        anc.append((id(cur), cur['height']))
        cur = cur['parent']
    r_paths[TIP_MAP[l['name']]] = anc

# ultrametricity check
tip_heights = [h for l in r_leaves for k, h in r_paths[TIP_MAP[l['name']]] if k == id(l)]
print('Tip heights (should all be 0):', set(tip_heights))
print('Root height: %.2f Ma' % rt_height)

n8 = len(SP8)
D_phy_R = np.zeros((n8, n8))
for i in range(n8):
    for j in range(n8):
        si = {k for k, _ in r_paths[SP8[i]]}
        mrca_h = min(h for k, h in r_paths[SP8[j]] if k in si)   # MRCA height above present
        D_phy_R[i, j] = 2 * mrca_h   # patristic = up to MRCA and back down

# ecological divergence matrices from long table
eco = pd.read_csv('/mnt/user-uploads/P_eco.csv')
assert len(eco) == 56 and set(eco['Spp1']) == set(eco['Spp2'])
ECO_MAP = {'P_arc': 'arcosae', 'P_dun': 'dunni', 'P_hes': 'hespere', 'P_lan': 'lansbergii',
           'P_nas': 'nasutum', 'P_oph': 'ophryomegas', 'P_por': 'porrasi', 'P_yuc': 'yucatanicum'}
D_ecoD = np.zeros((n8, n8)); D_ecoI = np.zeros((n8, n8))
for _, row in eco.iterrows():
    i, j = SP8.index(ECO_MAP[row['Spp1']]), SP8.index(ECO_MAP[row['Spp2']])
    D_ecoD[i, j] = D_ecoD[j, i] = row['Divergence_D']
    D_ecoI[i, j] = D_ecoI[j, i] = row['Divergence_I']
assert np.allclose(D_ecoD, D_ecoD.T) and np.allclose(D_ecoI, D_ecoI.T)

# morphological distances
size_mean = sp_means_R['logCS']
D_size = np.abs(size_mean.values[:, None] - size_mean.values[None, :])

pc_cols = [f'PC{k}' for k in range(1, n95 + 1)]
Zshape = sp_means_R[pc_cols] / datR[pc_cols].std()          # standardize by total SD
D_shape = np.sqrt(((Zshape.values[:, None, :] - Zshape.values[None, :, :]) ** 2).sum(-1))

Zall = (sp_means_R[['logCS'] + pc_cols] / datR[['logCS'] + pc_cols].std())
D_morph_R = np.sqrt(((Zall.values[:, None, :] - Zall.values[None, :, :]) ** 2).sum(-1))

# save
csvs = {'R_D_size': D_size, 'R_D_shape': D_shape, 'R_D_morph': D_morph_R,
        'R_D_ecoD': D_ecoD, 'R_D_ecoI': D_ecoI, 'R_D_phy': D_phy_R}
for name, M in csvs.items():
    pd.DataFrame(M, index=SP8, columns=SP8).to_csv(f'/mnt/results/data/{name}.csv')

print('\nMatrix ranges (off-diagonal):')
for name, M in csvs.items():
    off = M[~np.eye(n8, dtype=bool)]
    print('  %-10s %.3f - %.3f' % (name, off.min(), off.max()))
print('\nCorrelation between eco metrics (28 pairs): %.3f'
      % np.corrcoef(D_ecoD[np.triu_indices(n8, 1)], D_ecoI[np.triu_indices(n8, 1)])[0, 1])

# ------------------------------------------------------------------------------------------------ #

## Model comparison ##

# ------------------------------------------------------------------------------------------------ #
	  
N_PERM_R = 9999
iu8 = np.triu_indices(n8, 1)

RESPONSES_R = {'Size (|dlogCS|)': D_size,
               'Shape (PC space)': D_shape,
               'Combined morphology': D_morph_R}
ECOS_R = {"Schoener's D": D_ecoD, "Hellinger's I": D_ecoI}

real_tables = {}
real_mrm = {}
real_mantel = {}

for resp_name, Dresp in RESPONSES_R.items():
    yv = Dresp[iu8]
    for eco_name, Deco in ECOS_R.items():
        ev = Deco[iu8]; pv = D_phy_R[iu8]
        X0 = np.ones((len(yv), 1))
        X1 = np.column_stack([np.ones(len(yv)), ev])
        X2 = np.column_stack([np.ones(len(yv)), pv])
        X3 = np.column_stack([np.ones(len(yv)), ev, pv])
        fits = {'M0 null': ols_fit(X0, yv), 'M1 ecology': ols_fit(X1, yv),
                'M2 phylogeny': ols_fit(X2, yv), 'M3 ecology + phylogeny': ols_fit(X3, yv)}
        real_tables[(resp_name, eco_name)] = compare_models(fits, len(yv)).assign(
            response=resp_name, eco=eco_name)

        # MRM permutation tests for each model
        mrm_out = {}
        for mname, Xm in [('M1', X1), ('M2', X2), ('M3', X3)]:
            beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
            resid = yv - Xm @ beta
            r2 = 1 - (resid**2).sum() / ((yv - yv.mean())**2).sum()
            stats_ = np.zeros((N_PERM_R, len(beta) + 1))
            for k in range(N_PERM_R):
                perm = rng.permutation(n8)
                yp = Dresp[np.ix_(perm, perm)][iu8]
                bk, *_ = np.linalg.lstsq(Xm, yp, rcond=None)
                rk = yp - Xm @ bk
                r2k = 1 - (rk**2).sum() / ((yp - yp.mean())**2).sum()
                stats_[k] = np.append(bk, r2k)
            p_vals = (stats_ >= np.append(beta, r2)).mean(axis=0)
            mrm_out[mname] = {'beta': beta, 'R2': r2, 'p': p_vals}
        real_mrm[(resp_name, eco_name)] = mrm_out

    # Mantel tests (with D as the eco metric)
    r_me, p_me = mantel(Dresp, D_ecoD, n_perm=N_PERM_R)
    r_mp, p_mp = mantel(Dresp, D_phy_R, n_perm=N_PERM_R)
    real_mantel[resp_name] = {'morph~ecoD': (r_me, p_me), 'morph~phy': (r_mp, p_mp)}

r_ep_R, p_ep_R = mantel(D_ecoD, D_phy_R, n_perm=N_PERM_R)
real_mantel['ecoD~phy'] = {'': (r_ep_R, p_ep_R)}

tableR_all = pd.concat(real_tables.values())
tableR_all.to_csv('/mnt/results/data/R_model_table_8sp.csv', index=False)

pd.set_option('display.float_format', lambda v: f'{v:,.3f}')
for (resp_name, eco_name), t in real_tables.items():
    print('=' * 95)
    print(f'{resp_name} | ecology: {eco_name}')
    print(t[['model', 'logLik', 'K', 'AICc', 'dAICc', 'weight']].to_string(index=False))
    for mname, out in real_mrm[(resp_name, eco_name)].items():
        terms = ['int', 'eco', 'phy'] if mname == 'M3' else (['int', 'eco'] if mname == 'M1' else ['int', 'phy'])
        ps = ' '.join(f'{t_}={b:+.3f}(p={p:.4f})' for t_, b, p in zip(terms, out['beta'], out['p']))
        print(f'  MRM {mname}: R2={out["R2"]:.3f}  {ps}')
print('=' * 95)
print('Mantel one-sided tests (9,999 permutations):')
for k, d in real_mantel.items():
    for sub, (r, p) in d.items():
        label = f'{k} {sub}'.strip()
        print(f'  {label}: r = {r:.3f}, p = {p:.4f}')


for resp_name in ['Shape (PC space)', 'Combined morphology']:
    for eco_name in ["Schoener's D", "Hellinger's I"]:
        t = real_tables[(resp_name, eco_name)]
        print('=' * 95)
        print(f'{resp_name} | ecology: {eco_name}')
        print(t[['model', 'logLik', 'K', 'AICc', 'dAICc', 'weight']].to_string(index=False))
        for mname, out in real_mrm[(resp_name, eco_name)].items():
            terms = ['int', 'eco', 'phy'] if mname == 'M3' else (['int', 'eco'] if mname == 'M1' else ['int', 'phy'])
            ps = ' '.join(f'{t_}={b:+.3f}(p={p:.4f})' for t_, b, p in zip(terms, out['beta'], out['p']))
            print(f'  MRM {mname}: R2={out["R2"]:.3f}  {ps}')

SP5 = ['dunni', 'lansbergii', 'nasutum', 'ophryomegas', 'yucatanicum']
idx5 = [SP8.index(s) for s in SP5]
n5 = len(SP5)
iu5 = np.triu_indices(n5, 1)

mats5 = {'Size': D_size[np.ix_(idx5, idx5)], 'Shape': D_shape[np.ix_(idx5, idx5)],
         'Combined': D_morph_R[np.ix_(idx5, idx5)],
         'ecoD': D_ecoD[np.ix_(idx5, idx5)], 'ecoI': D_ecoI[np.ix_(idx5, idx5)],
         'phy': D_phy_R[np.ix_(idx5, idx5)]}

sup_tables = {}; sup_mrm = {}; sup_mantel = {}
for resp_name, Dresp in [('Size', mats5['Size']), ('Shape', mats5['Shape']), ('Combined', mats5['Combined'])]:
    yv = Dresp[iu5]
    for eco_name, Deco in [("Schoener's D", mats5['ecoD']), ("Hellinger's I", mats5['ecoI'])]:
        ev = Deco[iu5]; pv = mats5['phy'][iu5]
        X0 = np.ones((len(yv), 1))
        X1 = np.column_stack([np.ones(len(yv)), ev])
        X2 = np.column_stack([np.ones(len(yv)), pv])
        X3 = np.column_stack([np.ones(len(yv)), ev, pv])
        fits = {'M0 null': ols_fit(X0, yv), 'M1 ecology': ols_fit(X1, yv),
                'M2 phylogeny': ols_fit(X2, yv), 'M3 ecology + phylogeny': ols_fit(X3, yv)}
        sup_tables[(resp_name, eco_name)] = compare_models(fits, len(yv)).assign(
            response=resp_name, eco=eco_name)
        mrm_out = {}
        for mname, Xm in [('M1', X1), ('M2', X2), ('M3', X3)]:
            beta, *_ = np.linalg.lstsq(Xm, yv, rcond=None)
            resid = yv - Xm @ beta
            r2 = 1 - (resid**2).sum() / ((yv - yv.mean())**2).sum()
            stats_ = np.zeros((N_PERM_R, len(beta) + 1))
            for k in range(N_PERM_R):
                perm = rng.permutation(n5)
                yp = Dresp[np.ix_(perm, perm)][iu5]
                bk, *_ = np.linalg.lstsq(Xm, yp, rcond=None)
                rk = yp - Xm @ bk
                r2k = 1 - (rk**2).sum() / ((yp - yp.mean())**2).sum()
                stats_[k] = np.append(bk, r2k)
            p_vals = (stats_ >= np.append(beta, r2)).mean(axis=0)
            mrm_out[mname] = {'beta': beta, 'R2': r2, 'p': p_vals}
        sup_mrm[(resp_name, eco_name)] = mrm_out
    r_me, p_me = mantel(Dresp, mats5['ecoD'], n_perm=N_PERM_R)
    r_mp, p_mp = mantel(Dresp, mats5['phy'], n_perm=N_PERM_R)
    sup_mantel[resp_name] = {'morph~ecoD': (r_me, p_me), 'morph~phy': (r_mp, p_mp)}

sup_mantel['ecoD~phy'] = {'': mantel(mats5['ecoD'], mats5['phy'], n_perm=N_PERM_R)}

tableS_all = pd.concat(sup_tables.values())
tableS_all.to_csv('/mnt/results/data/S_model_table_5sp.csv', index=False)

for (resp_name, eco_name), t in sup_tables.items():
    print('=' * 95)
    print(f'{resp_name} | ecology: {eco_name}')
    print(t[['model', 'logLik', 'K', 'AICc', 'dAICc', 'weight']].to_string(index=False))
    for mname, out in sup_mrm[(resp_name, eco_name)].items():
        terms = ['int', 'eco', 'phy'] if mname == 'M3' else (['int', 'eco'] if mname == 'M1' else ['int', 'phy'])
        ps = ' '.join(f'{t_}={b:+.3f}(p={p:.4f})' for t_, b, p in zip(terms, out['beta'], out['p']))
        print(f'  MRM {mname}: R2={out["R2"]:.3f}  {ps}')
print('=' * 95)
print('Mantel one-sided tests (5 species):')
for k, d in sup_mantel.items():
    for sub, (r, p) in d.items():
        print(f'  {f"{k} {sub}".strip()}: r = {r:.3f}, p = {p:.4f}')

for (resp_name, eco_name), t in sup_tables.items():
    if resp_name == 'Shape':
        print('=' * 95)
        print(f'{resp_name} | ecology: {eco_name}')
        print(t[['model', 'logLik', 'K', 'AICc', 'dAICc', 'weight']].to_string(index=False))
        for mname, out in sup_mrm[(resp_name, eco_name)].items():
            terms = ['int', 'eco', 'phy'] if mname == 'M3' else (['int', 'eco'] if mname == 'M1' else ['int', 'phy'])
            ps = ' '.join(f'{t_}={b:+.3f}(p={p:.4f})' for t_, b, p in zip(terms, out['beta'], out['p']))
            print(f'  MRM {mname}: R2={out["R2"]:.3f}  {ps}')

# ------------------------------------------------------------------------------------------------ #

## Summary ##

# ------------------------------------------------------------------------------------------------ #
	  
print()
print('=== SUMMARY (main 8-species | supplemental 5-species) ===')
for resp in ['Size', 'Shape', 'Combined']:
    rn = [k for k in RESPONSES_R if k.startswith(resp)][0]
    t8 = real_tables[(rn, "Schoener's D")]
    t5 = sup_tables[(resp, "Schoener's D")]
    best8 = t8.iloc[0]; best5 = t5.iloc[0]
    m8 = real_mrm[(rn, "Schoener's D")]['M3']; m5 = sup_mrm[(resp, "Schoener's D")]['M3']
    print(f'{resp}:')
    print(f'  8sp best: {best8["model"]} (w={best8["weight"]:.2f}) | M3: eco p={m8["p"][1]:.3f}, phy p={m8["p"][2]:.3f}, R2={m8["R2"]:.3f}')
    print(f'  5sp best: {best5["model"]} (w={best5["weight"]:.2f}) | M3: eco p={m5["p"][1]:.3f}, phy p={m5["p"][2]:.3f}, R2={m5["R2"]:.3f}')
print()
print('Mantel 8sp: size~phy r=%.3f p=%.4f | shape~phy r=%.3f p=%.4f | size~eco r=%.3f p=%.4f | shape~eco r=%.3f p=%.4f | eco~phy r=%.3f p=%.4f'
      % (real_mantel['Size (|dlogCS|)']['morph~phy'][0], real_mantel['Size (|dlogCS|)']['morph~phy'][1],
         real_mantel['Shape (PC space)']['morph~phy'][0], real_mantel['Shape (PC space)']['morph~phy'][1],
         real_mantel['Size (|dlogCS|)']['morph~ecoD'][0], real_mantel['Size (|dlogCS|)']['morph~ecoD'][1],
         real_mantel['Shape (PC space)']['morph~ecoD'][0], real_mantel['Shape (PC space)']['morph~ecoD'][1],
         real_mantel['ecoD~phy'][''][0], real_mantel['ecoD~phy'][''][1]))
print('Mantel 5sp: size~phy r=%.3f p=%.4f | shape~phy r=%.3f p=%.4f | size~eco r=%.3f p=%.4f | shape~eco r=%.3f p=%.4f | eco~phy r=%.3f p=%.4f'
      % (sup_mantel['Size']['morph~phy'][0], sup_mantel['Size']['morph~phy'][1],
         sup_mantel['Shape']['morph~phy'][0], sup_mantel['Shape']['morph~phy'][1],
         sup_mantel['Size']['morph~ecoD'][0], sup_mantel['Size']['morph~ecoD'][1],
         sup_mantel['Shape']['morph~ecoD'][0], sup_mantel['Shape']['morph~ecoD'][1],
         sup_mantel['ecoD~phy'][''][0], sup_mantel['ecoD~phy'][''][1]))
print()
print('CS by species (mm):')
print(datR.groupby('species')['CS_mm'].agg(['count', 'min', 'median', 'max']).round(1))
print()
print('Shape PCs (variance %):', np.round(var_expl[:5] * 100, 2))
