"""Descriptive reductions of saved arrays only; no model calls or updates."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_constructed as G
from analyze_results import digest

raw=ROOT/'data/derived/takiddin-2021-robust-poisoning/sequential-saved-state-20261002-attempt1'
r=json.loads((raw/'result.json').read_text())
stages=['intermediate',*[f'gru_{i}' for i in range(1,9)],'classifier_hidden','probability']
summary={'scope':'post-run descriptive saved-array reductions; no additional inference',
         'result_sha256':digest(raw/'result.json'), 'script_sha256':digest(__file__), 'cases':{}}
for key,o in r['observations'].items():
    case={'first_profile_constant_native_stage':next((k for k in stages if o['arrays'][k]['max_axis0_range']==0),None),
          'native_profile_ranges':{k:o['arrays'][k]['max_axis0_range'] for k in stages},
          'weight_gradient_group_l2':{},'time_sequences':{}}
    for condition,g in o['gradients'].items():
        groups={k:0. for k in G.GROUPS}
        for array,target in g['arrays'].items():
            if '/' in target:groups[G.group(target)]+=o['arrays'][array]['l2']**2
        case['weight_gradient_group_l2'][condition]={k:float(np.sqrt(v)) for k,v in groups.items()}
    assert digest(raw/(key+'.npz'))==r['artifact_sha256'][key+'.npz']
    with np.load(raw/(key+'.npz'),allow_pickle=False) as a:
        for name in ['encoder_memory','decoder_hidden','bridge_logits','intermediate',*[f'gru_{i}' for i in range(1,8)]]:
            v=a[name].astype(np.float64)
            delta=np.ptp(v,axis=0).max(axis=1)
            peak=np.max(np.abs(v),axis=(0,2))
            case['time_sequences'][name]={'max_absolute_by_step':peak.tolist(),
                'max_profile_range_by_step':delta.tolist(),
                'all_profiles_equal_from_step':next((i+1 for i in range(len(delta)) if np.all(delta[i:]==0)),None)}
    summary['cases'][key]=case
print(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False))
