"""Read saved gate records and witnesses; no model calls or parameter changes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RAW=ROOT/'data/derived/takiddin-2021-robust-poisoning/sequential-gates-20261002-attempt1'
r=json.loads((RAW/'result.json').read_text())
summary={'scope':'post-run reductions of saved gates; no inference',
    'raw_result_sha256':hashlib.sha256((RAW/'result.json').read_bytes()).hexdigest(),
    'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'states':{}}
for name,state in r['states'].items():
    cells={}
    for cell,c in state['cells'].items():
        p=c['per_step'];row,unit=c['maximum_final_hidden_location'];j=c['witness_rows'].index(row)
        q={'maximum_final_hidden_location':[row,unit],
           'hidden_max_by_step':[v['max'] for v in p['h']],
           'candidate_positive_count_by_step':[v['positive_count'] for v in p['candidate']],
           'candidate_pre_max_by_step':[v['max'] for v in p['candidate_pre']],
           'blocked_by_bias_by_step':c['blocked_by_bias_per_step'],
           'recurrently_supported_injection_by_step':c['recurrently_supported_injection_per_step']}
        if cell=='gru_8':
            q.update(candidate_zero_tail_from_step=c['candidate_zero_tail_from_step'],
                update_min_by_step=[v['min'] for v in p['update_gate']],
                update_max_by_step=[v['max'] for v in p['update_gate']],
                final_identity_error=c['weighted_injection_identity_max_error'])
        with np.load(RAW/name/(cell+'.npz'),allow_pickle=False) as a:
            fields=['h','candidate_pre','input_candidate','recurrent_candidate','bias_candidate','retained','injected']
            fields+=['update_gate'] if cell=='gru_8' else ['c_previous','input_gate','forget_gate','output_gate']
            q['final_maximum_witness']={k:[float(v) for v in a[k][j,:,unit]] for k in fields}
        cells[cell]=q
    summary['states'][name]=cells
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')

fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
t=np.arange(1,49)
for state,label,color in [('initial','Initial','#247ba0'),('final_p00','Unpoisoned','#d66a1c'),('final_p30','Poisoned','#7b4ab0')]:
    axes[0].plot(t,summary['states'][state]['gru_8']['hidden_max_by_step'],label=label,color=color,lw=2)
axes[0].axvline(4,color='#666666',ls=':',lw=1)
axes[0].text(23,1e-6,'Unpoisoned: no new candidate\nfrom step 4 onward',fontsize=9,color='#945017')
axes[0].set(title='GRU8 state across the forward pass',xlabel='Time step',ylabel='Maximum hidden activation',yscale='log')
axes[0].legend(frameon=False,loc='lower left')
for cell,label,color in [('encoder_1','Encoder 1','#367d50'),('encoder_3','Encoder 3','#247ba0'),('decoder_3','Decoder 3','#7b4ab0')]:
    axes[1].plot(t,summary['states']['final_p30'][cell]['hidden_max_by_step'],label=label,color=color,lw=2)
axes[1].set(title='Poisoned LSTM state growth',xlabel='Step within each LSTM stage',ylabel='Maximum hidden activation',yscale='log')
axes[1].legend(frameon=False,loc='upper left')
for ax in axes:
    ax.set_xlim(1,48);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Saved weights: recurrent dynamics on 100 fixed test profiles',fontsize=13)
fig.savefig(HERE/'gate-dynamics.png',dpi=200)
fig.savefig(HERE/'gate-dynamics.svg')
plt.close(fig)
