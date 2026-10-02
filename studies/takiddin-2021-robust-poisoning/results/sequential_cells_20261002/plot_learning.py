"""Plot preserved histories; no additional model call or metric selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
for name,label,color,style in [('tanh_normal','tanh / normal','#247ba0','-'),
                             ('tanh_reversed','tanh / reversed','#d66a1c','--'),
                             ('relu_normal','ReLU / normal','#555555','-')]:
    rows=json.loads((HERE/(name+'_history.json')).read_text())
    valid=[r for r in rows if r['loss'] is not None]
    ax.plot([r['updates'] for r in valid],[r['loss'] for r in valid],label=label,color=color,ls=style,lw=2)
ax.axvline(245,color='#777777',ls=':',lw=1)
ax.text(150,1e-4,'ReLU: nonfinite loss at update 245\nReversed reference held by stop rule',fontsize=9)
ax.set(yscale='log',xlim=(1,300),ylim=(1e-10,2),xlabel='Optimizer updates',ylabel='Training binary cross-entropy',
       title='48-step constructed learning with one fixed initialization')
ax.legend(frameon=False,loc='center left',bbox_to_anchor=(1.01,.6));ax.grid(alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.savefig(HERE/'learning-curves.png',dpi=200);fig.savefig(HERE/'learning-curves.svg');plt.close(fig)
