import json,shutil,sys
from pathlib import Path
import numpy as np
root=Path.cwd()
sys.path.insert(0,str(root/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_constructed as G
G.R.configure_tensorflow(False)
import keras
import tensorflow as tf
old=root/'data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1'
new=root/'data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1-recovery'
r=json.loads((old/'result.json').read_text())
assert r['status']=='started' and r['cases']=={}
assert r['source_sha256']['studies/takiddin-2021-robust-poisoning/reproduction/models.py']==G.digest(G.STUDY/'reproduction/models.py')
assert r['data_sha256']==G.digest(old/'constructed_data.npz')
with np.load(old/'constructed_data.npz') as f: arrays={k:f[k] for k in f.files}
for k,v in G.fixture_data().items():np.testing.assert_array_equal(v,arrays[k])
new.mkdir(exist_ok=False)
shutil.copy2(old/'constructed_data.npz',new/'constructed_data.npz')
shutil.copytree(old/'normal',new/'normal')
G.M.register_sequential_layer()
with tf.device('/CPU:0'):
 model=keras.models.load_model(old/'normal/final.keras')
 assert int(model.optimizer.iterations.numpy())==300
 with np.load(old/'normal/outputs.npz') as f:
  np.testing.assert_array_equal(model(arrays['test_x']).numpy(),f['probability'])
  probe=keras.Model(model.input,model.get_layer('attention_decoder').output)
  seq,att=probe(arrays['test_x'])
  np.testing.assert_array_equal(seq.numpy(),f['sequence'])
  np.testing.assert_array_equal(att.numpy(),f['attention'])
 with np.load(old/'normal/initial_weights.npz') as f:
  initial=[f[f'w{i}'] for i in range(len(f.files))]
 initial_by_path={v.path:w for v,w in zip(model.weights,initial)}
 case=json.loads((old/'normal/result.json').read_text())
 assert case['initial_weights_sha256']==G.R.weight_hash(initial)
 history=json.loads((old/'normal/history.json').read_text())
 assert len(history)==300 and history[-1]['updates']==300
 case.update(status='complete',updates=300,elapsed_seconds=history[-1]['elapsed_seconds'],
             report_recovered_without_refit=True)
 case=G.finish_record(new/'normal',model,case,initial_by_path,arrays['train_x'],arrays['train_y'],arrays['test_x'],arrays['test_y'])
 r['cases']['normal']=case
 r['recovery']={'original_attempt':str(old.relative_to(root)),'normal_refitted':False,
   'reason':'np.bool_ in final report failed JSON encoding after saved 300-update fit',
   'runner_sha256':G.digest(Path(G.__file__)),'script_sha256':G.digest(Path(__file__)),
   'original_files_sha256':{str(p.relative_to(old)):G.digest(p) for p in old.rglob('*') if p.is_file()}}
 (new/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 r['cases']['reversed']=G.run_one(new/'reversed',arrays,True)
 cases=list(r['cases'].values())
 r['paired_initial_weights']=len({c['initial_weights_sha256'] for c in cases})==1
 r['gate_passed']=r['paired_initial_weights'] and all(c['learning_gate_passed'] and c['gradient_group_gate_passed'] and c['weights_finite'] for c in cases)
 r['status']='passed' if r['gate_passed'] else 'failed_gate'
 (new/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 for rel,h in r['recovery']['original_files_sha256'].items():assert G.digest(old/rel)==h
 print(json.dumps({'status':r['status'],'cases':{k:{q:v[q] for q in ('status','updates','test','learning_gate_passed','gradient_group_gate_passed')} for k,v in r['cases'].items()}},indent=2))
