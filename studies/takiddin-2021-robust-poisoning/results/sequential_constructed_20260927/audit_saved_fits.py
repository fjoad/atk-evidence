import json,sys,subprocess
from pathlib import Path
from unittest.mock import patch
import numpy as np
root=Path.cwd()
sys.path.insert(0,str(root/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_constructed as G
G.R.configure_tensorflow(False)
import keras
import tensorflow as tf
run=root/'data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1-recovery'
r=json.loads((run/'result.json').read_text());report={'scope':'read-only audit of constructed saved fits; zero further updates','cases':{}}
assert G.digest(run/'constructed_data.npz')==r['data_sha256']
with np.load(run/'constructed_data.npz') as f: arrays={k:f[k] for k in f.files}
for k,v in G.fixture_data().items():np.testing.assert_array_equal(v,arrays[k])
for rel,h in r['source_sha256'].items():
 source=subprocess.check_output(['git','show','b9d8223:'+rel])
 import hashlib
 assert hashlib.sha256(source).hexdigest()==h,rel
for rel,h in r['recovery']['original_files_sha256'].items():assert G.digest(root/r['recovery']['original_attempt']/rel)==h
G.M.register_sequential_layer()
for name,case in r['cases'].items():
 d=run/name
 for filename,h in case['artifact_sha256'].items():assert G.digest(d/filename)==h,filename
 model=keras.models.load_model(d/'final.keras')
 assert int(model.optimizer.iterations.numpy())==300
 assert G.R.weight_hash(model.get_weights())==case['final_weights_sha256']
 with np.load(d/'initial_weights.npz') as f: initial=[f[f'w{i}'] for i in range(len(f.files))]
 assert G.R.weight_hash(initial)==case['initial_weights_sha256']
 y=arrays['test_y'] if name=='normal' else 1-arrays['test_y']
 x=arrays['test_x']; probability=model(x).numpy()
 np.testing.assert_array_equal(G.measurements(y,probability)['accuracy'],case['test']['accuracy'])
 assert G.measurements(y,probability)==case['test']
 front=model.get_layer('attention_decoder')
 layers=[front.output[0]]+[model.get_layer(f'sequence_gru_{i}').output for i in range(1,9)]+[model.get_layer('classifier_hidden').output,model.outputs[0]]
 names=['intermediate']+[f'gru_{i}' for i in range(1,9)]+['classifier_hidden','probability']
 values=keras.Model(model.input,layers)(x)
 observations={n:{'min':float(v.numpy().min()),'max':float(v.numpy().max()),'std':float(v.numpy().std()),'nonzero_fraction':float(np.mean(v.numpy()!=0))} for n,v in zip(names,values)}
 raw=[];original=front.projection.call
 def capture(inputs, *args, **kwargs):
  raw.append((tf.matmul(inputs,front.projection.kernel)+front.projection.bias).numpy())
  return original(inputs, *args, **kwargs)
 with patch.object(front.projection,'call',capture):sequence,attention=front(x)
 with np.load(d/'outputs.npz') as f:
  np.testing.assert_array_equal(probability,f['probability'])
  np.testing.assert_array_equal(sequence.numpy(),f['sequence'])
  np.testing.assert_array_equal(attention.numpy(),f['attention'])
 assert len(raw)==8
 raw=np.stack(raw,axis=1)
 observations['intermediate_raw_preactivation']={'min':float(raw.min()),'max':float(raw.max()),'nonpositive_fraction':float(np.mean(raw<=0))}
 report['cases'][name]={'artifact_hashes_verified':True,'fresh_reload_exact':True,'optimizer_updates':300,'measurements':case['test'],'layers':observations}
 if name=='normal':report['complement_against_reversed_labels']=G.measurements(1-arrays['test_y'],1-probability)
report['paired_initial_weights']=r['paired_initial_weights']
report['gate_passed']=r['gate_passed']
print(json.dumps(report,indent=2,allow_nan=False))
