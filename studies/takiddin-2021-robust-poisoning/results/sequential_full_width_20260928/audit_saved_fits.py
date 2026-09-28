from pathlib import Path
from unittest.mock import patch
import sys,json,hashlib,subprocess
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_constructed as G
G.R.configure_tensorflow(False)
import tensorflow as tf
import keras
run=root/'data/derived/takiddin-2021-robust-poisoning/sequential-full-width-20260928-attempt1'
r=json.loads((run/'result.json').read_text())
files={str(p.relative_to(run)):G.digest(p) for p in run.rglob('*') if p.is_file()}
for path,h in r['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',r['code_commit']+':'+path])).hexdigest()==h,path
old=root/'data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1-recovery/constructed_data.npz'
assert G.digest(old)==r['previous_data_sha256']
with np.load(old) as f: arrays={k:f[k] for k in f.files}
for k,v in G.fixture_data().items():np.testing.assert_array_equal(v,arrays[k])
report={'scope':'read-only full-width eight-step software artifacts; zero updates','cases':{},'sources_and_data_verified':True}
G.M.register_sequential_layer()
def summary(v):
 v=np.asarray(v)
 return {'min':float(v.min()),'max':float(v.max()),'positive_fraction':float(np.mean(v>0)),'std':float(v.std())}
for name,record in r['cases'].items():
 case=run/name
 for f,h in record['artifact_sha256'].items():assert G.digest(case/f)==h,f
 history=json.loads((case/'history.json').read_text())
 assert len(history)==300 and [x['updates'] for x in history]==list(range(1,301))
 model=keras.models.load_model(case/'final.keras')
 assert model.count_params()==9240802 and int(model.optimizer.iterations.numpy())==300
 final=model.get_weights()
 assert G.R.weight_hash(final)==record['final_weights_sha256']
 with np.load(case/'initial_weights.npz') as f: initial=[f[f'w{i}'] for i in range(len(f.files))]
 assert G.R.weight_hash(initial)==record['initial_weights_sha256']
 labels=arrays['test_y'] if name=='normal' else 1-arrays['test_y']
 with np.load(case/'outputs.npz') as f:
  np.testing.assert_array_equal(model(arrays['test_x']).numpy(),f['probability'])
  np.testing.assert_array_equal(labels,f['observed_test_y'])
  assert G.measurements(labels,f['probability'])==record['test']
  output_reference={k:f[k] for k in ('sequence','attention')}
 front=model.get_layer('attention_decoder')
 names=['intermediate']+[f'gru_{i}' for i in range(1,9)]+['classifier_hidden','probability']
 tensors=[front.output[0]]+[model.get_layer(f'sequence_gru_{i}').output for i in range(1,9)]+[model.get_layer('classifier_hidden').output,model.outputs[0]]
 probe=keras.Model(model.input,tensors)
 states={}
 for state,weights in (('initial',initial),('final',final)):
  model.set_weights(weights);states[state]={}
  for split in ('train','test'):
   x=arrays[split+'_x'];y=arrays[split+'_y'] if name=='normal' else 1-arrays[split+'_y']
   values=probe(x)
   obs={n:summary(v.numpy()) for n,v in zip(names,values)}
   raw=[];incoming=[];original=front.projection.call
   def capture(inputs,*args,**kwargs):
    incoming.append(inputs.numpy())
    raw.append((tf.matmul(inputs,front.projection.kernel)+front.projection.bias).numpy())
    return original(inputs,*args,**kwargs)
   with patch.object(front.projection,'call',capture):sequence,attention=front(x)
   assert len(raw)==8
   obs['projection_raw']=summary(np.stack(raw,axis=1));obs['projection_input']=summary(np.stack(incoming,axis=1))
   if state=='final' and split=='test':
    np.testing.assert_array_equal(sequence.numpy(),output_reference['sequence'])
    np.testing.assert_array_equal(attention.numpy(),output_reference['attention'])
   states[state][split]={'layers':obs,'metrics':G.measurements(y,values[-1].numpy())}
  grad=G.gradient_report(model,arrays['train_x'],arrays['train_y'] if name=='normal' else 1-arrays['train_y'])
  assert grad['group_l2']==record[state+'_gradient']['group_l2']
  states[state]['gradient_group_l2']=grad['group_l2']
 model.set_weights(final)
 assert G.R.weight_hash(model.get_weights())==record['final_weights_sha256']
 assert int(model.optimizer.iterations.numpy())==300
 report['cases'][name]={'artifact_hashes_verified':True,'fresh_reload_exact':True,'updates':300,'states':states,
                       'unchanged_biases':all(np.array_equal(a,b) for v,a,b in zip(model.weights,initial,final) if 'bias' in v.path)}
report['paired_initial_weights']=len({c['initial_weights_sha256'] for c in r['cases'].values()})==1
report['identical_final_weights_across_label_orientations']=len({c['final_weights_sha256'] for c in r['cases'].values()})==1
report['all_original_full_width_files_unchanged']=all(G.digest(run/p)==h for p,h in files.items())
assert report['all_original_full_width_files_unchanged']
print(json.dumps(report,indent=2,allow_nan=False))
