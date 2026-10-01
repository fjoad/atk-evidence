from pathlib import Path
import sys,json,hashlib,subprocess,math
import numpy as np
root=Path.cwd();sys.path.insert(0,str(root/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_constructed as G
G.R.configure_tensorflow(False)
import keras
run=root/'data/derived/takiddin-2021-robust-poisoning/sequential-interface-20261001-attempt1'
r=json.loads((run/'result.json').read_text())
assert r['status']=='complete'
for p,h in r['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',r['code_commit']+':'+p])).hexdigest()==h,p
arrays=G.fixture_data();G.M.register_sequential_layer()
report={'scope':'constructed interface audit; no updates','branches':{},'historical_relu_reload':{}}
for activation,branch in r['branches'].items():
 report['branches'][activation]={}
 for name,c in branch['cases'].items():
  d=run/activation/name
  for f,h in c['artifact_sha256'].items():assert G.digest(d/f)==h,f
  history=json.loads((d/'history.json').read_text());assert len(history)==300 and history[-1]['updates']==300
  model=keras.models.load_model(d/'final.keras')
  assert model.count_params()==9240802 and model.get_layer('attention_decoder').bridge_activation==activation
  assert int(model.optimizer.iterations.numpy())==300
  assert G.R.weight_hash(model.get_weights())==c['final_weights_sha256']
  with np.load(d/'initial_weights.npz') as f:weights=[f[f'w{i}'] for i in range(len(f.files))]
  assert G.R.weight_hash(weights)==r['historical_relu_initial_weights_sha256']
  y=arrays['test_y'] if name=='normal' else 1-arrays['test_y']
  actual=model(arrays['test_x']).numpy()
  with np.load(d/'outputs.npz') as f:
   np.testing.assert_array_equal(actual,f['probability']);np.testing.assert_array_equal(y,f['observed_test_y'])
   probe=keras.Model(model.input,model.get_layer('attention_decoder').output)
   sequence,attention=probe(arrays['test_x'])
   np.testing.assert_array_equal(sequence.numpy(),f['sequence']);np.testing.assert_array_equal(attention.numpy(),f['attention'])
  p=np.clip(actual.astype('float64'),1e-7,1-1e-7)
  bce=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
  accuracy=float(np.mean((actual>.5)==y))
  assert accuracy==c['test']['accuracy'] and abs(bce-c['test']['bce'])<1e-14
  assert accuracy>=.9 and bce<math.log(2)/2
  report['branches'][activation][name]={'hashes_verified':True,'fresh_reload_exact':True,'updates':300,'accuracy':accuracy,'bce':bce}
old=root/'data/derived/takiddin-2021-robust-poisoning/sequential-full-width-20260928-attempt1/normal'
model=keras.models.load_model(old/'final.keras')
assert model.get_layer('attention_decoder').bridge_activation=='relu'
with np.load(old/'outputs.npz') as f:
 np.testing.assert_array_equal(model(arrays['test_x']).numpy(),f['probability'])
 probe=keras.Model(model.input,model.get_layer('attention_decoder').output)
 sequence,attention=probe(arrays['test_x'])
 np.testing.assert_array_equal(sequence.numpy(),f['sequence']);np.testing.assert_array_equal(attention.numpy(),f['attention'])
report['historical_relu_reload']={'activation_default_preserved':True,'probability_sequence_attention_exact':True}
print(json.dumps(report,indent=2))
