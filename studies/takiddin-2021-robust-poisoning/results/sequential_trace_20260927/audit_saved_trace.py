import json,sys,hashlib,subprocess
from pathlib import Path
import numpy as np
root=Path.cwd()
sys.path.insert(0,str(root/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_constructed as G
G.R.configure_tensorflow(False)
import tensorflow as tf
import keras
run=root/'data/derived/takiddin-2021-robust-poisoning/sequential-trace-20260927-attempt1'
old=root/'data/derived/takiddin-2021-robust-poisoning/sequential-constructed-20260927-attempt1-recovery'
r=json.loads((run/'result.json').read_text())
assert r['status']=='complete' and r['updates_completed']==50
for p,h in r['artifact_sha256'].items():assert G.digest(run/p)==h,p
for p,h in r['original_file_sha256'].items():assert G.digest(old/p)==h,p
for p,h in r['source_sha256'].items():assert hashlib.sha256(subprocess.check_output(['git','show',r['code_commit']+':'+p])).hexdigest()==h,p
history=json.loads((old/'reversed/history.json').read_text())
rows=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
assert [x['step'] for x in rows]==list(range(51))
assert [x['fit_loss'] for x in rows[1:]]==[x['loss'] for x in history[:50]]
with np.load(old/'constructed_data.npz') as f: data={k:f[k] for k in f.files}
with np.load(old/'reversed/initial_weights.npz') as old_initial,np.load(run/'step-000.npz') as snap:
 for key in old_initial.files:np.testing.assert_array_equal(old_initial[key],snap[key])
for row in rows:
 with np.load(run/f"step-{row['step']:03d}.npz") as snap:
  assert int(snap['opt0'])==row['step']
  for split in ('train','test'):
   p=snap[split+'_probability'].ravel().astype('float64'); y=1-data[split+'_y'].ravel()
   accuracy=float(np.mean((p>.5)==y))
   loss=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)))
   assert accuracy==row[split]['metrics']['accuracy']
   assert abs(loss-row[split]['metrics']['bce'])<1e-14
   for stage,s in row[split]['layers'].items():
    v=snap[split+'_'+stage]
    assert float(np.mean(v>0))==s[3]
    assert float(v.min())==s[0] and float(v.max())==s[1]
report={'scope':'saved constructed states only; zero updates','snapshot_count':51,'all_hashes_verified':True,'all_50_old_losses_exact':True,'originals_unchanged':True,'inference_checks':{}}
model=G.M.sequential_ensemble(G.MODEL_SEED,**G.SMALL)
probe=keras.Model(model.input,[model.get_layer('attention_decoder').output[0],model.get_layer('sequence_gru_8').output,model.get_layer('classifier_hidden').output,model.outputs[0]])
for step in (0,1,2,50):
 with np.load(run/f'step-{step:03d}.npz') as s:
  model.set_weights([s[f'w{i}'] for i in range(len(model.weights))])
  values=probe(data['train_x'])
  differences=[]
  for name,v in zip(('intermediate','gru_8','classifier_hidden','probability'),values):
   np.testing.assert_allclose(v.numpy(),s['train_'+name],rtol=1e-5,atol=1e-7)
   differences.append(float(np.max(np.abs(v.numpy()-s['train_'+name]))))
  grads=G.gradient_report(model,data['train_x'],1-data['train_y'])
  assert all((grads['group_l2'][g]==0)==(rows[step]['gradient_group_l2'][g]==0) for g in G.GROUPS)
  report['inference_checks'][str(step)]={'max_absolute_difference':max(differences),'gradient_zero_pattern_matches':True}
layer=model.get_layer('sequence_gru_8')
indices=[r['weight_paths'].index(v.path) for v in layer.weights]
with np.load(run/'step-001.npz') as a,np.load(run/'step-002.npz') as b:
 x0,x1=a['train_gru_7'],b['train_gru_7']
 w0,w1=[a[f'w{i}'] for i in indices],[b[f'w{i}'] for i in indices]
 saved0,saved1=a['train_gru_8'],b['train_gru_8']

def stats(x):return {'min':float(x.min()),'max':float(x.max()),'positive_fraction':float(np.mean(x>0))}
def manual(x,params):
 k,u,b=params; width=u.shape[0];h=np.zeros((x.shape[0],width),dtype='float64'); pre=[]
 k,u,b=[z.astype('float64') for z in params]
 def sigmoid(z):return 1/(1+np.exp(-z))
 for t in range(x.shape[1]):
  projected=x[:,t,:].astype('float64')@k+b
  z=sigmoid(projected[:,:width]+h@u[:,:width])
  reset=sigmoid(projected[:,width:2*width]+h@u[:,width:2*width])
  candidate=projected[:,2*width:]+(reset*h)@u[:,2*width:]
  pre.append(candidate)
  h=z*h+(1-z)*np.maximum(candidate,0)
 return h,np.stack(pre,axis=1)

def forward(x,w):
 layer.set_weights(w)
 native=layer(x).numpy()
 calculated,candidate=manual(x,w)
 np.testing.assert_allclose(calculated,native,rtol=1e-5,atol=1e-7)
 return native,{'output':stats(native),'candidate_preactivation':stats(candidate),'manual_native_max_error':float(np.max(np.abs(calculated-native)))}
variants={'old_input_old_params':(x0,w0),'old_input_new_params':(x0,w1),'new_input_old_params':(x1,w0),'new_input_new_params':(x1,w1)}
old_candidate=[v.copy() for v in w1];old_candidate[2][16:]=w0[2][16:]
new_candidate=[v.copy() for v in w0];new_candidate[2][16:]=w1[2][16:]
variants['new_input_new_params_old_candidate_bias']=(x1,old_candidate)
variants['old_input_old_params_new_candidate_bias']=(x0,new_candidate)
report['gru8_counterfactuals']={}
for name,(x,w) in variants.items():
 actual,summary=forward(x,w)
 if name=='old_input_old_params':np.testing.assert_allclose(actual,saved0,rtol=1e-5,atol=1e-7)
 if name=='new_input_new_params':np.testing.assert_array_equal(actual,saved1)
 report['gru8_counterfactuals'][name]=summary
report['candidate_bias_before']=w0[2][16:].tolist();report['candidate_bias_after']=w1[2][16:].tolist()
for p,h in r['artifact_sha256'].items():assert G.digest(run/p)==h,p
for p,h in r['original_file_sha256'].items():assert G.digest(old/p)==h,p
print(json.dumps(report,indent=2,allow_nan=False))
