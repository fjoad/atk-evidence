"""One new cell interpretation and its matched 48-step software reference."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import sequential_constructed as G
import sequential_saved_state as D

ROOT,STUDY,M,R=G.ROOT,G.STUDY,G.M,G.R
INITIAL_GPU_SHA='11c3bf3746800eebd9e695a9dda3abaafc9e7c445cbe8044a2110ca4da77f549'


def fixture_data():
    rng=np.random.default_rng(20260926)
    y=np.repeat(np.array([0.,1.],dtype=np.float32),16)[:,None]
    return {k:v for split in ('train','test') for k,v in (
        (split+'_x',(.75*(2*y[:,:,None]-1)+.05*rng.normal(size=(32,48,1))).astype(np.float32)),
        (split+'_y',y.copy()))}


def layer_snapshot(model,x,path):
    import tensorflow as tf
    import keras
    names=['intermediate',*[f'gru_{i}' for i in range(1,9)],'classifier_hidden','probability']
    front=model.get_layer('attention_decoder')
    probe=keras.Model(model.input,[front.output[0],*[model.get_layer(f'sequence_gru_{i}').output for i in range(1,9)],
        model.get_layer('classifier_hidden').output,model.outputs[0]])
    values=probe(x,training=False)
    arrays={k:v.numpy() for k,v in zip(names,values)}
    trace=D.frontend_trace(front,tf.convert_to_tensor(x))
    np.testing.assert_array_equal(arrays['intermediate'],trace['traced_intermediate'])
    arrays.update({k:trace[k] for k in ('encoder_memory','decoder_hidden','bridge_logits')})
    np.savez_compressed(path,**arrays)
    result={k:D.describe(v) for k,v in arrays.items()}
    if model.get_layer('attention_decoder').cell_activation=='tanh':
        for k in ['encoder_memory','decoder_hidden',*[f'gru_{i}' for i in range(1,9)]]:
            if result[k]['finite_count']!=result[k]['size'] or max(abs(result[k]['min']),abs(result[k]['max']))>1.000001:
                raise ValueError('Tanh hidden-state bound violated')
    return result


def case_passes(c):
    return bool(c.get('status')=='complete' and c.get('updates')==300
        and c.get('learning_gate_passed') and c.get('gradient_group_gate_passed')
        and c.get('weights_finite') and c.get('reload_exact')
        and c.get('maximum_constrained_norm') is not None and c['maximum_constrained_norm']<=1.00001)


def run_case(output,arrays,activation,reverse,*,updates=300,fit_guard_seconds=360,fixture_dimensions=None):
    import tensorflow as tf
    import keras
    output.mkdir();dims={} if fixture_dimensions is None else dict(fixture_dimensions)
    if set(dims)-{'encoder_units','gru_units','dense_units'}:raise ValueError('Invalid software fixture dimensions')
    model=M.sequential_ensemble(20260920,timesteps=48,bridge_activation='sigmoid',cell_activation=activation,**dims)
    if fixture_dimensions is None and model.count_params()!=9240802:raise ValueError('Parameter count differs')
    if fixture_dimensions is None and any('GPU:0' not in v.value.device for v in model.trainable_weights):raise ValueError('Wrong device')
    x,xt=arrays['train_x'],arrays['test_x'];y=1-arrays['train_y'] if reverse else arrays['train_y'];yt=1-arrays['test_y'] if reverse else arrays['test_y']
    initial=model.get_weights();initial_trainable={v.path:v.numpy().copy() for v in model.trainable_weights}
    np.savez_compressed(output/'initial_weights.npz',**{f'w{i}':w for i,w in enumerate(initial)})
    c={'status':'started','cell_activation':activation,'bridge_activation':'sigmoid','label_orientation':'reversed' if reverse else 'normal',
        'parameter_count':model.count_params(),'timesteps':48,'requested_updates':updates,'fit_guard_seconds':fit_guard_seconds,
        'model_seed':20260920,'initial_weights_sha256':R.weight_hash(initial),
        'initial_test':G.measurements(yt,model(xt).numpy()),'initial_gradient':G.gradient_report(model,x,y)}
    if fixture_dimensions is None and c['initial_weights_sha256']!=INITIAL_GPU_SHA:raise ValueError('Prior GPU initialization differs')
    np.savez_compressed(output/'initial_predictions.npz',train_probability=model(x).numpy(),test_probability=model(xt).numpy(),observed_train_y=y,observed_test_y=yt)
    c['initial_layers']=layer_snapshot(model,xt,output/'initial_layers.npz')
    save=lambda:(output/'result.json').write_text(json.dumps(c,indent=2,allow_nan=False)+'\n')
    save();history=[];stops=[];start=time.monotonic()
    class Recorder(keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            loss=float(logs['loss']);n=int(self.model.optimizer.iterations.numpy())
            history.append({'epoch':epoch+1,'updates':n,'loss':loss if np.isfinite(loss) else None,'elapsed_seconds':time.monotonic()-start})
            (output/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False)+'\n')
            if epoch==0 or (epoch+1)%50==0:print(json.dumps({'activation':activation,'orientation':c['label_orientation'],**history[-1]}),flush=True)
            if not np.isfinite(loss):stops.append('nonfinite_loss');self.model.stop_training=True
            elif time.monotonic()-start>=fit_guard_seconds and epoch+1<updates:stops.append('time_guard');self.model.stop_training=True
    options=tf.data.Options();options.deterministic=True;options.threading.private_threadpool_size=1;options.threading.max_intra_op_parallelism=1
    dataset=tf.data.Dataset.from_tensor_slices((x,y)).batch(32).with_options(options)
    try:
        model.fit(dataset,epochs=updates,shuffle=False,verbose=0,callbacks=[Recorder()])
        c['status']=stops[0] if stops else 'complete'
    except Exception as exc:c.update(status='error',error=f'{type(exc).__name__}: {exc}')
    c.update(updates=int(model.optimizer.iterations.numpy()),elapsed_seconds=time.monotonic()-start)
    model.save(output/'final.keras');save()
    c=G.finish_record(output,model,c,initial_trainable,x,y,xt,yt)
    c['final_layers']=layer_snapshot(model,xt,output/'final_layers.npz')
    np.savez_compressed(output/'predictions.npz',train_probability=model(x).numpy(),test_probability=model(xt).numpy(),observed_train_y=y,observed_test_y=yt)
    c['model_json']=json.loads(model.to_json());c['optimizer_config']=model.optimizer.get_config()
    restored=keras.models.load_model(output/'final.keras')
    p=keras.Model(restored.input,[restored.outputs[0],*restored.get_layer('attention_decoder').output])
    restored_probability,seq,att=p(xt)
    np.testing.assert_array_equal(restored_probability.numpy(),model(xt).numpy())
    with np.load(output/'outputs.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(seq.numpy(),z['sequence']);np.testing.assert_array_equal(att.numpy(),z['attention'])
    if R.weight_hash(restored.get_weights())!=c['final_weights_sha256'] or int(restored.optimizer.iterations.numpy())!=c['updates']:
        raise ValueError('Reloaded state differs')
    c['reload_exact']=True;c['case_gate_passed']=case_passes(c)
    c['artifact_sha256']={p.name:R.digest(p) for p in output.iterdir() if p.name!='result.json'}
    save();print(json.dumps({'case':activation+'_'+c['label_orientation'],'status':c['status'],'test':c['test'],'passed':c['case_gate_passed']}),flush=True)
    return c


def run(output,previous):
    R.require_compute();output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__).resolve(),Path(G.__file__).resolve(),Path(D.__file__).resolve(),
        STUDY/'reproduction/models.py',STUDY/'SEQUENTIAL_RECURRENT_CONTROL.md',
        STUDY/'checks/run_sequential_cell_learning.sbatch',ROOT/'tests/test_sequential_cell_learning.py',
        ROOT/'docs/plans/2026-10-02-sequential-recurrent-control.md']
    r={'status':'started','scope':'full-width 48-step constructed learning only; no CER inputs',
       'research_inputs_loaded':False,'research_fits':0,'planned_software_fits':4,'cases':{},
       'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'source_sha256':{str(p.relative_to(ROOT)):R.digest(p) for p in sources},
       'candidate_predeclared':'tanh','model_seed':20260920,'data_seed':20260926}
    save=lambda:(output/'result.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    save();start=time.monotonic()
    try:
        r['prior_pair_sha256']=D.original_hashes(previous);r['runtime']=R.configure_tensorflow(True)
        import tensorflow as tf
        arrays=fixture_data();np.savez_compressed(output/'constructed_data.npz',**arrays)
        r['data_sha256']=R.digest(output/'constructed_data.npz')
        r['constant_prior']=G.measurements(arrays['test_y'],np.full((32,1),.5))
        r['mean_rule_accuracy']={str(reverse):float(np.mean(((arrays['test_x'].mean(axis=(1,2))>0)^reverse)==(1-arrays['test_y'] if reverse else arrays['test_y']).ravel())) for reverse in (False,True)}
        for activation in ('tanh','relu'):
            for reverse in (False,True):
                name=activation+('_reversed' if reverse else '_normal');r['active_case']=name;save();gc.collect()
                with tf.device('/GPU:0'):r['cases'][name]=run_case(output/name,arrays,activation,reverse)
                save()
                if r['cases'][name]['status']!='complete':raise RuntimeError('Stop after incomplete/numerically failed fixed case')
        r['paired_initial_weights']=len({c['initial_weights_sha256'] for c in r['cases'].values()})==1
        r['branch_passes']={a:all(case_passes(r['cases'][a+'_'+label]) for label in ('normal','reversed')) for a in ('tanh','relu')}
        r['tanh_constructed_gate_passed']=r['paired_initial_weights'] and r['branch_passes']['tanh']
        r['status']='complete'
    except Exception as exc:r.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        r['elapsed_seconds']=time.monotonic()-start
        if 'prior_pair_sha256' in r:r['prior_pair_unchanged']=D.original_hashes(previous)==r['prior_pair_sha256']
        r['artifact_sha256']={str(p.relative_to(output)):R.digest(p) for p in output.rglob('*') if p.is_file() and p!=output/'result.json'}
        save()
    return r


def fresh_reload(output):
    R.require_compute();R.configure_tensorflow(True)
    import tensorflow as tf
    import keras
    M.register_sequential_layer()
    r=json.loads((output/'result.json').read_text());assert r['status']=='complete'
    with np.load(output/'constructed_data.npz') as z:xt=z['test_x']
    result={}
    for name,c in r['cases'].items():
        keras.backend.clear_session();gc.collect()
        with tf.device('/GPU:0'):
            m=keras.models.load_model(output/name/'final.keras')
            probe=keras.Model(m.input,[m.outputs[0],*m.get_layer('attention_decoder').output])
            probability,sequence,attention=probe(xt)
        with np.load(output/name/'outputs.npz') as old:
            for current,key in [(probability,'probability'),(sequence,'sequence'),(attention,'attention')]:
                np.testing.assert_array_equal(current.numpy(),old[key])
        assert m.get_layer('attention_decoder').cell_activation==c['cell_activation']
        assert R.weight_hash(m.get_weights())==c['final_weights_sha256']
        assert int(m.optimizer.iterations.numpy())==300
        result[name]={'probability_sequence_attention_exact':True,'weights_exact':True,'updates':300}
    return {'status':'verified','scope':'fresh-process constructed GPU reload; no updates','cases':result}


def clean_config(v):
    if isinstance(v,dict):return {k:clean_config(x) for k,x in v.items() if k!='shared_object_id'}
    if isinstance(v,list):return [clean_config(x) for x in v]
    return v


def saved_weights(case,c):
    import io,zipfile,h5py
    front='layers/sequential_attention_decoder'
    paths=[f'{front}/vars/{i}' for i in range(4)]
    for family,base,cell in [('encoder','lstm','/cell'),('decoder','lstm_cell','')]:
        for i in range(3):
            suffix=f'_{i}' if i else ''
            paths.extend(f'{front}/{family}/{base}{suffix}{cell}/vars/{j}' for j in range(3))
    paths.extend(f'{front}/projection/vars/{i}' for i in range(2))
    for i in range(8):
        suffix=f'_{i}' if i else '';paths.extend(f'layers/gru{suffix}/cell/vars/{j}' for j in range(3))
    paths.extend(f'layers/{layer}/vars/{i}' for layer in ('dense','dense_1') for i in range(2))
    with zipfile.ZipFile(case/'final.keras') as z:
        assert clean_config(json.loads(z.read('config.json')))==clean_config(c['model_json'])
        with h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h:
            weights=[h[p][()] for p in paths]
            assert int(h['optimizer/vars/0'][()])==300
            assert all(np.isfinite(h['optimizer/vars/'+key][()]).all() for key in h['optimizer/vars'])
    assert sum(w.size for w in weights)==9240802 and all(np.isfinite(w).all() for w in weights)
    assert R.weight_hash(weights)==c['final_weights_sha256']
    norms=[float(np.linalg.norm(v.astype(np.float64),axis=0).max()) for p,v in zip(paths,weights)
           if v.ndim==2 or p==front+'/vars/2']
    np.testing.assert_allclose(max(norms),c['maximum_constrained_norm'],rtol=1e-12,atol=1e-30)
    assert max(norms)<=1.00001
    return {'parameter_count':9240802,'weight_hash_verified':True,'optimizer_updates':300,'config_verified':True,'maximum_constrained_norm':max(norms)}


def audit(output,previous):
    import hashlib
    r=json.loads((output/'result.json').read_text())
    assert r['status']=='complete' and not r['research_inputs_loaded'] and r['research_fits']==0
    assert r['prior_pair_unchanged'] and D.original_hashes(previous)==r['prior_pair_sha256']
    assert set(r['cases'])=={'tanh_normal','tanh_reversed','relu_normal','relu_reversed'}
    for p,h in r['source_sha256'].items():
        data=subprocess.check_output(['git','show',r['code_commit']+':'+p],cwd=ROOT)
        assert hashlib.sha256(data).hexdigest()==h,p
    for p,h in r['artifact_sha256'].items():assert R.digest(output/p)==h,p
    arrays=fixture_data()
    with np.load(output/'constructed_data.npz') as z:
        for k in arrays:np.testing.assert_array_equal(z[k],arrays[k])
    result={}
    for name,c in r['cases'].items():
        case=output/name
        assert c==json.loads((case/'result.json').read_text())
        assert c['status']=='complete' and c['updates']==300 and c['timesteps']==48
        assert c['parameter_count']==9240802 and c['bridge_activation']=='sigmoid'
        assert c['cell_activation']==name.split('_')[0]
        with np.load(case/'initial_weights.npz') as z:w=[z[f'w{i}'] for i in range(len(z.files))]
        assert R.weight_hash(w)==c['initial_weights_sha256']==INITIAL_GPU_SHA
        history=json.loads((case/'history.json').read_text())
        assert len(history)==300 and [h['updates'] for h in history]==list(range(1,301))
        assert all(np.isfinite(h['loss']) for h in history)
        reverse=name.endswith('reversed')
        with np.load(case/'predictions.npz') as z:
            for split in ('train','test'):
                labels=1-arrays[split+'_y'] if reverse else arrays[split+'_y']
                np.testing.assert_array_equal(z['observed_'+split+'_y'],labels)
                assert G.measurements(labels,z[split+'_probability'])==c[split]
        for stage in ('initial','final'):
            with np.load(case/(stage+'_layers.npz')) as z:
                for k,stored in c[stage+'_layers'].items():
                    actual=D.describe(z[k])
                    for field,v in stored.items():
                        if isinstance(v,float):np.testing.assert_allclose(actual[field],v,rtol=1e-12,atol=1e-30)
                        else:assert actual[field]==v
        assert case_passes(c)==c['case_gate_passed']
        result[name]={'status':'verified','metrics_replayed':True,'layer_summaries_replayed':True,
            'case_gate_passed':c['case_gate_passed'],'serialized':saved_weights(case,c)}
    reload=json.loads((output/'fresh_reload.json').read_text());assert reload['status']=='verified'
    assert set(reload['cases'])==set(r['cases'])
    expected={a:all(result[a+'_'+label]['case_gate_passed'] for label in ('normal','reversed')) for a in ('tanh','relu')}
    assert expected==r['branch_passes'] and expected['tanh']==r['tanh_constructed_gate_passed']
    return {'status':'verified','scope':'constructed artifact audit; no model inference','cases':result,
        'fresh_process_reload_verified':True,'paired_initial_weights':True,'research_fits':0,'local_model_calls':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--reload',action='store_true');mode.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.reload:print(json.dumps(fresh_reload(a.output),indent=2,sort_keys=True))
    elif a.audit:print(json.dumps(audit(a.output,a.previous),indent=2,sort_keys=True))
    else:run(a.output,a.previous)
