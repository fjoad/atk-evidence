"""Close the stopped comparison's artifact checks; never fit or resume a model."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--code-root',type=Path,required=True);p.add_argument('--attempt',type=Path,required=True)
p.add_argument('--failed-attempt',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-reload',action='store_true')
a=p.parse_args()
sys.path.insert(0,str(a.code_root/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_cell_learning as C
r=json.loads((a.attempt/'result.json').read_text())
assert r['status']=='failed' and r['active_case']=='relu_normal'
assert set(r['cases'])=={'tanh_normal','tanh_reversed','relu_normal'}
assert r['cases']['relu_normal']['status']=='nonfinite_loss' and r['cases']['relu_normal']['updates']==245
assert not (a.attempt/'relu_reversed').exists()
assert r['prior_pair_unchanged'] and r['failed_attempt_unchanged']
assert C.D.original_hashes(a.previous)==r['prior_pair_sha256']
for f,h in r['failed_pre_fit_attempt_sha256'].items():assert C.R.digest(a.failed_attempt/f)==h
for f,h in r['artifact_sha256'].items():assert C.R.digest(a.attempt/f)==h

if a.gpu_reload:
    C.R.require_compute();runtime=C.R.configure_tensorflow(True)
    import tensorflow as tf
    import keras
    C.M.register_sequential_layer()
    assert C.R.digest(C.STUDY/'reproduction/models.py')==r['source_sha256']['studies/takiddin-2021-robust-poisoning/reproduction/models.py']
    with np.load(a.attempt/'constructed_data.npz') as z:x=z['test_x']
    results={}
    for name in ('tanh_normal','tanh_reversed'):
        c=r['cases'][name];assert C.case_passes(c)
        keras.backend.clear_session()
        with tf.device('/GPU:0'):
            m=keras.models.load_model(a.attempt/name/'final.keras')
            before=C.D.state_hash(m)
            probe=keras.Model(m.input,[m.outputs[0],*m.get_layer('attention_decoder').output])
            values=probe(x,training=False)
        with np.load(a.attempt/name/'outputs.npz') as old:
            for value,key in zip(values,('probability','sequence','attention')):np.testing.assert_array_equal(value.numpy(),old[key])
        assert m.get_layer('attention_decoder').cell_activation=='tanh'
        assert C.R.weight_hash(m.get_weights())==c['final_weights_sha256']
        assert int(m.optimizer.iterations.numpy())==300 and C.D.state_hash(m)==before
        results[name]={'probability_sequence_attention_exact':True,'weights_optimizer_unchanged':True,'updates':300}
    out={'status':'verified','scope':'fresh-process verification of completed constructed models only',
        'fits':0,'updates':0,'runtime':runtime,'cases':results,
        'training_result_sha256':C.R.digest(a.attempt/'result.json'),
        'verification_script_sha256':C.R.digest(__file__)}
else:
    import subprocess,h5py
    for path,h in r['source_sha256'].items():
        source=subprocess.check_output(['git','show',r['code_commit']+':'+path],cwd=a.code_root)
        assert hashlib.sha256(source).hexdigest()==h
    arrays=C.fixture_data()
    with np.load(a.attempt/'constructed_data.npz') as z:
        for k,v in arrays.items():np.testing.assert_array_equal(v,z[k])
    results={}
    for name,c in r['cases'].items():
        d=a.attempt/name;assert c==json.loads((d/'result.json').read_text())
        assert c['parameter_count']==9240802 and c['timesteps']==48
        assert c['optimizer_updates_before_fit']==0 and 'CPU:' in c['input_pipeline_device']
        assert all('GPU:0' in dev for dev in c['weight_devices'])
        with np.load(d/'initial_weights.npz') as z:initial=[z[f'w{i}'] for i in range(len(z.files))]
        assert C.R.weight_hash(initial)==c['initial_weights_sha256']==C.INITIAL_GPU_SHA
        history=json.loads((d/'history.json').read_text())
        assert len(history)==c['updates'] and [v['updates'] for v in history]==list(range(1,c['updates']+1))
        if name.startswith('tanh'):
            assert c['updates']==300 and all(np.isfinite(v['loss']) for v in history)
            weights=C.saved_weights(d,c)
        else:
            assert history[-1]['loss'] is None and all(np.isfinite(v['loss']) for v in history[:-1])
            with zipfile.ZipFile(d/'final.keras') as z:
                assert C.clean_config(json.loads(z.read('config.json')))==C.clean_config(c['model_json'])
                with h5py.File(io.BytesIO(z.read('model.weights.h5')),'r') as h:
                    vals={}
                    h['layers'].visititems(lambda key,v:vals.update({key:v[()]}) if isinstance(v,h5py.Dataset) else None)
                    assert int(h['optimizer/vars/0'][()])==245
            n=sum(int(np.count_nonzero(~np.isfinite(v))) for v in vals.values())
            assert (n==0)==c['weights_finite']
            weights={'preserved_nonfinite_failure':True,'nonfinite_parameter_values':n,'optimizer_updates':245,'configuration_verified':True}
        with np.load(d/'predictions.npz') as z:
            for split in ('train','test'):
                labels=1-arrays[split+'_y'] if name.endswith('reversed') else arrays[split+'_y']
                np.testing.assert_array_equal(z['observed_'+split+'_y'],labels)
                assert C.G.measurements(labels,z[split+'_probability'])==c[split]
        for stage in ('initial','final'):
            with np.load(d/(stage+'_layers.npz')) as z:
                for key,stored in c[stage+'_layers'].items():
                    actual=C.D.describe(z[key])
                    for field,v in stored.items():
                        if isinstance(v,float):np.testing.assert_allclose(actual[field],v,rtol=1e-12,atol=1e-30)
                        else:assert actual[field]==v
        assert C.case_passes(c)==c['case_gate_passed']
        results[name]={'status':'verified','gate_passed':c['case_gate_passed'],'updates':c['updates'],'serialized':weights}
    assert r['cases']['tanh_normal']['initial_arrays_match_failed_pre_fit_attempt']
    out={'status':'verified','scope':'partial comparison artifact audit; no model calls',
        'comparison_completed':False,'stop_rule_respected':True,'unrun_cases':['relu_reversed'],
        'cases':results,'tanh_two_case_learning_gate_passed':True,'fresh_gpu_reload_required_separately':True,
        'original_pair_and_failed_attempt_unchanged':True,'fits':0,'model_calls':0,
        'training_result_sha256':C.R.digest(a.attempt/'result.json')}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':out['status'],'output':str(a.output),'fits':0}))
