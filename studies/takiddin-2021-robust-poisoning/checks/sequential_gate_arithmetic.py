"""Fixed native gate decomposition, with no optimizer steps or repaired trajectory."""
import argparse
import gc
import hashlib
import inspect
import os
import socket
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import expit
import sequential_saved_state as D

ROOT, STUDY, M, R = D.ROOT, D.STUDY, D.M, D.R
OBS_MANIFEST=STUDY/'results/sequential_saved_state_20261002/artifact-sha256.txt'
LIBRARY_HASHES = {'LSTMCell.call': 'af0376bd97c30e6633d3959f9c3c9f1fcd1f849655ccba3879126dab06d68e23', 'LSTMCell.fused': 'd5beb4addccf5bb77701dce5a3b8c1eada6f83ccd0dd45ce5e683ad0963503f3', 'GRUCell.call': '196d42bc7207b9c5710f9d7db0b24752cb8bd37f86c3fc30c6cf039560bbd0f8'}


def library_hashes():
    import keras
    return {name: hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for name,f in {
        'LSTMCell.call':keras.layers.LSTMCell.call,
        'LSTMCell.fused':keras.layers.LSTMCell._compute_carry_and_output_fused,
        'GRUCell.call':keras.layers.GRUCell.call}.items()}


def step(cell, x, state, kind):
    """Separate matrix terms, preserving native operation order and native next state."""
    import tensorflow as tf
    assert cell.implementation==2 and cell.activation.__name__=='relu'
    assert cell.recurrent_activation.__name__=='sigmoid' and cell.dropout==cell.recurrent_dropout==0
    h0=state[0]
    direct=tf.matmul(x,cell.kernel)
    if kind=='lstm':
        c0=state[1]; recurrent=tf.matmul(h0,cell.recurrent_kernel)
        z=tf.split((direct+recurrent)+cell.bias,4,axis=-1)
        i,f,o=[tf.sigmoid(z[k]) for k in (0,1,3)]
        candidate=tf.nn.relu(z[2]); retained=f*c0; injected=i*candidate
        c=retained+injected; h=o*tf.nn.relu(c)
        u=cell.units
        values={'h_previous':h0,'c_previous':c0,'input_gate':i,'forget_gate':f,'output_gate':o,
            'input_gate_pre':z[0],'forget_gate_pre':z[1],'output_gate_pre':z[3],
            'input_candidate':direct[:,2*u:3*u], 'recurrent_candidate':recurrent[:,2*u:3*u],
            'bias_candidate':tf.broadcast_to(cell.bias[2*u:3*u],tf.shape(c)),
            'candidate_pre':z[2],'candidate':candidate,'retained':retained,'injected':injected,'manual_h':h,'manual_c':c}
    else:
        assert not cell.reset_after
        u=cell.units; xb=tf.split(direct+cell.bias,3,axis=-1)
        recurrent=tf.matmul(h0,cell.recurrent_kernel[:,:2*u])
        z=tf.sigmoid(xb[0]+recurrent[:,:u]); reset=tf.sigmoid(xb[1]+recurrent[:,u:])
        rc=tf.matmul(reset*h0,cell.recurrent_kernel[:,2*u:])
        pre=xb[2]+rc; candidate=tf.nn.relu(pre)
        retained=z*h0; injected=(1-z)*candidate; h=retained+injected
        values={'h_previous':h0,'update_gate':z,'reset_gate':reset,
            'update_gate_pre':xb[0]+recurrent[:,:u],'reset_gate_pre':xb[1]+recurrent[:,u:],
            'input_candidate':direct[:,2*u:],'recurrent_candidate':rc,
            'bias_candidate':tf.broadcast_to(cell.bias[2*u:],tf.shape(h)),
            'candidate_pre':pre,'candidate':candidate,'retained':retained,'injected':injected,'manual_h':h}
    native,native_state=cell(x,state,training=False)
    np.testing.assert_allclose(h.numpy(),native.numpy(),rtol=1e-5,atol=1e-30)
    values['h']=native
    if kind=='lstm':
        np.testing.assert_allclose(c.numpy(),native_state[1].numpy(),rtol=1e-5,atol=1e-30)
        values['c']=native_state[1]
    return native_state,{k:v.numpy() for k,v in values.items()}


def sequence(cell, x, kind, initial=None):
    import tensorflow as tf
    state=initial or [tf.zeros((len(x),cell.units),dtype=tf.float32) for _ in range(2 if kind=='lstm' else 1)]
    rows=[]
    for t in range(x.shape[1]):
        state,row=step(cell,x[:,t,:],state,kind);rows.append(row)
    return state,{k:np.stack([r[k] for r in rows],axis=1) for k in rows[0]}


def compact(v):
    """One time step; all profiles/units, with no outcome-based threshold selection."""
    v=np.asarray(v)
    if not np.isfinite(v).all():raise ValueError('Nonfinite gate arithmetic')
    return {'min':float(v.min()),'max':float(v.max()),'zero_count':int(np.count_nonzero(v==0)),
        'positive_count':int(np.count_nonzero(v>0)),'size':int(v.size),
        'one_count':int(np.count_nonzero(v==1)), 'le_1e_minus6':int(np.count_nonzero(v<=1e-6)),
        'ge_1_minus1e_minus6':int(np.count_nonzero(v>=1-1e-6)),
        'l2':float(np.linalg.norm(v.astype(np.float64).ravel()))}


def pack(trace, kind, path):
    h=trace['h'];flat=int(np.argmax(h[:,-1,:]))
    extreme_row,extreme_unit=np.unravel_index(flat,h[:,-1,:].shape)
    witnesses=sorted(set([0,1,int(extreme_row)]))
    trace['candidate_without_bias']=np.maximum(trace['input_candidate']+trace['recurrent_candidate'],0)
    trace['candidate_without_recurrent']=np.maximum(trace['input_candidate']+trace['bias_candidate'],0)
    summary={'kind':kind,'witness_rows':witnesses,'maximum_final_hidden_location':[int(extreme_row),int(extreme_unit)],
        'batch_size':len(h),'steps':h.shape[1],
        'blocked_by_bias_per_step':[int(np.count_nonzero((trace['candidate'][:,t,:]==0)&(trace['candidate_without_bias'][:,t,:]>0))) for t in range(h.shape[1])],
        'recurrently_supported_injection_per_step':[int(np.count_nonzero((trace['injected'][:,t,:]>0)&(trace['candidate_without_recurrent'][:,t,:]==0))) for t in range(h.shape[1])],
        'supported_by_recurrent_per_step':[int(np.count_nonzero((trace['candidate'][:,t,:]>0)&(trace['candidate_without_recurrent'][:,t,:]==0))) for t in range(h.shape[1])],
        'per_step':{k:[compact(v[:,t,:]) for t in range(v.shape[1])] for k,v in trace.items()},
        'maximum_native_manual_hidden_error':float(np.max(np.abs(h-trace['manual_h'])))}
    saved={k:v[witnesses] for k,v in trace.items()}
    saved['all_h']=h
    if kind=='lstm':
        saved['all_c']=trace['c']
        summary['maximum_native_manual_cell_error']=float(np.max(np.abs(trace['c']-trace['manual_c'])))
    else:
        saved['all_update_gate']=trace['update_gate'];saved['all_injected']=trace['injected']
        saved['all_candidate']=trace['candidate'];saved['all_retained']=trace['retained']
        total=trace['h_previous'][:,0,:].astype(np.float64)
        for t in range(h.shape[1]):total=trace['update_gate'][:,t,:].astype(np.float64)*total+trace['injected'][:,t,:].astype(np.float64)
        np.testing.assert_allclose(total,h[:,-1,:],rtol=1e-4,atol=1e-30)
        summary['weighted_injection_identity_max_error']=float(np.max(np.abs(total-h[:,-1,:])))
        active=np.any(trace['candidate']>0,axis=(0,2))
        summary['last_step_with_any_positive_candidate']=int(np.flatnonzero(active)[-1]+1) if active.any() else None
        summary['candidate_zero_tail_from_step']=next((i+1 for i in range(len(active)) if not active[i:].any()),None)
    np.savez_compressed(path,**saved)
    path.with_suffix('.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary


def inspect_model(model,x,old,output):
    import tensorflow as tf
    before=D.state_hash(model);front=model.get_layer('attention_decoder')
    reports={};value=x;states=[]
    for i,layer in enumerate(front.encoder):
        state,trace=sequence(layer.cell,value,'lstm')
        native,h,c=layer(value,training=False)
        np.testing.assert_array_equal(trace['h'],native.numpy())
        np.testing.assert_array_equal(state[1].numpy(),c.numpy())
        reports[f'encoder_{i+1}']=pack(trace,'lstm',output/f'encoder_{i+1}.npz')
        value=native;states.append([h,c]);del trace
    np.testing.assert_array_equal(value.numpy(),old['encoder_memory'])
    memory=value;projected=tf.einsum('btd,dk->btk',memory,front.attention_encoder_kernel)
    states=list(reversed(states));previous=tf.zeros((len(x),1),dtype=tf.float32)
    rows=[[],[],[]];bridges=[];hidden=[]
    for t in range(front.steps):
        context,_=front.attend(memory,projected,states[-1][0]);value=tf.concat([context,previous],axis=-1)
        for i,cell in enumerate(front.decoder):
            states[i],row=step(cell,value,states[i],'lstm');rows[i].append(row);value=states[i][0]
        hidden.append(value.numpy());previous=front.projection(value);bridges.append(previous.numpy())
    np.testing.assert_array_equal(np.stack(hidden,axis=1),old['decoder_hidden'])
    np.testing.assert_array_equal(np.stack(bridges,axis=1),old['intermediate'])
    for i,records in enumerate(rows):
        trace={k:np.stack([r[k] for r in records],axis=1) for k in records[0]}
        reports[f'decoder_{i+1}']=pack(trace,'lstm',output/f'decoder_{i+1}.npz')
    del rows,trace,hidden,bridges
    cell=model.get_layer('sequence_gru_8').cell
    _,trace=sequence(cell,tf.convert_to_tensor(old['gru_7']),'gru')
    np.testing.assert_array_equal(trace['h'][:,-1,:],old['gru_8'])
    reports['gru_8']=pack(trace,'gru',output/'gru_8.npz')
    if D.state_hash(model)!=before:raise ValueError('Model/optimizer changed')
    return {'cells':reports,'state_before_sha256':before,'state_after_sha256':D.state_hash(model),
        'optimizer_updates':int(model.optimizer.iterations.numpy()),'native_saved_parity_exact':True}


def observation_hashes(directory):
    found={}
    for line in OBS_MANIFEST.read_text().splitlines():
        expected,name=line.split('  ',1);name=name.removeprefix('./')
        actual=R.digest(directory/name)
        if actual!=expected:raise ValueError('Previous observation changed: '+name)
        found[name]=actual
    if len(found)!=10:raise ValueError('Unexpected observation inventory')
    return found


def run(previous,observations,output):
    R.require_compute();output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    sources=[Path(__file__).resolve(),Path(D.__file__).resolve(),OBS_MANIFEST,D.MANIFEST,
        ROOT/'tests/test_sequential_gate_arithmetic.py',STUDY/'checks/run_sequential_gate_arithmetic.sbatch',
        ROOT/'docs/plans/2026-10-02-sequential-gate-arithmetic.md',*sorted((STUDY/'reproduction').glob('*.py'))]
    result={'status':'started','scope':'fixed native gate arithmetic; no training or repaired trajectory',
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):R.digest(p) for p in sources},'model_fits':0,'optimizer_updates':0,
        'selection':'same first100 test rows as408550; labels unused','states':{},
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'host':socket.gethostname()}
    save=lambda:(output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if R.digest(STUDY/'reproduction/models.py')!=D.MODEL_SHA:raise ValueError('Model source changed')
        result['pair_sha256']=D.original_hashes(previous);result['observation_sha256']=observation_hashes(observations)
        result['runtime']=R.configure_tensorflow(True)
        import tensorflow as tf
        import keras
        result['library_hashes']=library_hashes()
        if result['library_hashes']!=LIBRARY_HASHES:raise ValueError('Pinned cell source differs')
        M.register_sequential_layer()
        with np.load(observations/'selected_inputs.npz',allow_pickle=False) as z:x=z['test_x'][:,:,None]
        for name in ('initial','final_p00','final_p30'):
            if time.monotonic()-start>=420:raise TimeoutError('Seven-minute observation guard')
            keras.backend.clear_session();gc.collect();level='p00' if name=='initial' else name[-3:]
            record=json.loads((previous/level/'result.json').read_text())
            with tf.device('/GPU:0'):
                if name=='initial':
                    model=M.sequential_ensemble(20260920,bridge_activation='sigmoid')
                    with np.load(previous/'p00/initial_weights.npz',allow_pickle=False) as z:
                        model.set_weights([z[f'weight_{i}'] for i in range(len(z.files))])
                else:model=keras.models.load_model(previous/level/'model.keras')
                expected=record['neural'][('initial' if name=='initial' else 'final')+'_weights_sha256']
                if R.weight_hash(model.get_weights())!=expected or model.count_params()!=9240802:raise ValueError('Restored model differs')
                if any('GPU:0' not in v.value.device for v in model.trainable_weights):raise ValueError('Wrong model device')
                folder=output/name;folder.mkdir()
                with np.load(observations/(name+'_test.npz'),allow_pickle=False) as old:
                    np.testing.assert_array_equal(x,old['input'])
                    tick=time.monotonic();result['states'][name]=inspect_model(model,tf.convert_to_tensor(x),old,folder)
                    result['states'][name]['seconds']=time.monotonic()-tick
            print(json.dumps({'state':name,'seconds':result['states'][name]['seconds'],'native_parity_exact':True}),flush=True)
            save();del model
        result['status']='complete'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        result['elapsed_seconds']=time.monotonic()-start
        try:
            if 'pair_sha256' in result:result['pair_unchanged']=D.original_hashes(previous)==result['pair_sha256']
            if 'observation_sha256' in result:result['observations_unchanged']=observation_hashes(observations)==result['observation_sha256']
        except Exception as exc:result.update(status='failed',preservation_error=str(exc))
        result['artifact_sha256']={str(p.relative_to(output)):R.digest(p) for p in output.rglob('*') if p.is_file() and p.name!='result.json'}
        save()
    return result


def audit(previous,observations,output):
    """Replay saved state summaries and raw witness equations; no model calls."""
    r=json.loads((output/'result.json').read_text())
    assert r['status']=='complete' and r['model_fits']==r['optimizer_updates']==0
    assert r['pair_unchanged'] and r['observations_unchanged']
    assert D.original_hashes(previous)==r['pair_sha256']
    assert observation_hashes(observations)==r['observation_sha256']
    for p,h in r['source_sha256'].items():
        source=subprocess.check_output(['git','show',r['code_commit']+':'+p],cwd=ROOT)
        assert hashlib.sha256(source).hexdigest()==h,p
    for p,h in r['artifact_sha256'].items():assert R.digest(output/p)==h,p
    close=lambda x,y:np.testing.assert_allclose(x,y,rtol=1e-5,atol=1e-30)
    count=0
    for name,state in r['states'].items():
        assert state['state_before_sha256']==state['state_after_sha256']
        assert state['optimizer_updates']==(0 if name=='initial' else 2250)
        assert state['native_saved_parity_exact']
        for cell,record in state['cells'].items():
            assert json.loads((output/name/(cell+'.json')).read_text())==record
            with np.load(output/name/(cell+'.npz'),allow_pickle=False) as a:
                assert all(np.isfinite(a[k]).all() for k in a.files)
                ids=record['witness_rows'];h=a['all_h']
                row,col=np.unravel_index(int(np.argmax(h[:,-1,:])),h[:,-1,:].shape)
                assert ids==sorted(set([0,1,int(row)]))
                assert record['maximum_final_hidden_location']==[int(row),int(col)]
                np.testing.assert_array_equal(a['h'],h[ids])
                np.testing.assert_array_equal(a['h_previous'][:,1:],a['h'][:,:-1])
                full={'h':'all_h','c':'all_c'} if record['kind']=='lstm' else {
                    'h':'all_h','update_gate':'all_update_gate','candidate':'all_candidate',
                    'retained':'all_retained','injected':'all_injected'}
                for short,long in full.items():
                    for t,stored in enumerate(record['per_step'][short]):
                        actual=compact(a[long][:,t,:])
                        for k,v in stored.items():
                            if isinstance(v,float):np.testing.assert_allclose(actual[k],v,rtol=1e-12,atol=1e-30)
                            else:assert actual[k]==v
                if record['kind']=='lstm':
                    pre=(a['input_candidate']+a['recurrent_candidate'])+a['bias_candidate']
                    np.testing.assert_array_equal(a['c'],a['all_c'][ids])
                    np.testing.assert_array_equal(a['c_previous'][:,1:],a['c'][:,:-1])
                    for gate in ('input_gate','forget_gate','output_gate'):
                        np.testing.assert_allclose(a[gate],expit(a[gate+'_pre'].astype(np.float64)),rtol=1e-6,atol=1e-7)
                    close(a['retained'],a['forget_gate']*a['c_previous'])
                    close(a['injected'],a['input_gate']*a['candidate'])
                    close(a['c'],a['retained']+a['injected'])
                    close(a['h'],a['output_gate']*np.maximum(a['c'],0))
                else:
                    pre=(a['input_candidate']+a['bias_candidate'])+a['recurrent_candidate']
                    for gate in ('update_gate','reset_gate'):
                        np.testing.assert_allclose(a[gate],expit(a[gate+'_pre'].astype(np.float64)),rtol=1e-6,atol=1e-7)
                    prev=np.concatenate([np.zeros_like(h[:,:1]),h[:,:-1]],axis=1)
                    close(a['all_retained'],a['all_update_gate']*prev)
                    close(a['all_injected'],(1-a['all_update_gate'])*a['all_candidate'])
                    close(h,a['all_retained']+a['all_injected'])
                    total=np.zeros_like(h[:,0,:],dtype=np.float64)
                    for t in range(h.shape[1]):total=a['all_update_gate'][:,t,:].astype(np.float64)*total+a['all_injected'][:,t,:]
                    np.testing.assert_allclose(total,h[:,-1,:],rtol=1e-4,atol=1e-30)
                close(a['candidate_pre'],pre);close(a['candidate'],np.maximum(pre,0))
                close(a['candidate_without_bias'],np.maximum(a['input_candidate']+a['recurrent_candidate'],0))
                close(a['candidate_without_recurrent'],np.maximum(a['input_candidate']+a['bias_candidate'],0))
                with np.load(observations/(name+'_test.npz'),allow_pickle=False) as old:
                    if cell=='encoder_3':np.testing.assert_array_equal(h,old['encoder_memory'])
                    if cell=='decoder_3':np.testing.assert_array_equal(h,old['decoder_hidden'])
                    if cell=='gru_8':np.testing.assert_array_equal(h[:,-1,:],old['gru_8'])
            count+=1
    assert count==21
    return {'status':'verified','states':3,'cell_traces':21,'all_batch_native_state_summaries_replayed':True,
        'fixed_and_extreme_witness_gate_equations_verified':True,'gru_full_batch_retention_injection_verified':True,
        'native_saved_sequence_checks':9,'original_pair_files_unchanged':15,'previous_observation_files_unchanged':10,
        'source_output_hashes_verified':True,'model_calls':0,'model_fits':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',action='store_true');a=p.parse_args()
    if a.audit:print(json.dumps(audit(a.previous,a.observations,a.output),indent=2,sort_keys=True))
    else:
        r=run(a.previous,a.observations,a.output);raise SystemExit(0 if r['status']=='complete' else 2)
