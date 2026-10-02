"""Fixed saved-state observation; no fitting or optimizer updates."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.special import expit

import sequential_constructed as G

STUDY, ROOT, M, R = G.STUDY, G.ROOT, G.M, G.R
STAGES = ['intermediate', *[f'gru_{i}' for i in range(1, 9)], 'classifier_hidden', 'probability']
MANIFEST = STUDY / 'results/sequential_pilot_20261001/pair-artifact-sha256.txt'
MODEL_SHA = '9c9201bb682ab0732dd25ffa469a8003173eeb166c503b5e7f63a81b525c1c13'


def describe(value):
    value = np.asarray(value)
    finite = np.isfinite(value)
    result = {'shape': list(value.shape), 'dtype': str(value.dtype), 'size': int(value.size),
              'finite_count': int(finite.sum()), 'zero_count': int(np.count_nonzero(value == 0))}
    if not finite.all() or not value.size:
        result.update(min=None, max=None, l2=None, max_axis0_range=None)
    else:
        wide = value.astype(np.float64)
        result.update(min=float(wide.min()), max=float(wide.max()),
                      l2=float(np.linalg.norm(wide.ravel())),
                      max_axis0_range=float(np.ptp(wide, axis=0).max()) if value.ndim >= 2 else None)
    return result


def state_hash(model):
    return R.weight_hash(model.get_weights() + [v.numpy() for v in model.optimizer.variables])


def readout_arithmetic(hidden, kernel, bias):
    """Only the final affine arithmetic is widened, not the network."""
    hidden, kernel, bias = map(np.asarray, (hidden, kernel, bias))
    z32 = hidden @ kernel + bias
    z64 = hidden.astype(np.float64) @ kernel.astype(np.float64) + bias.astype(np.float64)
    return {'readout_logits_float32': z32, 'readout_logits_float64': z64,
            'readout_probability_float64': expit(z64)}


def frontend_trace(front, x):
    """Observe native cells, with the same causal attention and feedback order."""
    import tensorflow as tf
    memory, projected, states = front.encode(x, training=False)
    previous = tf.zeros((tf.shape(x)[0], 1), dtype=x.dtype)
    hidden, logits, outputs, attention = [], [], [], []
    for _ in range(front.steps):
        context, weights = front.attend(memory, projected, states[-1][0])
        value, next_states = tf.concat((context, previous), axis=-1), []
        for cell, state in zip(front.decoder, states):
            value, state = cell(value, state, training=False)
            next_states.append(list(state))
        raw = tf.matmul(value, front.projection.kernel) + front.projection.bias
        previous = front.projection(value)
        states = next_states
        hidden.append(value); logits.append(raw); outputs.append(previous); attention.append(weights)
    return {'encoder_memory': memory.numpy(), 'decoder_hidden': tf.stack(hidden, axis=1).numpy(),
            'bridge_logits': tf.stack(logits, axis=1).numpy(),
            'traced_intermediate': tf.stack(outputs, axis=1).numpy(),
            'traced_attention': tf.stack(attention, axis=1).numpy()}


def observe(model, x, labels):
    """One unchanged state/batch; labels maps named observed-label conditions."""
    import tensorflow as tf
    import keras
    before, updates = state_hash(model), int(model.optimizer.iterations.numpy())
    front = model.get_layer('attention_decoder')
    tensors = [front.output[0], *[model.get_layer(f'sequence_gru_{i}').output for i in range(1, 9)],
               model.get_layer('classifier_hidden').output, model.outputs[0], front.output[1]]
    probe = keras.Model(model.input, tensors)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        values = probe(x, training=False)
        losses = {name: model.loss(tf.convert_to_tensor(y, dtype=tf.float32), values[-2])
                  for name, y in labels.items()}
    arrays = {'input': x.numpy(), **{name: value.numpy() for name, value in zip(STAGES, values[:-1])},
              'attention': values[-1].numpy()}
    target_names = [v.path for v in model.trainable_weights] + ['input'] + STAGES
    targets = list(model.trainable_weights) + [x] + list(values[:-1])
    gradients = {}
    for label_name, loss in losses.items():
        grad = tape.gradient(loss, targets)
        gradients[label_name] = {'loss': float(loss.numpy()), 'arrays': {}, 'missing_targets': [],
                                'label_counts': {str(v): int(np.count_nonzero(labels[label_name] == v)) for v in (0, 1)}}
        arrays['labels_' + label_name] = np.asarray(labels[label_name])
        for i, (target, value) in enumerate(zip(target_names, grad)):
            if value is None:
                gradients[label_name]['missing_targets'].append(target)
            else:
                key = f'gradient_{label_name}_{i:03d}'
                arrays[key] = value.numpy()
                gradients[label_name]['arrays'][key] = target
    del tape
    trace = frontend_trace(front, x)
    bridge_error = float(np.max(np.abs(trace['traced_intermediate'] - arrays['intermediate'])))
    attention_error = float(np.max(np.abs(trace['traced_attention'] - arrays['attention'])))
    np.testing.assert_allclose(trace['traced_intermediate'], arrays['intermediate'], atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(trace['traced_attention'], arrays['attention'], atol=1e-7, rtol=1e-6)
    arrays.update(trace)
    head, output = model.get_layer('classifier_hidden'), model.get_layer('attack_probability')
    arrays['hidden_preactivation_float64'] = (arrays['gru_8'].astype(np.float64)
        @ head.kernel.numpy().astype(np.float64) + head.bias.numpy().astype(np.float64))
    np.testing.assert_allclose(np.maximum(arrays['hidden_preactivation_float64'], 0),
                               arrays['classifier_hidden'], atol=1e-7, rtol=1e-5)
    arrays['readout_logits_gpu_float32'] = (tf.matmul(values[-3], output.kernel) + output.bias).numpy()
    np.testing.assert_array_equal(output.activation(tf.convert_to_tensor(arrays['readout_logits_gpu_float32'])).numpy(), arrays['probability'])
    arrays.update(readout_arithmetic(arrays['classifier_hidden'], output.kernel.numpy(), output.bias.numpy()))
    np.testing.assert_allclose(expit(arrays['readout_logits_float32']), arrays['probability'], atol=1e-7, rtol=1e-6)
    arrays['bridge_sigmoid_derivative'] = arrays['intermediate'] * (1-arrays['intermediate'])
    after = state_hash(model)
    if before != after or int(model.optimizer.iterations.numpy()) != updates:
        raise ValueError('Observer changed model or optimizer state')
    return arrays, {'state_before_sha256': before, 'state_after_sha256': after, 'optimizer_updates': updates,
                    'observer_bridge_max_error': bridge_error, 'observer_attention_max_error': attention_error,
                    'gradients': gradients, 'arrays': {k: describe(v) for k, v in arrays.items()}}


def original_hashes(previous):
    result = {}
    for line in MANIFEST.read_text().splitlines():
        expected, name = line.split('  ', 1)
        name = name.removeprefix('./')
        actual = R.digest(previous / name)
        if actual != expected:
            raise ValueError('Original artifact mismatch: ' + name)
        result[name] = actual
    if len(result) != 15:
        raise ValueError('Unexpected original inventory')
    return result


def run(previous, preparation, output):
    R.require_compute()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sources = [Path(__file__).resolve(), STUDY/'checks/sequential_constructed.py',
        STUDY/'checks/run_sequential_saved_state.sbatch', ROOT/'tests/test_sequential_saved_state.py',
        ROOT/'docs/plans/2026-10-02-sequential-saved-state-diagnostic.md', MANIFEST,
        *sorted((STUDY/'reproduction').glob('*.py'))]
    result = {'status': 'started', 'scope': 'fixed saved-state diagnostic; no model fitting',
        'code_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)):R.digest(p) for p in sources},
        'research_fits': 0, 'research_optimizer_updates': 0, 'observations': {},
        'selection': {'training_indices': list(range(100)), 'test_indices': list(range(100))}}
    save = lambda: (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if R.digest(STUDY/'reproduction/models.py') != MODEL_SHA:
            raise ValueError('Frozen model implementation changed')
        result['original_sha256'] = original_hashes(previous)
        result['runtime'] = R.configure_tensorflow(True)
        import tensorflow as tf
        import keras
        prepared, records = {}, {}
        result['input_sha256'] = {}
        for level in ('p00','p30'):
            prepared[level], _, hashes = R.load_preparation(preparation/f'generalized-two-class-{level}')
            records[level] = json.loads((previous/level/'result.json').read_text())
            result['input_sha256'][level] = hashes
            if hashes != records[level]['input_sha256']:
                raise ValueError('Inputs differ from original run')
        for name in ('train_x','test_x'):
            np.testing.assert_array_equal(prepared['p00'][name], prepared['p30'][name])
        if R.digest(previous/'p00/initial_weights.npz') != R.digest(previous/'p30/initial_weights.npz'):
            raise ValueError('Initial weight archives differ')
        np.savez_compressed(output/'selected_inputs.npz',
            train_x=prepared['p00']['train_x'][:100], test_x=prepared['p00']['test_x'][:100],
            train_y_p00=prepared['p00']['train_observed_y'][:100],
            train_y_p30=prepared['p30']['train_observed_y'][:100],
            test_y=prepared['p00']['test_true_y'][:100])
        M.register_sequential_layer()
        for state in ('initial','final_p00','final_p30'):
            if time.monotonic()-started >= 420:
                raise TimeoutError('Seven-minute diagnostic guard')
            keras.backend.clear_session(); gc.collect()
            level = 'p00' if state == 'initial' else state[-3:]
            with tf.device('/GPU:0'):
                if state == 'initial':
                    model = M.sequential_ensemble(20260920, bridge_activation='sigmoid')
                    with np.load(previous/'p00/initial_weights.npz', allow_pickle=False) as z:
                        model.set_weights([z[f'weight_{i}'] for i in range(len(z.files))])
                else:
                    model = keras.models.load_model(previous/level/'model.keras')
            expected_hash = records[level]['neural'][('initial' if state == 'initial' else 'final')+'_weights_sha256']
            if R.weight_hash(model.get_weights()) != expected_hash or model.count_params() != 9240802:
                raise ValueError('Unexpected restored model')
            if any('GPU:0' not in v.value.device for v in model.trainable_weights):
                raise ValueError('Model device mismatch')
            for split in ('train','test'):
                if time.monotonic()-started >= 420:
                    raise TimeoutError('Seven-minute diagnostic guard')
                x = prepared['p00'][split+'_x'][:100,:,None]
                conditions = ('p00','p30') if state == 'initial' else (level,)
                labels = ({k:prepared[k]['train_observed_y'][:100,None].astype(np.float32) for k in conditions}
                          if split == 'train' else {})
                tick = time.monotonic()
                with tf.device('/GPU:0'):
                    arrays, report = observe(model, x, labels)
                report['seconds'] = time.monotonic()-tick
                key = state+'_'+split
                np.savez_compressed(output/(key+'.npz'), **arrays)
                result['observations'][key] = report
                if any(v['finite_count'] != v['size'] for v in report['arrays'].values()):
                    raise ValueError('Nonfinite diagnostic observations preserved')
                if split == 'test':
                    with np.load(previous/level/'representations.npz',allow_pickle=False) as z:
                        expected_rep = z['initial' if state=='initial' else 'final'][:100]
                        if state=='initial': expected_p = z['initial_probabilities'][:100]
                    if state != 'initial':
                        with np.load(previous/level/'predictions.npz',allow_pickle=False) as z:
                            expected_p = z['raw_probabilities'][:100]
                    np.testing.assert_array_equal(arrays['intermediate'][:,:,0], expected_rep)
                    np.testing.assert_array_equal(arrays['probability'][:,0], expected_p)
                    report['saved_test_probability_and_representation_exact'] = True
                print(json.dumps({'observation':key,'seconds':report['seconds'],
                    'zero_stages':[n for n in STAGES if report['arrays'][n]['zero_count']==report['arrays'][n]['size']],
                    'probability_profile_range':report['arrays']['probability']['max_axis0_range']}),flush=True)
                save()
                del arrays
            del model
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        try:
            if 'original_sha256' in result:
                result['originals_unchanged'] = original_hashes(previous) == result['original_sha256']
            if 'input_sha256' in result:
                result['prepared_inputs_unchanged'] = all(R.load_preparation(
                    preparation/f'generalized-two-class-{level}')[2] == hashes
                    for level, hashes in result['input_sha256'].items())
                if not result['prepared_inputs_unchanged']:
                    raise ValueError('Prepared inputs changed')
        except Exception as exc:
            result.update(status='failed',preservation_error=f'{type(exc).__name__}: {exc}')
        result['artifact_sha256'] = {p.name:R.digest(p) for p in output.iterdir() if p.name!='result.json'}
        save()
    return result


def audit(output, previous, preparation):
    """Replay saved arrays and elementary derivatives without model inference."""
    import hashlib
    r=json.loads((output/'result.json').read_text())
    assert r['status']=='complete' and r['originals_unchanged'] and r['prepared_inputs_unchanged']
    assert r['research_fits']==r['research_optimizer_updates']==0
    assert original_hashes(previous)==r['original_sha256']
    assert r['selection']=={'training_indices':list(range(100)),'test_indices':list(range(100))}
    for p,h in r['source_sha256'].items():
        frozen=subprocess.check_output(['git','show',r['code_commit']+':'+p],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==h,p
    for f,h in r['artifact_sha256'].items():assert R.digest(output/f)==h,f
    prepared={}
    for level in ('p00','p30'):
        prepared[level],_,hashes=R.load_preparation(preparation/f'generalized-two-class-{level}')
        assert hashes==r['input_sha256'][level]
    expected={state+'_'+split for state in ('initial','final_p00','final_p30') for split in ('train','test')}
    assert set(r['observations'])==expected
    gradients_checked=0
    for key,obs in r['observations'].items():
        assert obs['state_before_sha256']==obs['state_after_sha256']
        assert obs['optimizer_updates']==(0 if key.startswith('initial_') else 2250)
        split=key.rsplit('_',1)[1]
        with np.load(output/(key+'.npz'),allow_pickle=False) as a:
            assert set(a.files)==set(obs['arrays'])
            np.testing.assert_array_equal(a['input'][:,:,0],prepared['p00'][split+'_x'][:100])
            for name,stored in obs['arrays'].items():
                actual=describe(a[name])
                for field,v in stored.items():
                    if isinstance(v,float):np.testing.assert_allclose(actual[field],v,atol=1e-30,rtol=1e-12)
                    else:assert actual[field]==v,(key,name,field)
            for condition,g in obs['gradients'].items():
                np.testing.assert_array_equal(a['labels_'+condition][:,0],prepared[condition]['train_observed_y'][:100])
                inverse={v:k for k,v in g['arrays'].items()}
                delta=(a['probability'].astype(np.float64)-a['labels_'+condition])/100
                np.testing.assert_allclose(a[inverse['attack_probability/bias']],delta.sum(axis=0),atol=1e-7,rtol=1e-5)
                np.testing.assert_allclose(a[inverse['attack_probability/kernel']],
                    a['classifier_hidden'].astype(np.float64).T@delta,atol=1e-12,rtol=1e-4)
                gradients_checked+=1
            if split=='test':
                level='p00' if key.startswith('initial') else key.split('_')[1]
                with np.load(previous/level/'representations.npz',allow_pickle=False) as old:
                    np.testing.assert_array_equal(a['intermediate'][:,:,0],old['initial' if key.startswith('initial') else 'final'][:100])
                    if key.startswith('initial'):np.testing.assert_array_equal(a['probability'][:,0],old['initial_probabilities'][:100])
                if not key.startswith('initial'):
                    with np.load(previous/level/'predictions.npz',allow_pickle=False) as old:
                        np.testing.assert_array_equal(a['probability'][:,0],old['raw_probabilities'][:100])
                assert obs['saved_test_probability_and_representation_exact']
    assert gradients_checked==4
    return {'status':'verified','observations':6,'native_saved_test_batches_exact':3,
        'independent_output_bias_and_kernel_derivatives_checked':4,
        'input_arrays_verified':20,'original_files_verified':15,'source_and_output_hashes_verified':True,
        'all_saved_summaries_replayed':True,'model_inferences':0,'model_fits':0}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--preparation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',action='store_true')
    a=p.parse_args()
    if a.audit:
        print(json.dumps(audit(a.output,a.previous,a.preparation),indent=2,sort_keys=True))
    else:
        r=run(a.previous,a.preparation,a.output)
        raise SystemExit(0 if r['status']=='complete' else 2)
