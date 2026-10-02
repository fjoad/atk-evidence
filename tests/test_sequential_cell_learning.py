"""Source-explicit cell activation and full-length fixture controls; no CER."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
import numpy as np
from tests.test_robust_feed_forward import HAS_TF

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_cell_learning as C

class FixtureContract(unittest.TestCase):
    def test_full_length_disjoint_fixture_and_reversed_zero_parameter_floor(self):
        a,b=C.fixture_data(),C.fixture_data()
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(a['train_x'].shape,(32,48,1))
        self.assertFalse(np.array_equal(a['train_x'],a['test_x']))
        for split in ('train','test'):
            y=a[split+'_y'].ravel();self.assertEqual(y.sum(),16)
            mean=a[split+'_x'].mean(axis=(1,2))>0
            np.testing.assert_array_equal(mean,y);np.testing.assert_array_equal(~mean,1-y)

    def test_other_detector_functions_are_unchanged(self):
        old=subprocess.check_output(['git','show','40f3bba:studies/takiddin-2021-robust-poisoning/reproduction/models.py'],cwd=ROOT,text=True)
        current=(C.STUDY/'reproduction/models.py').read_text()
        funcs=lambda text:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        a,b=funcs(old),funcs(current)
        for k in a:
            if k not in ('register_sequential_layer','sequential_ensemble'):self.assertEqual(a[k],b[k],k)

@unittest.skipUnless(HAS_TF,'requires pinned neural runtime')
class CellActivation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import keras
        cls.tf,cls.keras=tf,keras
        C.R.configure_tensorflow(False)
    def small(self,activation):
        return C.M.sequential_ensemble(20260920,encoder_units=(8,6,4),gru_units=8,dense_units=16,
            timesteps=48,bridge_activation='sigmoid',cell_activation=activation)

    def test_only_cell_activation_changes_and_initial_weights_match(self):
        hashes=[]
        for activation in ('relu','tanh'):
            m=self.small(activation);hashes.append(C.R.weight_hash(m.get_weights()));f=m.get_layer('attention_decoder')
            for layer in [*f.encoder,*f.decoder,*[m.get_layer(f'sequence_gru_{i}') for i in range(1,9)]]:
                self.assertEqual(layer.activation.__name__,activation)
                self.assertEqual(layer.recurrent_activation.__name__,'sigmoid')
            self.assertEqual(m.get_layer('classifier_hidden').activation.__name__,'relu')
            self.assertEqual(f.projection.activation.__name__,'sigmoid')
            self.assertEqual(m.get_layer('attack_probability').activation.__name__,'sigmoid')
            self.assertEqual(f.get_config()['cell_activation'],activation)
        self.assertEqual(hashes[0],hashes[1])
        with self.assertRaises(ValueError):self.small('linear')

    def test_tanh_hidden_bounds_under_large_signed_inputs(self):
        m=self.small('tanh');x=np.stack([np.full((48,1),-1000),np.full((48,1),1000)]).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:r=C.layer_snapshot(m,x,Path(d)/'layers.npz')
        for key in ['encoder_memory','decoder_hidden',*[f'gru_{i}' for i in range(1,9)]]:
            self.assertLessEqual(max(abs(r[key]['min']),abs(r[key]['max'])),1.000001)
            self.assertEqual(r[key]['finite_count'],r[key]['size'])

    def test_legacy_archive_and_new_tanh_reload_in_fresh_process(self):
        source=subprocess.check_output(['git','show','40f3bba:studies/takiddin-2021-robust-poisoning/reproduction/models.py'],cwd=ROOT,text=True)
        historical=types.ModuleType('old_sequential_cells');exec(compile(source,'old_sequential_cells','exec'),historical.__dict__)
        x=C.fixture_data()['test_x'][:2]
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for activation in ('relu','tanh'):
                if activation=='relu':m=historical.sequential_ensemble(20260920,encoder_units=(8,6,4),gru_units=8,dense_units=16,timesteps=48,bridge_activation='sigmoid')
                else:
                    self.keras.saving.register_keras_serializable(package='atk_evidence')(C.M.register_sequential_layer())
                    m=self.small('tanh')
                m.train_on_batch(x,np.array([[0.],[1.]],dtype=np.float32))
                m.save(d/(activation+'.keras'))
                f=self.keras.Model(m.input,[m.get_layer('attention_decoder').output[0],m.get_layer('sequence_gru_1').output,m.get_layer('classifier_hidden').output,m.outputs[0]])
                np.savez(d/(activation+'.npz'),x=x,**{f'v{i}':v.numpy() for i,v in enumerate(f(x,training=False))})
            code='''import sys,numpy as np,keras
from pathlib import Path
sys.path.insert(0,sys.argv[1]);import models;models.register_sequential_layer()
p=Path(sys.argv[2])
for activation in ('relu','tanh'):
 m=keras.models.load_model(p/(activation+'.keras'))
 assert m.get_layer('attention_decoder').cell_activation==activation
 assert int(m.optimizer.iterations.numpy())==1
 f=keras.Model(m.input,[m.get_layer('attention_decoder').output[0],m.get_layer('sequence_gru_1').output,m.get_layer('classifier_hidden').output,m.outputs[0]])
 with np.load(p/(activation+'.npz')) as a:
  for i,v in enumerate(f(a['x'],training=False)):np.testing.assert_array_equal(v.numpy(),a[f'v{i}'])
'''
            check=subprocess.run([sys.executable,'-c',code,str(C.STUDY/'reproduction'),str(d)],capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr)
        self.keras.saving.register_keras_serializable(package='atk_evidence')(C.M.register_sequential_layer())

    def test_partial_schedule_cannot_pass_and_preserves_artifacts(self):
        with tempfile.TemporaryDirectory() as d:
            r=C.run_case(Path(d)/'case',C.fixture_data(),'tanh',False,updates=2,fit_guard_seconds=0,
                fixture_dimensions={'encoder_units':(8,6,4),'gru_units':8,'dense_units':16})
            self.assertEqual(r['status'],'time_guard');self.assertEqual(r['updates'],1)
            self.assertFalse(r['case_gate_passed']);self.assertTrue(r['reload_exact'])
            self.assertTrue((Path(d)/'case/final.keras').exists())

if __name__=='__main__':unittest.main()
