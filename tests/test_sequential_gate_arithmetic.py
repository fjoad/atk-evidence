"""Analytic controls for native recurrent gate decomposition; no research data."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from tests.test_robust_feed_forward import HAS_TF
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_gate_arithmetic as G

@unittest.skipUnless(HAS_TF,'requires pinned neural runtime')
class GateArithmetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        import keras
        cls.tf,cls.keras=tf,keras
        G.R.configure_tensorflow(False)

    def test_gru_closed_candidate_has_exact_geometric_retention(self):
        tf=self.tf
        cell=self.keras.layers.GRUCell(1,activation='relu',reset_after=False,implementation=2)
        cell.build((2,1));cell.kernel.assign([[0.,0.,1.]])
        cell.recurrent_kernel.assign(tf.zeros_like(cell.recurrent_kernel));cell.bias.assign([0.,0.,-2.])
        _,trace=G.sequence(cell,tf.ones((2,8,1)),'gru',[tf.ones((2,1))])
        np.testing.assert_array_equal(trace['h'],np.broadcast_to(.5**np.arange(1,9)[None,:,None],(2,8,1)))
        with tempfile.TemporaryDirectory() as d:r=G.pack(trace,'gru',Path(d)/'test.npz')
        self.assertEqual(r['candidate_zero_tail_from_step'],1)
        self.assertEqual(r['blocked_by_bias_per_step'],[2]*8)
        self.assertEqual(r['weighted_injection_identity_max_error'],0)

    def test_lstm_closed_input_preserves_only_forget_scaled_memory(self):
        tf=self.tf;c=self.keras.layers.LSTMCell(1,activation='relu',implementation=2)
        c.build((2,1));c.kernel.assign(tf.zeros_like(c.kernel));c.recurrent_kernel.assign(tf.zeros_like(c.recurrent_kernel))
        c.bias.assign([-1000.,0.,0.,0.])
        _,v=G.step(c,tf.zeros((2,1)),[tf.ones((2,1))*2,tf.ones((2,1))*4],'lstm')
        np.testing.assert_array_equal(v['c'],2);np.testing.assert_array_equal(v['h'],1)
        np.testing.assert_array_equal(v['injected'],0)

    def test_unit_columns_do_not_bound_relu_lstm_state(self):
        tf=self.tf;c=self.keras.layers.LSTMCell(4,activation='relu',implementation=2)
        c.build((2,1));c.kernel.assign(tf.zeros_like(c.kernel))
        u=np.zeros((4,16),dtype=np.float32);u[:,8:12]=.5;c.recurrent_kernel.assign(u)
        b=np.full(16,20.,dtype=np.float32);b[8:12]=0;c.bias.assign(b)
        self.assertEqual(float(np.linalg.norm(u,axis=0).max()),1.)
        _,v=G.sequence(c,tf.zeros((2,5,1)),'lstm',[tf.ones((2,4)),tf.ones((2,4))])
        np.testing.assert_array_equal(v['h'],np.broadcast_to(3.**np.arange(1,6)[None,:,None],(2,5,4)))
        self.assertTrue(np.all(v['injected']>v['retained']))

    def test_full_observer_matches_native_sequences_and_preserves_state(self):
        tf=self.tf
        m=G.M.sequential_ensemble(20260920,encoder_units=(8,6,4),gru_units=8,dense_units=16,timesteps=8,bridge_activation='sigmoid')
        x=np.stack([np.full((8,1),-.75),np.full((8,1),.75)]).astype(np.float32)
        old,_=G.D.observe(m,x,{})
        before=G.D.state_hash(m)
        with tempfile.TemporaryDirectory() as d:r=G.inspect_model(m,tf.convert_to_tensor(x),old,Path(d))
        self.assertEqual(len(r['cells']),7)
        self.assertEqual(before,G.D.state_hash(m))
        self.assertTrue(r['native_saved_parity_exact'])
        self.assertEqual(G.library_hashes(),G.LIBRARY_HASHES)

if __name__=='__main__':unittest.main()
