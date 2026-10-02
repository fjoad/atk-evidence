"""Constructed discriminating controls for the saved-state observer."""
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import expit
from tests.test_robust_feed_forward import HAS_TF

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'studies/takiddin-2021-robust-poisoning/checks'))
import sequential_saved_state as D


class SummaryControls(unittest.TestCase):
    def test_time_variation_is_not_profile_variation(self):
        a=D.describe(np.array([[0.,1.],[0.,1.]],dtype=np.float32))
        b=D.describe(np.array([[0.,1.],[0.,2.]],dtype=np.float32))
        self.assertEqual(a['max_axis0_range'],0)
        self.assertEqual(b['max_axis0_range'],1)
        self.assertEqual(a['zero_count'],2)

    def test_nonfinite_values_remain_explicit(self):
        r=D.describe(np.array([[np.nan,0.],[np.inf,1.]],dtype=np.float32))
        self.assertEqual(r['finite_count'],2)
        self.assertIsNone(r['max_axis0_range'])
        self.assertIsNone(r['l2'])

    def test_constant_float32_probability_can_hide_affine_variation(self):
        r=D.readout_arithmetic(np.array([[20.],[21.]],dtype=np.float32),
                              np.ones((1,1),dtype=np.float32),np.zeros(1,dtype=np.float32))
        self.assertEqual(np.ptp(expit(r['readout_logits_float32'])),0)
        self.assertGreater(np.ptp(r['readout_probability_float64']),0)
        self.assertEqual(np.ptp(r['readout_logits_float64']),1)


@unittest.skipUnless(HAS_TF,'requires pinned isolated neural runtime')
class ObserverControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf
        cls.tf=tf
        D.R.configure_tensorflow(False)

    def make_model(self):
        return D.M.sequential_ensemble(20260920,encoder_units=(8,6,4),gru_units=8,
                                     dense_units=16,timesteps=8,bridge_activation='sigmoid')

    def test_active_observer_preserves_native_outputs_and_label_gradient_signs(self):
        m=self.make_model()
        x=np.stack([np.full((8,1),-.75),np.full((8,1),.75)]).astype(np.float32)
        a,r=D.observe(m,x,{'zero':np.zeros((2,1),dtype=np.float32),
                           'one':np.ones((2,1),dtype=np.float32)})
        np.testing.assert_array_equal(a['probability'],m(x,training=False).numpy())
        self.assertGreater(r['arrays']['intermediate']['max_axis0_range'],0)
        self.assertGreater(np.count_nonzero(a['classifier_hidden']),0)
        values=[]
        for case in ('zero','one'):
            paths=r['gradients'][case]['arrays']
            bias=next(k for k,v in paths.items() if v=='attack_probability/bias')
            values.append(float(a[bias][0]))
        self.assertGreater(values[0],0)
        self.assertLess(values[1],0)
        self.assertAlmostEqual(values[0]-values[1],1.,places=6)

    def test_dead_head_has_zero_upstream_gradient_but_live_output_bias(self):
        m=self.make_model();h=m.get_layer('classifier_hidden')
        h.kernel.assign(self.tf.zeros_like(h.kernel));h.bias.assign(-self.tf.ones_like(h.bias))
        x=np.stack([np.full((8,1),-.75),np.full((8,1),.75)]).astype(np.float32)
        before=D.state_hash(m)
        a,r=D.observe(m,x,{'positive':np.ones((2,1),dtype=np.float32)})
        self.assertEqual(D.state_hash(m),before)
        self.assertGreater(r['arrays']['input']['max_axis0_range'],0)
        self.assertEqual(np.count_nonzero(a['classifier_hidden']),0)
        paths=r['gradients']['positive']['arrays']
        bias=next(k for k,v in paths.items() if v=='attack_probability/bias')
        upstream=next(k for k,v in paths.items() if v=='input')
        self.assertAlmostEqual(float(a[bias][0]),-.5,places=6)
        self.assertEqual(np.count_nonzero(a[upstream]),0)
        np.testing.assert_array_equal(a['probability'],m(x,training=False).numpy())
        self.assertEqual(r['optimizer_updates'],0)

    def test_saturated_bridge_is_observed_without_repair(self):
        m=self.make_model();f=m.get_layer('attention_decoder')
        f.projection.kernel.assign(self.tf.zeros_like(f.projection.kernel));f.projection.bias.assign([-1000.])
        x=np.full((2,8,1),.75,dtype=np.float32)
        a,r=D.observe(m,x,{'positive':np.ones((2,1),dtype=np.float32)})
        np.testing.assert_array_equal(a['bridge_logits'],-1000.)
        self.assertEqual(np.count_nonzero(a['intermediate']),0)
        self.assertEqual(np.count_nonzero(a['bridge_sigmoid_derivative']),0)
        self.assertEqual(r['state_before_sha256'],r['state_after_sha256'])
        self.assertEqual(r['observer_bridge_max_error'],0)


if __name__=='__main__':unittest.main()
