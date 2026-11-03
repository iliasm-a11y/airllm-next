"""Synthetic refusal checks for a new dependency domain, never native proofs."""
import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).with_name('variant_compare.py')
spec=importlib.util.spec_from_file_location('new_backend_policy_validator',p)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
def observation():
    return {'registry_before':['CPU','BLAS','Metal'],'registry_after':['CPU','Metal'],
        'policy':'unregister-BLAS-before-model-v1','BLAS_registered_after':False,'CPU_registered_after':True,
        'accelerator_devices_after':0,'model_loaded':False,'bridge_matrix_dispatches':0}
class Policy(unittest.TestCase):
    def test_missing_actual_registration_observation_is_refused(self):
        for value in (None,{},False,{'policy':'unregister-BLAS-before-model-v1'}):
            with self.assertRaises(ValueError):v.require_backend_policy(value)
    def test_contradictory_transition_and_changed_proof_types_are_refused(self):
        for field,value in (('registry_before',['CPU','Metal']),('registry_after',['CPU','BLAS','Metal']),
            ('registry_after',['CPU']),('registry_after',['CPU','Metal','Metal']),('BLAS_registered_after',0),
            ('CPU_registered_after',1),('accelerator_devices_after',False),('model_loaded',True),
            ('bridge_matrix_dispatches',False),('bridge_matrix_dispatches',1),('policy','default')):
            row=observation();row[field]=value
            with self.subTest(field=field,value=value):
                with self.assertRaises(ValueError):v.require_backend_policy(row)
if __name__=='__main__':unittest.main()
