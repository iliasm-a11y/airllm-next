from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'air_llm'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests_ultra'))
import torch
import mlx.core as mx
import pytest

torch.set_num_threads(1)
mx.set_default_device(mx.cpu)

# Load the unchanged Ultra fixtures by path to avoid same-name conftest imports.
import importlib.util
_spec=importlib.util.spec_from_file_location('ultra_fixtures',Path(__file__).resolve().parents[1]/'tests_ultra/conftest.py')
_fixtures=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixtures)
save_tiny=_fixtures.save_tiny

@pytest.fixture
def tiny_glm(tmp_path):
    path=tmp_path/'glm';ref=save_tiny(path,'glm');return path,ref
