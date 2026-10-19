# SPDX-License-Identifier: Apache-2.0
"""All arms call the unchanged, measured K8V16 writer."""
import importlib.util
from pathlib import Path

OWNER = Path(__file__).resolve().parents[1] / 'tianshu_soft8_mixed_planes_20261019/writer.py'
spec = importlib.util.spec_from_file_location('_k8v16_sweep_actual_writer', OWNER)
owned = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owned)
write, encode_all = owned.write, owned.encode_all
