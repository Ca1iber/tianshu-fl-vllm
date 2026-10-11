# SPDX-License-Identifier: Apache-2.0
"""Bitwise GPU row-copy checks using an independent CPU oracle."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists():
        raise RuntimeError('Refusing to overwrite existing validation')
    lease = Path('/workspace/benchmark-results/20261011_121020_baseline_tianshu/reservation_deadline.json')
    deadline = datetime.fromisoformat(json.loads(lease.read_text())['stop_at_utc'])
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        raise RuntimeError('Reservation stop time reached')
    signal.alarm(min(1800, remaining))
    source = Path(__file__).with_name('experimental_gather.py')
    spec = importlib.util.spec_from_file_location('tianshu_experimental_gather', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    torch.manual_seed(2026101250)
    result = {'complete': False, 'official_benchmark': False,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'installed_files_modified': False, 'cases': [], 'adoption_allowed': False}
    with torch.inference_mode():
        for source_rows in (1, 565, 2048, 4096, 8192):
            for strided in (False, True):
                cpu_storage = torch.randn(source_rows, 4096 if strided else 2048,
                                          dtype=torch.bfloat16)
                cpu = cpu_storage[:, ::2] if strided else cpu_storage
                storage = cpu_storage.cuda()
                hidden = storage[:, ::2] if strided else storage
                for dtype in (torch.int32, torch.int64):
                    for rows in (0, 1, 33, 64, 256):
                        indices_cpu = torch.randint(source_rows, (rows,), dtype=dtype)
                        indices = indices_cpu.cuda()
                        expected = cpu[indices_cpu.long()]
                        output = module.gather_rows(hidden, indices)
                        exact = torch.equal(output.cpu(), expected)
                        tail = torch.full((rows + 13, 2048), 17., device='cuda',
                                          dtype=hidden.dtype)
                        if rows:
                            module._gather_rows[(rows, 8)](
                                hidden, indices, tail, source_rows, rows,
                                hidden.stride(0), hidden.stride(1), WIDTH=2048,
                                BLOCK=256, num_warps=4)
                        tail_check = torch.all(tail[rows:] == 17.).item()
                        copied_check = torch.equal(tail[:rows].cpu(), expected)
                        case = dict(source_rows=source_rows, selected_rows=rows,
                                    strided_columns=strided, indices_dtype=str(dtype),
                                    cpu_reference_bitwise=exact,
                                    padded_tail_unchanged=tail_check,
                                    preallocated_copy_bitwise=copied_check)
                        result['cases'].append(case)
                        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
                        if not (exact and tail_check and copied_check):
                            raise RuntimeError('Gather validation failed: ' + str(case))
        result['complete'] = True
        result['passed_cases'] = len(result['cases'])
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'complete': True, 'passed_cases': result['passed_cases']}))


if __name__ == '__main__':
    main()
