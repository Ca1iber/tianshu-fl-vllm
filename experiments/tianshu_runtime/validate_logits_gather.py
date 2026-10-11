#!/usr/bin/env python3
"""Bitwise checks against CPU indexing, plus changing-input graph replay."""
import json
from pathlib import Path
import sys
import torch
from iluvatar_logits_gather import gather_rows


def main():
    torch.manual_seed(20261011)
    cases = []
    for rows in [565, 576, 1537, 1664, 2048]:
        for strided in [False, True]:
            width = 4096 if strided else 2048
            source = torch.randn(rows, width, device='cuda', dtype=torch.bfloat16)
            if strided:
                source = source[:, ::2]
            cpu = source.cpu()
            for dtype in [torch.int32, torch.int64]:
                for count in [0, 1, 3, 16, 64, 256]:
                    idx = torch.randint(0, rows, (count,), dtype=dtype, device='cuda')
                    if count:
                        idx[0] = rows - 1
                    result = gather_rows(source, idx)
                    expected = cpu[idx.cpu().long()]
                    assert torch.equal(result.cpu().view(torch.int16), expected.view(torch.int16))
                    cases.append(dict(source_rows=rows, selected_rows=count, strided=strided,
                                      indices_dtype=str(dtype), bitwise_equal=True))
    source = torch.randn(1537, 2048, device='cuda', dtype=torch.bfloat16)
    idx = torch.arange(64, device='cuda', dtype=torch.int32)
    for _ in range(3):
        gather_rows(source, idx)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gather_rows(source, idx)
    for _ in range(3):
        source.copy_(torch.randn_like(source))
        idx.copy_(torch.randint(0, 1537, (64,), device='cuda', dtype=torch.int32))
        graph.replay()
        expected = source.cpu()[idx.cpu().long()]
        assert torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
    report = dict(passed=True, cases=cases, case_count=len(cases), changing_input_graph_replays=3,
                  contract='Scheduler supplies valid nonnegative row indices; no general advanced indexing semantics.')
    Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
    print('Bitwise cases passed:', len(cases), '; changing-input graph replays: 3', flush=True)


if __name__ == '__main__':
    main()
