# SPDX-License-Identifier: Apache-2.0
"""Read-only layer snapshots for the failed padded-request model comparison."""
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import re


def install(module, directory):
    import torch
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context, is_forward_context_available
    directory = Path(directory)
    original_load = module.ModelRunnerFL.load_model
    original_forward = module.ModelRunnerFL._model_forward
    captured = {}
    active = [None]
    installed = {}

    def save(name, value):
        directory.mkdir(parents=True, exist_ok=True)
        p = directory / f'{name}_{os.getpid()}.json'
        tmp = p.with_suffix('.tmp');tmp.write_text(json.dumps(value, indent=2)+'\n');tmp.replace(p)

    def key(context):
        d = context.batch_descriptor
        if (d is not None and not d.uniform and d.num_reqs == 64
                and d.num_tokens in (64,72,80,88)):
            return d.num_reqs, d.num_tokens
        return None

    def tensors(value, path):
        if isinstance(value, torch.Tensor):
            yield path, value
        elif isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                yield from tensors(child, path+'.'+str(i))

    def hook(name, input_phase):
        def observe(owner, args, output=None):
            if not is_forward_context_available():
                return
            ctx = get_forward_context()
            capture_key = key(ctx)
            if (ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL and capture_key is not None
                    and torch.cuda.is_current_stream_capturing()):
                target = captured.setdefault(capture_key, {})
                count = capture_key[1]
            elif active[0] is not None and ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE:
                d = ctx.batch_descriptor
                assert d is not None
                label = 'padded' if (d.num_reqs,d.num_tokens) == active[0]['key'] else 'unpadded'
                target = active[0]['references'].setdefault(label, {})
                count = d.num_tokens
            else:
                return
            values = list(tensors(args if input_phase else output, 'in' if input_phase else 'out'))
            for suffix, tensor in values:
                if tensor.is_cuda and tensor.ndim and tensor.shape[0] == count:
                    target[name+'/'+suffix] = tensor.detach().clone()
        return observe

    @wraps(original_load)
    def load(owner, *args, **kwargs):
        result = original_load(owner, *args, **kwargs)
        model = owner.get_model()
        names = dict(model.named_modules())
        decoders = sorted((int(re.search(r'\.layers\.(\d+)$',n).group(1)),n)
            for n in names if re.search(r'\.layers\.(\d+)$',n))
        assert [i for i,n in decoders] == list(range(42))
        first = decoders[0][1]
        targets = [(n, False) for i,n in decoders]
        wanted = ('input_layernorm','self_attn.qkv_proj','self_attn.rotary_emb',
                  'self_attn.attn','self_attn.o_proj','post_attention_layernorm',
                  'mlp.gate_up_proj','mlp.act_fn','mlp.down_proj')
        for child in wanted:
            name = first+'.'+child
            assert name in names, name
            targets.append((name,True))
        targets.extend((n,False) for n in names if n.endswith('embed_tokens') or n == first.rsplit('.layers.',1)[0]+'.norm')
        for name, detail in targets:
            if detail:
                names[name].register_forward_pre_hook(hook(name,True))
            names[name].register_forward_hook(hook(name,False))
        installed.update(targets=[n for n,d in targets], first_decoder=first,
            model_type=type(model).__module__+'.'+type(model).__qualname__,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            extra_clones_inside_capture=True, model_outputs_modified=False,
            purpose='Diagnostic only; retains original failing numeric gate')
        save('installed', installed)
        return result

    def compare(a, b, count):
        assert a.shape[1:] == b.shape[1:] and a.shape[0] >= count and b.shape[0] >= count
        a,b = a[:count],b[:count]
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        if not finite:
            return dict(passed=False,finite=False)
        diff=(a.float()-b.float()).abs();limit=.02+.01*b.float().abs()
        bad=int((diff>limit).sum().item())
        return dict(passed=bad==0,finite=True,bad_elements=bad,max_abs=diff.max().item(),
                    max_tolerance_ratio=(diff/limit).max().item(),bitwise=bool(torch.equal(a,b)),
                    shape=list(a.shape))

    @wraps(original_forward)
    def forward(owner,*args,**kwargs):
        ctx = get_forward_context() if is_forward_context_available() else None
        selected = key(ctx) if ctx is not None else None
        n=owner.input_batch.num_reqs
        eligible = (ctx is not None and ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
                    and selected is not None and 16<=n<64 and active[0] is None)
        if not eligible:
            return original_forward(owner,*args,**kwargs)
        frame = dict(key=selected,requests=int(n),queries=int(owner.query_start_loc.np[n]),references={})
        active[0] = frame
        error=None
        try:
            return original_forward(owner,*args,**kwargs)
        except BaseException as caught:
            error=repr(caught)
            raise
        finally:
            active[0]=None
            if True:
                record=dict(key=list(selected),requests=frame['requests'],queries=frame['queries'],
                            original_numeric_error=error,diagnostic_only=True,installed=installed)
                try:
                    candidate=captured[selected];refs=frame['references']
                    assert candidate, 'No layer snapshots were captured'
                    assert {'padded','unpadded'} <= set(refs)
                    assert set(candidate)==set(refs['padded'])==set(refs['unpadded'])
                    rows=[]
                    for name,tensor in candidate.items():
                        rows.append(dict(name=name,padded=compare(tensor,refs['padded'][name],frame['queries']),
                                         unpadded=compare(tensor,refs['unpadded'][name],frame['queries'])))
                    record.update(complete=True,rows=rows,
                        first_unpadded_numeric_difference=next((r['name'] for r in rows if not r['unpadded']['passed']),None),
                        first_unpadded_bit_difference=next((r['name'] for r in rows if not r['unpadded']['bitwise']),None),
                        all_padded_bitwise=all(r['padded']['bitwise'] for r in rows))
                except BaseException as trace_error:
                    record.update(complete=False,trace_error=repr(trace_error))
                save('trace',record)

    module.ModelRunnerFL.load_model=load
    module.ModelRunnerFL._model_forward=forward
