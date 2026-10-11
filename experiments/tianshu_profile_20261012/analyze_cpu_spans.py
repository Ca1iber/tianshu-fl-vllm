# SPDX-License-Identifier: Apache-2.0
"""Separate recorded GPU work from inclusive host spans in a torch trace."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics


def union_us(intervals):
    total=0
    end=None
    for start,stop in sorted(intervals):
        if end is None or start>=end:
            total+=stop-start
            end=stop
        elif stop>end:
            total+=stop-end
            end=stop
    return total


def analyze(path):
    with gzip.open(path,'rt') as f:
        data=json.load(f)
    groups=defaultdict(list)
    intervals=defaultdict(list)
    for event in data.get('traceEvents',[]):
        if event.get('ph')!='X' or event.get('dur',0)<=0:
            continue
        category=event.get('cat','').lower()
        if category not in ('kernel','cuda_driver','cuda_runtime','cpu_op'):
            continue
        name=event.get('name','')
        groups[(category,name)].append(event['dur'])
        intervals[category].append((event['ts'],event['ts']+event['dur']))
    results={}
    for category,events in intervals.items():
        rows=[]
        for (cat,name),durations in groups.items():
            if cat!=category:continue
            values=sorted(durations)
            rows.append({'name':name,'calls':len(values),'inclusive_ms':sum(values)/1000,
                         'median_ms':statistics.median(values)/1000,
                         'p99_ms':values[min(len(values)-1,int(.99*len(values)))]/1000,
                         'max_ms':values[-1]/1000})
        results[category]={'recorded_union_ms':union_us(events)/1000,
                           'recorded_span_ms':(max(e for _,e in events)-min(s for s,_ in events))/1000,
                           'largest_inclusive_spans':sorted(rows,key=lambda x:x['inclusive_ms'],reverse=True)[:25]}
    return {'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'categories':results,'limits':[
                'Host durations are inclusive; nested CPU operations must not be added as self time.',
                'The union spans recorded events across threads; waiting can be included.',
                'Graph replay kernel-body coverage is not established by this analysis.',
                'A cold or short-warmup trace cannot establish steady production proportions.']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.trace)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
