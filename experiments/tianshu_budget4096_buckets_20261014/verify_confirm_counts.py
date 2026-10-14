# SPDX-License-Identifier: Apache-2.0
"""Correct only a legacy summary label after full640-request confirmation audit.

The frozen original verifier checks all five rounds but reports the old three-
round pair count192. Preserve that source while the live quality chain freezes
it; independently recount all pairwise texts and report the correct320 pairs.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    result=subprocess.run(['python3',str(Path(__file__).with_name('verify_downloaded_confirm.py')),
                           '--root',str(root)],check=True,capture_output=True,text=True)
    audit=json.loads(result.stdout)
    rounds=[json.loads((root/mode/'result.json').read_text())['short_4k_rounds']
            for mode in ('control','candidate')]
    assert len(rounds[0])==len(rounds[1])==5
    total=equal=0
    for a,b in zip(*rounds):
        assert len(a['responses'])==len(b['responses'])==64
        for x,y in zip(a['responses'],b['responses']):
            assert x['seed']==y['seed'];total+=1;equal+=x['text']==y['text']
    assert total==320 and audit['successful_requests']==2*total==640
    assert equal==audit['paired_texts_equal']
    audit.update(legacy_verifier_paired_total_label=audit['paired_total'],paired_total=total,
                 pair_count_independently_recomputed=True)
    (root/'local_download_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':main()
