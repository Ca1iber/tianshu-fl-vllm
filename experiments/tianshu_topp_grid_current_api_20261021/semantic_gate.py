# SPDX-License-Identifier: Apache-2.0
"""Independent request semantics: baseline equality OR the original strict oracle."""
CONTRACT='EVERY_ROW: retained_unchanged AND (exact_current_baseline OR strict_CPU_1e-6)'
COLUMNS=('kept_mass','smallest_kept_probability','CPU_p','minimum_kept_logit','maximum_removed_logit','nonempty')

def verdict(mass,smallest,p,low,high,nonempty,exact,retained):
    strict=bool(nonempty and mass>=p-1e-6 and mass-smallest<=p+1e-6 and high<=low)
    return dict(exact_current_baseline=bool(exact),strict_CPU=strict,retained_unchanged=bool(retained),
                relative_gate=bool(retained and (exact or strict)))

def assess_rows(actual,expected,source,summary):
    import torch
    assert actual.device.type==expected.device.type==source.device.type=='cpu'
    rows=[]
    for i,scalars in enumerate(zip(*(summary[k] for k in COLUMNS))):
        finite=torch.isfinite(actual[i])
        exact=bool(torch.equal(actual[i],expected[i]))
        retained=bool(torch.equal(actual[i][finite],source[i][finite]))
        row=verdict(*scalars,exact,retained)
        row.update(row=i,adopted_mask_difference=int((finite!=torch.isfinite(expected[i])).sum()))
        rows.append(row)
    assert all(r['exact_current_baseline'] for r in rows)==summary['exact_adopted_baseline']
    assert all(r['retained_unchanged'] for r in rows)==summary['retained_values_unchanged']
    assert all(r['strict_CPU'] for r in rows)==summary['CPU_strict_semantics']
    assert sum(r['adopted_mask_difference'] for r in rows)==summary['adopted_mask_difference']
    return rows

def check_cpu():
    good=(.9500001,.000002,.95,-.6,-.61,True)
    bad_order=(.9500001,.000002,.95,-.6,-.5999999,True)
    bad_mass=(.9,.000002,.95,-.6,-.61,True)
    count=0
    for scalars,exact,retained,want in ((good,False,True,True),(good,True,True,True),
        (bad_order,True,True,True),(bad_order,False,True,False),
        (bad_mass,True,True,True),(bad_mass,False,True,False),
        (good,True,False,False),(good,False,False,False)):
        assert verdict(*scalars,exact,retained)['relative_gate']==want;count+=1
    # An unchanged baseline edge in one request must not invalidate an independently valid change in another.
    assert all(verdict(*s,e,True)['relative_gate'] for s,e in ((bad_order,True),(good,False)))
    assert not all(verdict(*s,e,True)['relative_gate'] for s,e in ((bad_order,False),(good,False)))
    return count+2
