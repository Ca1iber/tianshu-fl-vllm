# SPDX-License-Identifier: Apache-2.0
"""Keep handlers/filters and INFO+, change only the FlagGems DEBUG threshold."""
import logging

def describe(logger):
    return dict(name=logger.name,level=logger.level,effective_level=logger.getEffectiveLevel(),disabled=logger.disabled,
        propagate=logger.propagate,handlers=[dict(identity=id(h),kind=type(h).__module__+'.'+type(h).__qualname__,
            level=h.level,owned=getattr(h,'_flaggems_owned',False),filename=getattr(h,'baseFilename',None),
            filters=[dict(identity=id(f),kind=type(f).__module__+'.'+type(f).__qualname__) for f in h.filters]) for h in logger.handlers])
def apply(logger,mode):
    assert mode in ('control','candidate') and logger.getEffectiveLevel()==logging.DEBUG and not logger.disabled
    before=describe(logger)
    if mode=='candidate':logger.setLevel(logging.INFO)
    after=describe(logger)
    assert after['handlers']==before['handlers'] and after['propagate']==before['propagate'] and not after['disabled']
    assert all(logger.isEnabledFor(level) for level in (logging.INFO,logging.WARNING,logging.ERROR,logging.CRITICAL))
    assert logger.isEnabledFor(logging.DEBUG)==(mode=='control')
    return dict(complete=True,mode=mode,before=before,after=after,INFO_WARNING_ERROR_CRITICAL_preserved=True,
        handlers_and_filters_preserved=True,operator_functions_and_data_and_RNG_not_accessed=True)
