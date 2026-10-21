# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

def with_warmup(original,warm,finished):
    @contextmanager
    def profile(*args,**kwargs):
        with original(*args,**kwargs) as result:
            warm(result)
            yield result
        finished(result)
    return profile
