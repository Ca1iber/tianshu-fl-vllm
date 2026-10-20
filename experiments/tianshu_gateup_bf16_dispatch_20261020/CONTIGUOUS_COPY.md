# Explicit contiguous-input normalization diagnostic

Follow-up to the preserved M565/M1537 row-stride failures. Test both layouts for both methods with the explicit a.contiguous() operation inside the timed Graph. Numerical tolerance is unchanged. Every method must pass every phase/layout; the runner requires all four cases to have both mm and linear true. The copy is real work included in graph timing, not an input substitution outside measurement. No service installation or model score is asserted.
