# Fresh8192 capacity supersedes the small-screen planning input

The historical KV_6144_THEORETICAL_CAPACITY_20261021.json and NEXT_KV_PREFILL_BOUND_REVIEW_20261021.md used the actual small8192screen capacity40,219blocks. They remain correct descriptions of that observation and its conditional CPU arithmetic. Fresh8192Math and the currentformal4K both record40,178blocks including one reserved null block. Same445runtime/64environment does not identify the cause of the41block difference.

For nominal independent16Kinput+1Koutput, native FutureKV forecast isceil(17408/16)=1088blocks/request. Thirty-seven requests need40,256usable blocks, or40,257total including null. At40,178actualtotal, the gap is79blocks, not38. At516,096bytes per all42layer KVblock:

| Arena pages | Bytes saved versus8192 | Nominal extra blocks | Nominal total using40178 | Reaches37forecasts arithmetically |
| --- | ---: | ---: | ---: | --- |
|6144|33,554,432|65|40,243|no|
|5120|50,331,648|97|40,275|yes,18blockmargin|
|4096|67,108,864|130|40,308|yes,51blockmargin|

This is CPU allocation arithmetic, not measured GPU capacity, admission or16Kperformance. Actual memory peaks/reserves/allocator behavior may change the result. The initial bound-only test must retain arena8192 to isolate the tighter prefill maxK factor. A separate capacity test can use4096with existing safe fallback after correctness evidence; first measure actual recovered blocks/admissions. PartialMath stage maximum2404pages is an observation, not a universal bound. No profile adoption or16Kscore follows from this note.
