# M5 recovery evidence — partial campaign

The draft PR has measured hardware results. This directory does not claim that the 100-job campaign is complete.

- `gpu/`: full raw 160-case grid, guard and independent C diagnostic; 288 raw tensor files. Reuse is bitwise exact to the original GPU reference in all128cases. TensorOps is globally rejected under the existing numeric contract.
- `h2-reference16-independent.json`: C3 two-pair AB/BA accepted strict evidence summary, source1a9c727. Full original JSON/FP32 files are preserved locally in the selected ZIP.
- `h2-reference16-final-independent.json`: independent revalidation of the repeated accepted C5 block on the final source treef278b90.
- `router/public-numeric-proof.json`: targeted FP32 activation rows and router weights with hashes and exact rational replay. Run `python3 router/verify-public-proof.py router/public-numeric-proof.json`. It requires only Pythonstdlib, loads no model and uses no GPU. It proves the recorded numeric mechanism, not full-model performance.
- `delivery-snapshot.json`: explicitly partial counts at its timestamp. A launched/clean worker alone is not an accepted paired result.
- `resource-snapshot.json` and `cleanup/`: separately measured RAM, compressed memory, swap, disk and audited SIGTERM-only helper cleanup. No checkpoints, proofs or user files were deleted.

The complete selected local ZIP has39,231,866bytes, SHA-256d5c29f90b896bae40eb6d80ee2801f8dd15108602332f1f4bf3fbd92fb30611e,1023members verified byCRC andSHA256. It preserves fourC3accepted workers,288GPUrawfiles,323historical protocol/source files and router captures. It is a partial selection of an incomplete campaign and is not uploaded as a Git binary. Full raw evidence of all attempts stays in the local outputs folder.
