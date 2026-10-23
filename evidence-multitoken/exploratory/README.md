# Exploratory controls, excluded from selected timing results

These first CPU/Metal batches used a target-generated continuation with EOS
stopping disabled. The first prompt contains repeated generated EOS tokens;
this can inflate reuse relative to ordinary completed generation. The arrays
still validate causal arithmetic, but the final batch campaign uses a declared
natural passage and the generation campaign stops at the configured EOS.
The Metal exploratory run also overlaps part of a CPU-only metadata replay.
The smoke runs remain diagnostics and are excluded from performance summaries.
