# Distinct resident-graph measurement v2

The original 23-case suite remains failed with FIXTURE_RSS_LIMIT at138133504B.
This separate21-case suite excludes M8192/K4096/N64 and128; they are not qualified.
Each completed graph executes five times in the same process with the same
resident synthetic weights. The first operation includes provider initialization.
The following four operations are timed separately, with every output SHA256
required equal. Full CPU/GPU raw output and all native terminal/publication
traces remain checked. Expected deliberate fatal still stops on its first tile.
All RSS128MiB, logical64MiB, Metal64MiB, swap32MiB, original native/power/UID
guards and deadlines remain unchanged. No inference or SSD speed is measured.
