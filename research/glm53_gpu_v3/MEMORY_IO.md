# Memory and IO observer: scope and admission

`system_memory.py` and `process_io.py` add read-only observations. They do not
change production, launch a model, enumerate other processes, request privileged
task ports, or replace the existing RSS/swap/pressure guards.

## Measured sources

| Quantity | Source | Scope / limitation |
|---|---|---|
| Wired, active/inactive/free, file-backed, compressor pages | `vm_stat` | System-wide; page-size conversion uses the reported page size |
| Compressed physical memory | Pages occupied by compressor | Compressed storage itself, distinct from uncompressed page equivalents |
| Compressor logical page equivalents | Pages stored in compressor | Not physical RAM or SSD traffic |
| Available-memory estimate | Free + inactive + speculative pages | Reclaimability estimate; these categories must not be added again to file-backed/anonymous totals |
| Swap used and pressure | `sysctl` | Global; growth does not identify the causing process |
| Process RSS, wired, footprint | `proc_pid_rusage`, v4 or v2 fallback | One authorized PID; footprint is an OS ledger metric, not simply private RSS |
| Process OS-accounted storage bytes / pageins | `proc_pid_rusage` | Paging/read-ahead and non-weight IO may be included; no device/NAND attribution |
| Process faults, COW faults, context switches | `proc_pidinfo(PROC_PIDTASKINFO)` | One PID; signed32 overflow/reset is rejected, not converted into invented traffic |
| CPU core-seconds | Mach timebase + process ticks | Measured timebase; multicore time can exceed wall time and cannot be subtracted as pure weight wait |
| Compressed/internal/external/device bytes | `TASK_VM_INFO` | Observer self only; an external native engine requires a native self-report |
| Device reads / operations | Optional existing block-counter callback | Controller-global, includes other processes; never NAND cells or per-process physical transfers |

The observers deliberately report `None` for physical NAND bytes, per-PID device
transfer bytes, exact file-cache-hit bytes, allocation/copy bytes and pure weight
wait. `faults - pageins` includes phenomena other than file-cache hits. Logical
weight bytes minus process OS reads is also not a valid cache-hit measurement.
`rusage.wired_bytes` does not establish the total GPU driver's pinned allocation.
Global IORegistry GPU memory must not be assigned to the native model PID.

For an external engine, `task_vm.compressed_bytes` and `device_bytes` remain
unknown. A native hook can report its own `task_info(TASK_VM_INFO)` alongside
operations; this observer does not use `task_for_pid` or silently substitute
Python's self-memory for the engine. Internal/external/reusable/physical footprint
fields have different accounting definitions and must not be blindly summed.

## Coherence and identity

Each snapshot records UTC and a monotonic start/end bracket. `sample_owned()`
brackets sequential system/process/optional-device reads without suspending the
process; it explicitly does not claim an atomic snapshot. `proc_pid_rusage`
identity is checked before/after task inspection and against the observer's
initial process start-abstime. Delta helpers reject PID reuse, reboot identity
changes, cumulative counter decreases/wrap, unavailable sources and overlapping
or reversed snapshot intervals. Gauge decreases are valid and retained.

```python
from research.glm53_gpu_v3.system_memory import SystemMemoryObserver, compare_system
from research.glm53_gpu_v3.process_io import ProcessIOObserver, sample_owned, compare_process
from research.glm53_gpu_v2.storage_counters import block_snapshot

system = SystemMemoryObserver()
owned = ProcessIOObserver(authorized_native_pid)
sample = sample_owned(owned, system, block_device_observer=block_snapshot)
```

No command-line strings, process inventories or unrelated PIDs are recorded.
The optional block callback retains its global device-counter scope.

## Prior guard events: what they establish

`evidence/prior-guard-context.json` extracts two prior recorded events, with
source hashes. The CPU A stage stopped during load before any request at RSS
6,494,486,528 bytes, OS footprint 379,031,568 bytes, global pressure1 and global
swap growth46,399,488 bytes. The GPU B event stopped at RSS4,022,665,216 bytes,
footprint607,472,288 bytes, global pressure2 and swap growth46,530,560 bytes.

These are correct system-wide guard rejections. They do not show that the model
has only the stated footprint, that its private memory caused the swap increase,
or that GPU pinning caused the second event. Clean file mappings can contribute
to RSS with different footprint accounting, while other applications and global
compressor/wired state compete for RAM. The old captures did not measure
per-engine compression or GPU pinning; those fields remain unknown. No user
applications were terminated to improve this observer's results.

## Partial dense-cache admission

`partial_dense_cache_admission()` is a conservative experimental preflight:

1. At least two valid observations over a five-second quiet interval; pressure
   normal throughout and no global swap growth, including intermediate samples.
2. Charge **2 × new packed host-cache bytes + in-flight bytes + known device
   increment** until a measured direct-buffer path disproves duplicated source
   residency. Factor1 is explicit and needs that evidence.
3. Add this increment to a measured baseline peak RSS and retain the existing
   cap, at most8.5GiB. The function refuses a larger cap.
4. Preserve at least2GiB estimated reclaimable system memory by default. The
   estimate is not an allocation guarantee; live guards remain required.

Begin with256MiB then512MiB only if the current observed baseline/preflight
permits them. For the historical6.866GB peak and8.5GiB cap, approximately2.261GB
RSS allowance remains: a1GiB copy cache consumes approximately2.147GB before
in-flight workspace and driver overhead. It cannot be admitted simply because
the model files are mapped or because private-footprint accounting is smaller.
Do not admit the complete9.44GB dense set on this basis. A preflight pass is not
a durable-stability or performance qualification.

## Tools and smoke checks

The actual self-only probes in `evidence/observer-tool-permissions.json` show:

- `fs_usage -t1 -f diskio <observer_pid>` is installed but returns "must be run as
  root". No sudo was requested and no trace was collected.
- One-second `sample` and `vmmap -summary` succeed for the observer PID. This
  does not establish permission for a differently signed engine, and these
  diagnostics can perturb execution. They are optional and absent from timed
  samples.

`evidence/observer-self-smoke-final.json` records working live self libprocv4,
process fault counters, self `TASK_VM_INFO`, and initial/final global snapshots.
It does not run a model, GPU kernel or deliberate weight IO. Process/SDK ctypes
layouts are checked against the installed public SDK prefixes. Run portable
fixtures with:

```sh
python -m unittest research.glm53_gpu_v3.test_system_memory \
  research.glm53_gpu_v3.test_process_io -v
```

For a new read-only self/owned-PID observation, use an unused output path:

```sh
python -m research.glm53_gpu_v3.process_io \
  --owned-pid <authorized_native_pid> --output /absolute/path/to/new-observation.json
```
