"""Aligned snapshot boundaries derived from complete native token IDs."""

BATCHES = (8, 16, 32, 64, 128)


def choose_boundary(prompts, microbatch, requested=None):
    if type(microbatch) is not int or microbatch not in BATCHES:
        raise ValueError("Unsupported snapshot microbatch")
    if not prompts or any(not isinstance(p, list) or not p or
                          any(type(t) is not int or t < 0 for t in p) for p in prompts):
        raise ValueError("Complete native token IDs are required")
    common = 0
    for values in zip(*prompts):
        if any(value != values[0] for value in values):
            break
        common += 1
    # Every arm must compute a nonempty private suffix before sampling.
    common = min(common, min(map(len, prompts)) - 1)
    if requested is not None:
        if type(requested) is not int or not 0 < requested <= common or requested % microbatch:
            raise ValueError("Requested boundary is outside the aligned common prefix")
        return requested
    boundary = common // microbatch * microbatch
    if boundary <= 0:
        raise ValueError("No aligned common prefix with a private suffix")
    return boundary


def report_boundary(report):
    # Old receipts describe the original fixed boundary only.
    boundary = report.get("prefix_boundary", 208)
    if type(boundary) is not int or boundary <= 0:
        raise ValueError("Invalid declared prefix boundary")
    if "prefix_boundary" in report and (
            report.get("boundary_policy") != "native-lcp-aligned-v1" or
            type(report.get("microbatch")) is not int or
            report["microbatch"] not in BATCHES or boundary % report["microbatch"]):
        raise ValueError("Declared prefix boundary must be aligned")
    return boundary
