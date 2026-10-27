#!/usr/bin/env python3
"""Conditional logical-transport roofline from caller-declared measurements.

This tool measures nothing and supplies no bandwidth default. Logical bytes
and effective rate must share one accounting domain. The result is a scenario,
never certification of physical SSD traffic or an actual engine speedup.
"""
import argparse
import json
import math


def number(value, name, *, minimum=0., strict=False):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and not (value <= minimum if strict else value < minimum))
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f'{name} must be finite and {"greater than" if strict else "at least"} {minimum}')
    return float(value)


def estimate_limit(*, logical_bytes_per_token, effective_transport_bytes_per_second,
                   measurement_label, additional_cache_coverage=0.,
                   accepted_tokens_per_block=1., target_pass_equivalents_per_block=1.,
                   draft_logical_bytes_per_block=0., draft_seconds_per_block=0.,
                   compute_seconds_per_block=None):
    """Model observed ordinary traffic plus explicitly declared scenario changes.

    ``logical_bytes_per_token`` is measured ordinary-token transport traffic;
    baseline cache effects are already included. ``additional_cache_coverage``
    is an *additional* reduction of that measured traffic, not baseline cache
    coverage. Pass equivalents must include verification's larger expert union
    and any rollback/replay traffic. Tokens/block means actual verified final
    tokens emitted, including a verified correction when applicable. Neither
    extra expert traffic nor a free bonus token is inferred automatically.
    """
    logical = number(logical_bytes_per_token, 'logical_bytes_per_token')
    rate = number(effective_transport_bytes_per_second,
                  'effective_transport_bytes_per_second', strict=True)
    coverage = number(additional_cache_coverage, 'additional_cache_coverage')
    if coverage > 1:
        raise ValueError('additional_cache_coverage must be at most one')
    emitted = number(accepted_tokens_per_block, 'accepted_tokens_per_block', strict=True)
    passes = number(target_pass_equivalents_per_block,
                    'target_pass_equivalents_per_block', strict=True)
    draft_bytes = number(draft_logical_bytes_per_block, 'draft_logical_bytes_per_block')
    draft_seconds = number(draft_seconds_per_block, 'draft_seconds_per_block')
    compute = (None if compute_seconds_per_block is None else
               number(compute_seconds_per_block, 'compute_seconds_per_block'))
    if not isinstance(measurement_label, str) or not measurement_label.strip():
        raise ValueError('A nonempty measurement/provenance label is required')
    target_bytes = logical * (1 - coverage) * passes
    traffic = target_bytes + draft_bytes
    transport_seconds = traffic / rate
    # Host reads and target compute can overlap. Their sum is not presented as
    # a measured latency. Both scenarios below are explicit scheduling models.
    compute_and_draft = None if compute is None else compute + draft_seconds
    ideal_overlap = None if compute_and_draft is None else max(transport_seconds, compute_and_draft)
    serial = None if compute_and_draft is None else transport_seconds + compute_and_draft
    reciprocal = lambda seconds: None if seconds is None or seconds == 0 else emitted / seconds
    result = dict(schema_version=1, status='conditional_scenario_not_measured',
        measurement_label=measurement_label.strip(), caller_inputs_verified=False,
        physical_ssd_bandwidth_certified=False,
        inputs=dict(logical_bytes_per_token=logical,
            effective_transport_bytes_per_second=rate,
            additional_cache_coverage=coverage, accepted_tokens_per_block=emitted,
            target_pass_equivalents_per_block=passes,
            draft_logical_bytes_per_block=draft_bytes,
            draft_seconds_per_block=draft_seconds, compute_seconds_per_block=compute),
        assumptions=[
            'Rate and byte counts use the same logical transport domain, not inferred physical SSD traffic.',
            'Measured ordinary traffic already includes its baseline cache effects.',
            'Additional cache coverage applies only to the remaining measured target traffic.',
            'Pass equivalents include larger routed expert unions and all verification/replay traffic.',
            'Tokens per block are verified final emitted tokens; no unverified proposal or free bonus is counted.',
            'All draft traffic and time must be declared; zero is an explicit assumption, not a measurement.',
            'Ideal overlap and serial scheduling are scenarios, not measured GPU/read attribution.'],
        estimates=dict(target_logical_bytes_per_block=target_bytes,
            total_logical_bytes_per_block=traffic,
            logical_bytes_per_verified_final_token=traffic / emitted,
            transport_seconds_per_block=transport_seconds,
            conditional_transport_ceiling_tokens_per_second=reciprocal(transport_seconds),
            ideal_overlap_seconds_per_block=ideal_overlap,
            ideal_overlap_tokens_per_second=reciprocal(ideal_overlap),
            serial_seconds_per_block=serial,
            serial_tokens_per_second=reciprocal(serial)),
        limits=[
            'No hardware experiment, source verification or physical SSD measurement is performed.',
            'No full engine ceiling is established without measured compute, draft, replay and synchronization costs.',
            'A changed workload/cache/route pattern can invalidate the declared effective rate and traffic model.',
            'A null rate means that this declared domain provides no finite bound, not infinite engine performance.'])
    if any(value is not None and not math.isfinite(value) for value in result['estimates'].values()):
        raise ValueError('Declared scenario overflows finite arithmetic')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--logical-bytes-per-token', type=float, required=True)
    parser.add_argument('--effective-transport-bytes-per-second', type=float, required=True)
    parser.add_argument('--measurement-label', required=True)
    parser.add_argument('--additional-cache-coverage', type=float, default=0.)
    parser.add_argument('--accepted-tokens-per-block', type=float, default=1.)
    parser.add_argument('--target-pass-equivalents-per-block', type=float, default=1.)
    parser.add_argument('--draft-logical-bytes-per-block', type=float, default=0.)
    parser.add_argument('--draft-seconds-per-block', type=float, default=0.)
    parser.add_argument('--compute-seconds-per-block', type=float)
    args = parser.parse_args()
    try:
        result = estimate_limit(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
