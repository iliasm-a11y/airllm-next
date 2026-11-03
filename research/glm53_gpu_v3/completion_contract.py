"""Offline validation of completed native server requests and process shutdown."""
from __future__ import annotations


def record_server_exit(report, returncode):
    normal = type(returncode) is int and returncode == 0
    report['server_shutdown'] = {'normal_exit': normal, 'exit_code': returncode,
        'scope': 'owned llama-server exited0 after requested graceful shutdown; no forced-kill acceptance'}
    if not normal:
        report['status'] = 'failed'
        report.setdefault('error', 'Owned server did not exit normally: ' + str(returncode))
    return normal


def validate_completion(request, requested):
    """Pinned /completion return_tokens stream includes ordinary EOS token IDs.

    Native process_token sends a partial containing tkn.tok before setting its
    EOS stop flag. A partial may be suppressed by incomplete UTF8, so a missing
    ID cannot safely be assumed to be EOS alone. Reject incomplete transcripts
    instead of accepting an unconditional predicted-1 allowance.
    """
    request = request if isinstance(request, dict) else {}
    body, final = request.get('body'), request.get('final')
    body = body if isinstance(body, dict) else {}
    final = final if isinstance(final, dict) else {}
    tokens = request.get('tokens')
    valid_ids = isinstance(tokens, list) and bool(tokens) and all(type(t) is int and t >= 0 for t in tokens)
    count = len(tokens) if isinstance(tokens, list) else None
    predicted, stop_type = final.get('tokens_predicted'), final.get('stop_type')
    reasons = []
    if type(requested) is not int or requested < 1 or type(body.get('n_predict')) is not int or body['n_predict'] != requested:
        reasons.append('request token budget differs from planned positive count')
    if final.get('stop') is not True or final.get('truncated') is True:
        reasons.append('terminal completion missing or context truncated')
    if not valid_ids:
        reasons.append('generated ID transcript missing or invalid')
    if type(predicted) is not int or predicted < 1 or type(requested) is not int or predicted > requested:
        reasons.append('final predicted count missing or outside request budget')
    if not valid_ids or count != predicted:
        reasons.append('SSE ID transcript is incomplete or differs from final predicted count')
    if stop_type == 'limit':
        if predicted != requested:
            reasons.append('limit stop must produce exactly the planned token count')
    elif stop_type != 'eos':
        reasons.append('only planned limit or natural EOS stop is supported')
    timings = final.get('timings')
    timings = timings if isinstance(timings, dict) else {}
    if 'predicted_n' in timings and (type(timings['predicted_n']) is not int or timings['predicted_n'] != predicted):
        reasons.append('server final timing predicted count differs')
    if 'events' in request:
        events = request['events']
        try:
            flattened = [t for e in events for t in e['response'].get('tokens', [])]
        except (KeyError, TypeError, AttributeError):
            flattened = None
        if flattened != tokens:
            reasons.append('SSE event token IDs differ from recorded generated transcript')
    return {'passed': not reasons, 'reasons': reasons, 'requested_tokens': requested,
        'generated_ids': count, 'final_tokens_predicted': predicted, 'stop_type': stop_type,
        'scope': 'terminal request/count/complete emitted-ID agreement; not new logits/state fidelity',
        'eos_id_omission_accepted': False}
