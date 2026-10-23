"""Loopback asynchronous job API for the owner-thread AgentEngine.

HTTP parsing, tokenization, state changes and forward steps all run on the
same thread that constructs the local model. The server yields to HTTP polling
between token quanta; it never constructs a second model or worker MLX stream.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import ipaddress
import json
from .agent_engine import AgentEngine, _JOB_ID


def make_agent_server(model, *, state_dir, model_name='airllm-next', host='127.0.0.1',
                      port=8080, max_new_tokens=512, max_body_bytes=1024*1024,
                      quantum_tokens=8, max_requests=64,
                      max_transcript_bytes=32*1024**2, max_journal_bytes=32*1024**2,
                      max_journal_write_bytes=256*1024**2):
    """Construct on the model-owning thread, then call serve_forever there.

    state_dir is explicit because durable request transcripts may contain
    private prompts. One native context is resident at a time. Jobs are greedy
    token/prompt requests; tools, stochastic sampling and streaming are rejected.
    """
    if host == 'localhost':
        host = '127.0.0.1'
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError('Use a literal loopback address or localhost') from error
    if not address.is_loopback:
        raise ValueError('The agent API may bind only to a loopback address')
    if address.version != 4:
        raise ValueError('The agent API currently requires IPv4 loopback')
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('Invalid port')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('Invalid output token limit')
    if type(max_body_bytes) is not int or max_body_bytes < 1:
        raise ValueError('Invalid request body limit')
    if not isinstance(model_name, str) or not model_name:
        raise ValueError('Invalid local model name')
    try:
        model_name.encode('utf-8')
    except UnicodeError as error:
        raise ValueError('Invalid local model name') from error
    if state_dir is None:
        raise ValueError('A local state_dir is required for durable agent jobs')
    engine = AgentEngine(model, state_dir=state_dir, quantum_tokens=quantum_tokens,
                         max_requests=max_requests, max_transcript_bytes=max_transcript_bytes,
                         max_journal_bytes=max_journal_bytes,
                         max_journal_write_bytes=max_journal_write_bytes)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        server_version = 'AirLLMNextAgents/1'
        sys_version = ''

        def setup(self):
            self.request.settimeout(5)
            super().setup()

        def log_message(self, *args):
            return

        def respond(self, status, value):
            raw = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(raw)

        def error(self, status, code):
            self.respond(status, {'error': {'code': code}})

        def parts(self):
            parts = self.path.rstrip('/').split('/')
            if parts[:4] != ['', 'v1', 'agent', 'jobs']:
                raise KeyError('Unknown API path')
            if len(parts) > 4 and not _JOB_ID.fullmatch(parts[4]):
                raise KeyError('Unknown job')
            return parts[4:]

        def do_GET(self):
            try:
                if self.path == '/health':
                    self.respond(200, engine.status())
                    return
                if self.path == '/v1/models':
                    self.respond(200, {'object': 'list', 'data': [
                        {'id': model_name, 'object': 'model', 'owned_by': 'local', 'created': 0}]})
                    return
                parts = self.parts()
                if not parts:
                    self.respond(200, {'data': [engine.snapshot(job_id) for job_id in engine.jobs]})
                elif len(parts) == 1:
                    result = engine.snapshot(parts[0])
                    tokenizer = getattr(model, 'tokenizer', None)
                    if tokenizer is not None:
                        result['output_text'] = tokenizer.decode(result['output_ids'], skip_special_tokens=True)
                    self.respond(200, result)
                else:
                    raise KeyError('Unknown API path')
            except KeyError:
                self.error(404, 'not_found')
            except Exception:
                self.error(500, 'local_backend_error')

        def read_json(self):
            if self.headers.get('Transfer-Encoding') is not None:
                raise ValueError('unsupported_transfer_encoding')
            lengths = self.headers.get_all('Content-Length', [])
            if len(lengths) != 1:
                raise ValueError('content_length_required')
            size = int(lengths[0])
            if not 1 <= size <= max_body_bytes:
                raise MemoryError('request_body_limit')
            if self.headers.get('Content-Type', '').split(';')[0].lower() != 'application/json':
                raise ValueError('json_required')
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise ValueError('truncated_body')
            def reject_constant(value):
                raise ValueError('nonfinite_json')
            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError('duplicate_json_field')
                    value[key] = item
                return value
            payload = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique)
            if not isinstance(payload, dict):
                raise ValueError('object_required')
            return payload

        def do_POST(self):
            try:
                parts = self.parts()
                payload = self.read_json()
                if not parts:
                    allowed = {'model', 'input_ids', 'prompt', 'max_tokens', 'priority', 'agent_id', 'eos_token_id'}
                    if set(payload)-allowed or payload.get('model') != model_name:
                        raise ValueError('unsupported_request')
                    if ('input_ids' in payload) == ('prompt' in payload):
                        raise ValueError('one_prompt_required')
                    if 'input_ids' in payload:
                        ids = payload['input_ids']
                    else:
                        prompt = payload['prompt']
                        tokenizer = getattr(model, 'tokenizer', None)
                        if not isinstance(prompt, str) or not prompt or tokenizer is None:
                            raise ValueError('local_tokenizer_required')
                        ids = tokenizer.encode(prompt)
                    count = payload.get('max_tokens', min(32, max_new_tokens))
                    if type(count) is not int or not 1 <= count <= max_new_tokens:
                        raise ValueError('output_token_limit')
                    job_id = engine.submit(ids, max_tokens=count,
                        priority=payload.get('priority', 0), agent_id=payload.get('agent_id', 'default'),
                        eos_token_id=payload.get('eos_token_id'))
                    self.respond(202, engine.snapshot(job_id))
                elif len(parts) == 2 and parts[1] in ('pause', 'cancel', 'resume'):
                    if payload:
                        raise ValueError('empty_action_body_required')
                    result = getattr(engine, parts[1])(parts[0])
                    self.respond(200, result)
                else:
                    raise KeyError('Unknown API path')
            except KeyError:
                self.error(404, 'not_found')
            except MemoryError:
                self.error(429, 'local_budget_exhausted')
            except (ValueError, TypeError, UnicodeError, RecursionError):
                self.error(400, 'invalid_request')
            except (TimeoutError, OSError):
                self.error(503, 'local_io_unavailable')
            except Exception:
                self.error(500, 'local_backend_error')

        def do_DELETE(self):
            try:
                parts = self.parts()
                if len(parts) != 1:
                    raise KeyError('Unknown API path')
                engine.forget(parts[0])
                self.respond(200, {'id': parts[0], 'deleted': True})
            except KeyError:
                self.error(404, 'not_found')
            except ValueError:
                self.error(409, 'pause_or_cancel_before_delete')
            except OSError:
                self.error(503, 'local_io_unavailable')
            except Exception:
                self.error(500, 'local_backend_error')

    class AgentServer(HTTPServer):
        def service_actions(self):
            engine.step()

        def serve_forever(self, poll_interval=.01):
            engine._check_owner()
            return super().serve_forever(poll_interval=poll_interval)

        def server_close(self):
            try:
                super().server_close()
            finally:
                engine.close()

    try:
        server = AgentServer((host, port), Handler)
    except BaseException:
        engine.close()
        raise
    server.engine = engine
    return server


def serve_agents(model, **kwargs):
    server = make_agent_server(model, **kwargs)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
