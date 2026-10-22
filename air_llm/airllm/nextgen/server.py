"""Small, loopback-only OpenAI-compatible text API over an already local model.

This implements the non-streaming, single-choice subset. Unsupported sampling,
tool and multimodal options fail explicitly rather than silently changing them.
No HTTP client, model download or remote inference path exists in this module.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections.abc import Mapping
import ipaddress
import json
import math
import socket
from time import time
from uuid import uuid4


class RequestError(ValueError):
    pass


def make_server(model, *, model_name='airllm-next', host='127.0.0.1', port=8080,
                max_body_bytes=1024*1024, max_new_tokens=512):
    """Return a single-request-at-a-time server; caller owns model lifetime.

    Serialization bounds concurrent inference and queued request memory. Requests
    have a 10-second socket timeout; the model's own lock still protects callers
    using it outside this server. Construct the MLX model and call serve() on
    the same thread: MLX stream handles are thread-affine. Port zero is useful
    for local integration tests.
    """
    if host == 'localhost':
        host = '127.0.0.1'
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError('Use a literal loopback address or localhost') from error
    if not address.is_loopback:
        raise ValueError('The inference API may bind only to a loopback address')
    if not isinstance(model_name, str) or not model_name:
        raise ValueError('model_name must be a nonempty string')
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('Invalid port')
    if type(max_body_bytes) is not int or max_body_bytes < 1:
        raise ValueError('Invalid request body limit')
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError('Invalid generation limit')
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError('A local tokenizer is required to serve text requests')
    context_limit = min(model.options.max_context, model.config['max_position_embeddings'])
    vocab_size=model.config.get('vocab_size')

    def prepare(payload, chat):
        if not isinstance(payload, dict):
            raise RequestError('The JSON body must be an object')
        supported = {'model', 'max_tokens', 'temperature', 'stream', 'n',
                     'messages' if chat else 'prompt'}
        unknown = set(payload)-supported
        if unknown:
            raise RequestError('Unsupported request fields: '+', '.join(sorted(unknown)))
        if payload.get('model') != model_name:
            raise RequestError('Unknown model; inspect /v1/models for the local model ID')
        if type(payload.get('stream', False)) is not bool or payload.get('stream', False):
            raise RequestError('Only stream=false is supported')
        if type(payload.get('n', 1)) is not int or payload.get('n', 1) != 1:
            raise RequestError('Only n=1 is supported')
        count = payload.get('max_tokens', min(32, max_new_tokens))
        if type(count) is not int or not 1 <= count <= max_new_tokens:
            raise RequestError(f'max_tokens must be an integer from 1 to {max_new_tokens}')
        temperature = payload.get('temperature', 0.)
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature < 0:
            raise RequestError('temperature must be finite and nonnegative')
        if temperature!=0 and not getattr(model,'supports_sampling',True):
            raise RequestError('The selected local backend supports greedy temperature=0 only')
        if chat:
            messages = payload.get('messages')
            if not isinstance(messages, list) or not 1 <= len(messages) <= 1024:
                raise RequestError('messages must be a nonempty list of at most 1024 messages')
            for message in messages:
                if (not isinstance(message, dict) or set(message) != {'role', 'content'}
                        or message['role'] not in ('system', 'user', 'assistant')
                        or not isinstance(message['content'], str)):
                    raise RequestError('Each message must contain a supported role and text content')
            if not getattr(tokenizer, 'chat_template', None):
                raise RequestError('The local tokenizer has no chat template; use /v1/completions')
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                return_dict=False)
        else:
            prompt = payload.get('prompt')
            if not isinstance(prompt, str) or not prompt:
                raise RequestError('prompt must be a nonempty string')
            ids = tokenizer.encode(prompt)
        attention_mask=None
        if isinstance(ids,Mapping):
            # Transformers versions and local tokenizer wrappers may return a
            # BatchEncoding despite a request for IDs. Preserve only the one
            # unpadded sequence this runtime can execute.
            if 'input_ids' not in ids:
                raise RequestError('The tokenizer mapping must include input_ids')
            attention_mask=ids.get('attention_mask')
            ids=ids['input_ids']
        if hasattr(ids, 'tolist'):
            ids = ids.tolist()
        if not isinstance(ids, list) or not ids or any(type(value) is not int or value < 0 or value>2**31-1 for value in ids):
            raise RequestError('The tokenizer must return a nonempty sequence of token IDs')
        if attention_mask is not None:
            if hasattr(attention_mask,'tolist'):attention_mask=attention_mask.tolist()
            if (not isinstance(attention_mask,list) or len(attention_mask)!=len(ids)
                    or any(type(value) not in (int,bool) or value!=1 for value in attention_mask)):
                raise RequestError('Only an unpadded sequence with an all-ones attention mask is supported')
        if vocab_size is not None and any(value>=vocab_size for value in ids):
            raise RequestError('The tokenizer returned IDs outside the local model vocabulary')
        if len(ids)+count > context_limit:
            raise RequestError('Prompt plus requested completion exceeds the configured context limit')
        return ids, count, float(temperature)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'
        server_version = 'AirLLMNextLocal/1'
        sys_version = ''

        def setup(self):
            self.request.settimeout(10)
            super().setup()

        def log_message(self, format, *args):
            # Prompts and response bodies are never written to an access log.
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

        def error(self, status, message, kind='invalid_request_error'):
            self.respond(status, {'error': {'message': message, 'type': kind,
                                          'param': None, 'code': None}})

        def do_GET(self):
            if self.path != '/v1/models':
                self.error(404, 'Unknown API path')
                return
            self.respond(200, {'object': 'list', 'data': [
                {'id': model_name, 'object': 'model', 'created': 0, 'owned_by': 'local'}]})

        def do_POST(self):
            if self.path not in ('/v1/chat/completions', '/v1/completions'):
                self.error(404, 'Unknown API path')
                return
            if self.headers.get('Transfer-Encoding') is not None:
                self.error(400, 'Transfer-Encoding is unsupported; send Content-Length')
                return
            if len(self.headers.get_all('Content-Length', [])) != 1:
                self.error(411, 'Exactly one Content-Length header is required')
                return
            try:
                length = int(self.headers['Content-Length'])
            except (ValueError, TypeError):
                self.error(400, 'Invalid Content-Length')
                return
            if length < 1 or length > max_body_bytes:
                self.error(413, 'Request body exceeds the configured limit or is empty')
                return
            content_type = self.headers.get('Content-Type', '').split(';')[0].strip().lower()
            if content_type != 'application/json':
                self.error(415, 'Content-Type must be application/json')
                return
            try:
                body = self.rfile.read(length)
                if len(body) != length:
                    raise RequestError('Truncated request body')
                def reject_constant(value):
                    raise RequestError('Nonfinite JSON numbers are invalid')
                def unique_fields(pairs):
                    value={}
                    for key,item in pairs:
                        if key in value:raise RequestError('Duplicate JSON object fields are invalid')
                        value[key]=item
                    return value
                payload = json.loads(body, parse_constant=reject_constant,object_pairs_hook=unique_fields)
                chat = self.path == '/v1/chat/completions'
                ids, count, temperature = prepare(payload, chat)
            except (RequestError, UnicodeError, json.JSONDecodeError, ValueError, TypeError, RecursionError) as error:
                # RequestError messages are generated here and contain no local paths.
                self.error(400, str(error) if isinstance(error, RequestError) else 'Invalid request or tokenizer input')
                return
            except (TimeoutError, OSError):
                self.error(408, 'Request body timed out')
                return
            except Exception:
                self.error(500, 'The local tokenizer failed', 'server_error')
                return
            try:
                generated = model.generate([ids], max_new_tokens=count, temperature=temperature)
                sequences = generated.tolist() if hasattr(generated,'tolist') else generated
                if (not isinstance(sequences,list) or len(sequences)!=1 or not isinstance(sequences[0],list)
                        or sequences[0][:len(ids)]!=ids or not len(ids)<len(sequences[0])<=len(ids)+count
                        or any(type(value) is not int or value<0 or value>2**31-1 for value in sequences[0])):
                    raise RuntimeError('Invalid local backend generation result')
                result = sequences[0][len(ids):]
                if vocab_size is not None and any(value>=vocab_size for value in result):
                    raise RuntimeError('Local backend generated IDs outside its vocabulary')
                text = tokenizer.decode(result, skip_special_tokens=True)
                if not isinstance(text,str):raise RuntimeError('Invalid local tokenizer response')
                eos = model.config.get('eos_token_id')
                eos = eos if isinstance(eos, (list, tuple)) else [eos]
                reason = 'stop' if len(result)<count or result and result[-1] in eos else 'length'
                choice = {'index': 0, 'finish_reason': reason, 'logprobs': None}
                if chat:
                    choice['message'] = {'role': 'assistant', 'content': text}
                else:
                    choice['text'] = text
                self.respond(200, {'id': ('chatcmpl-' if chat else 'cmpl-')+uuid4().hex,
                    'object': 'chat.completion' if chat else 'text_completion',
                    'created': int(time()), 'model': model_name, 'choices': [choice],
                    'usage': {'prompt_tokens': len(ids), 'completion_tokens': len(result),
                              'total_tokens': len(ids)+len(result)}})
            except MemoryError:
                self.error(503, 'The local model reached its memory budget; reduce context or generation length',
                           'server_error')
            except Exception:
                self.error(500, 'Local inference failed; inspect the model directly for diagnostics', 'server_error')

        def do_OPTIONS(self):
            self.error(405, 'Browser cross-origin access is not supported')

        def do_PUT(self):
            self.error(405, 'Unsupported method')

        do_DELETE = do_PUT
        do_PATCH = do_PUT

    class LocalServer(HTTPServer):
        address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
        request_queue_size = 8

    return LocalServer((host, port), Handler)


def serve(model, **kwargs):
    """Run until interrupted; closing the HTTP server does not close the model."""
    with make_server(model, **kwargs) as server:
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
