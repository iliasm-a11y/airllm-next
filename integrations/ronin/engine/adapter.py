"""RONIN contract for the pinned GLM GGUF native backend.

The logical session table does not allocate KV state or duplicate the model.
One NativeGlmModel owns the physical scheduler and the native process.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from airllm.nextgen.glm_native import NativeGlmError, NativeGlmModel, NativeLimits

MODEL_ID = "ggml-org/GLM-5.3-Flash-GGUF"
MODEL_REVISION = "119034bc3d9d76eb41f91508e6f9fb19fde42672"


class EngineError(RuntimeError):
    def __init__(self, code, message, *, action=None, details=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.action, self.details = action, details or {}

    def as_dict(self):
        return {"code": self.code, "message": self.message,
                "action": self.action, "details": self.details}


class _CombinedCancellation:
    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event is not None and event.is_set() for event in self.events)


class Engine:
    def __init__(self, *, config=None, model_factory=NativeGlmModel):
        self.config = dict(config or {})
        self._cpu_fallback = self.config.pop("fallback_to_cpu_on_rejection", True)
        if type(self._cpu_fallback) is not bool:
            raise ValueError("fallback_to_cpu_on_rejection must be an explicit boolean")
        self._backend_selection = None
        self._model_factory = model_factory
        self.model_path = None
        self._model = None
        self._lock = threading.RLock()
        self._sessions = {}
        self._shutdown_cancel = threading.Event()
        self._counters = {"load_attempts": 0, "generation_attempts": 0,
                          "generated_tokens": 0, "requests_in_flight": 0,
                          "inference_wall_seconds": 0.0}

    def _options(self):
        options = dict(self.config)
        names = {"runtime_binary": "RONIN_GLM_RUNTIME",
                 "verification_file": "RONIN_GLM_VERIFICATION",
                 "state_dir": "RONIN_GLM_STATE_DIR"}
        for key, env in names.items():
            if key not in options and os.environ.get(env):
                options[key] = os.environ[env]
        for key, env in (("context_tokens", "RONIN_GLM_CONTEXT"),
                         ("threads", "RONIN_GLM_THREADS"),
                         ("gpu_layers", "RONIN_GLM_GPU_LAYERS"),
                         ("moe_cache_mib", "RONIN_GLM_MOE_CACHE_MIB")):
            if key not in options and os.environ.get(env):
                options[key] = int(os.environ[env])
        if "cpu_moe" not in options and os.environ.get("RONIN_GLM_CPU_MOE"):
            options["cpu_moe"] = os.environ["RONIN_GLM_CPU_MOE"] == "1"
        if "mmap_demand" not in options and os.environ.get("RONIN_GLM_MMAP_DEMAND"):
            options["mmap_demand"] = os.environ["RONIN_GLM_MMAP_DEMAND"] == "1"
        if "expert_prefetch" not in options and os.environ.get("RONIN_GLM_EXPERT_PREFETCH"):
            options["expert_prefetch"] = os.environ["RONIN_GLM_EXPERT_PREFETCH"] == "1"
        if "dense_prefetch" not in options and os.environ.get("RONIN_GLM_DENSE_PREFETCH"):
            options["dense_prefetch"] = os.environ["RONIN_GLM_DENSE_PREFETCH"] == "1"
        if isinstance(options.get("limits"), dict):
            options["limits"] = NativeLimits(**options["limits"])
        return options

    def _backend_options(self):
        """Reject an unqualified GPU request before constructing any backend.

        An explicit research opt-in still reaches the experimental backend.
        Resource/integrity errors and failures after activation are never retried:
        replaying an already-started request could duplicate a logical decision.
        """
        options = self._options()
        gpu = options.get("gpu_layers", 0)
        allowed = options.get("allow_unqualified_gpu", False)
        if type(allowed) is not bool:
            raise ValueError("allow_unqualified_gpu must be an explicit boolean")
        rejected = type(gpu) is int and gpu > 0 and not allowed
        self._backend_selection = {
            "requested_gpu_layers": gpu,
            "fallback_before_activation": bool(rejected and self._cpu_fallback),
            "rejection": "GPU_FIDELITY_UNQUALIFIED" if rejected else None,
            "experimental_opt_in": allowed,
        }
        if rejected and self._cpu_fallback:
            options.update(gpu_layers=0, cpu_moe=False, moe_cache_mib=0,
                           allow_unqualified_gpu=False,
                           threads=min(10, os.cpu_count() or 1))
        self._backend_selection["selected_gpu_layers"] = options.get("gpu_layers", 0)
        return options

    def _select(self, model_path):
        path = str(Path(model_path).expanduser().resolve())
        if self._model is not None and path != self.model_path:
            if self._counters["requests_in_flight"]:
                raise EngineError("ENGINE_BUSY", "Attendre la fin de la génération avant de changer de modèle.")
            self._model.close()
            self._model = None
        if self._model is None:
            self._model = self._model_factory(path, **self._backend_options())
            self.model_path = path
        return self._model

    def status(self, model_path=None):
        selected = model_path if model_path is not None else self.model_path
        backend = {}
        blockers = []
        try:
            with self._lock:
                if selected:
                    backend = self._select(selected).status()
                    for message in backend.get("checkpoint", {}).get("errors", []):
                        blockers.append({"code": "CHECKPOINT_UNAVAILABLE", "message": message})
                    if backend.get("failure"):
                        blockers.append({"code": "ENGINE_RESOURCE_STOP", "message": str(backend["failure"])})
                else:
                    blockers.append({"code": "MODEL_NOT_SELECTED", "message": "Sélectionner le dossier GLM-5.3-Flash Q2_K_S local."})
        except Exception as exc:
            blockers.append({"code": getattr(exc, "code", "CHECKPOINT_UNAVAILABLE"), "message": str(exc)})
        loaded = bool(backend.get("loaded", backend.get("running", False)))
        ready = bool(backend.get("ready", loaded)) and not blockers
        return {"state": "loaded" if loaded else "selected" if selected and not blockers else "unavailable",
                "ready": ready, "loaded": loaded,
                "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "quantization": "Q2_K_S"},
                "engine": {"id": "AirLLM Next / llama.cpp native", "shared": True,
                           "backend_selection": self._backend_selection},
                "capabilities": {"load": not blockers, "generation": ready,
                                 "streaming": ready, "cancellation": True,
                                 "tool_calls": False, "multimodal": False,
                                 "reasoning_effort": True, "reasoning_effort_scope": "chat_template_conditioning_only",
                                 "reasoning_effort_values": ["low", "high", "max"],
                                 "reasoning_effort_default": "GGUF template default; no override",
                                 "session_metadata": True,
                                 "session_kv_persistence": False,
                                 "max_concurrent_inferences": 1, "logical_agents": 600},
                "checkpoint": {"path": selected}, "blockers": blockers,
                "native": backend, "counters": self.counters()}

    def load(self, model_path, cancel=None):
        if self._shutdown_cancel.is_set():
            raise EngineError("ENGINE_CLOSED", "Le moteur est en cours de fermeture.")
        if not model_path:
            raise EngineError("MODEL_NOT_SELECTED", "Sélectionner le dossier GLM local.")
        with self._lock:
            self._counters["load_attempts"] += 1
            try:
                model=self._select(model_path)
                model.load(cancel=_CombinedCancellation(cancel, self._shutdown_cancel))
            except NativeGlmError as error:
                raise EngineError(error.code, str(error), details=error.details) from error
        return self.status()

    def unload(self):
        with self._lock:
            if self._counters["requests_in_flight"]:
                raise EngineError("ENGINE_BUSY", "Annuler la génération avant de décharger le moteur.")
            if self._model is not None:
                self._model.close()
            return {"loaded": False}

    def request_shutdown(self):
        self._shutdown_cancel.set()

    def close(self):
        self.request_shutdown()
        with self._lock:
            if self._model is not None:
                self._model.close()

    def create_session(self, mission_id):
        if not isinstance(mission_id, str) or not mission_id.strip():
            raise EngineError("INVALID_MISSION", "Une session doit appartenir à une mission.")
        session = {"id": uuid.uuid4().hex, "mission_id": mission_id,
                   "created_at": time.time(), "state": "metadata_only"}
        with self._lock:
            self._sessions[session["id"]] = session
        return dict(session)

    def sessions(self, mission_id):
        with self._lock:
            return [dict(s) for s in self._sessions.values() if s["mission_id"] == mission_id]

    def close_session(self, session_id, mission_id):
        with self._lock:
            if self._sessions.get(session_id, {}).get("mission_id") != mission_id:
                raise EngineError("SESSION_NOT_FOUND", "Session absente pour cette mission.")
            del self._sessions[session_id]

    def counters(self):
        with self._lock:
            result=dict(self._counters)
            scheduler=getattr(self._model,'scheduler',None)
            result['physical_inferences_active']=scheduler.status()['active'] if scheduler is not None else None
            return result

    def stream(self, messages, max_new_tokens=128, cancel=None, reasoning_effort=None, session_id=None):
        if self._shutdown_cancel.is_set():
            raise EngineError("ENGINE_CLOSED", "Le moteur est en cours de fermeture.")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 163_840:
            raise EngineError("INVALID_TOKEN_BUDGET", "Budget de génération entier attendu entre 1 et 163840 tokens.")
        if reasoning_effort is not None and reasoning_effort not in ("low", "high", "max"):
            raise EngineError("INVALID_REASONING_EFFORT", "Effort explicite attendu : low, high ou max.")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str) for m in messages
        ):
            raise EngineError("INVALID_MESSAGES", "Une liste de messages textuels structurés est nécessaire.")
        if cancel is not None and not callable(getattr(cancel, "is_set", None)):
            raise EngineError("INVALID_CANCELLATION", "Un événement d’annulation est attendu.")
        if cancel is not None and cancel.is_set():
            raise EngineError("CANCELLED", "Génération annulée avant démarrage.")
        with self._lock:
            if session_id is not None and session_id not in self._sessions:
                raise EngineError("SESSION_NOT_FOUND", "Session inconnue.")
            if self._model is None:
                raise EngineError("ENGINE_NOT_LOADED", "Charger le modèle GLM sélectionné avant de générer.")
            model = self._model
        return self._iterate(model, messages, max_new_tokens, cancel, reasoning_effort)

    def _iterate(self, model, messages, budget, cancel, reasoning_effort):
        started = time.monotonic()
        with self._lock:
            if model is not self._model:
                raise EngineError("ENGINE_MODEL_CHANGED", "Le modèle sélectionné a changé avant le démarrage de cette requête.")
            self._counters["generation_attempts"] += 1
            self._counters["requests_in_flight"] += 1
        generated = 0
        try:
            options = {"max_new_tokens": budget, "cancel": _CombinedCancellation(cancel, self._shutdown_cancel)}
            if reasoning_effort is not None:
                options["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
            for event in model.stream_chat(messages, **options):
                tokens = event.get("tokens", [])
                if isinstance(tokens, list):
                    generated += len(tokens)
                yield event
        except NativeGlmError as error:
            raise EngineError(error.code, str(error), details=error.details) from error
        finally:
            with self._lock:
                self._counters["requests_in_flight"] -= 1
                self._counters["generated_tokens"] += generated
                self._counters["inference_wall_seconds"] += time.monotonic() - started
