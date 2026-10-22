"""
prisme_sim.py — modèle de coût « octets par token accepté » (OTA) pour l'exécution exacte
d'un MoE géant hors-cœur (GLM-5.3, Qwen3.8) sur Mac 16 Go + SSD.

Toutes les grandeurs sont en octets et secondes. Les paramètres mesurés sont chargés depuis
des fichiers JSON quand ils existent ; sinon des valeurs par défaut explicitement marquées
« hypothèse » sont utilisées. Le modèle est volontairement simple et vérifiable :

  B_pass = rho_stream * (A_nr + s * sum_l E|U_l(S) \ C_l|) + draft_bytes
  T_pass = max(B_pass / beta_eff, F_pass / Phi, B_pass_raw / delta) + overhead
  OTA    = E[B_pass] / E[tau] ;  temps/token = E[T_pass] / E[tau]   (renouvellement-récompense)
"""
from __future__ import annotations
import json, math, dataclasses as dc
from typing import Sequence, Optional
import numpy as np

GB = 1e9

# ---------------------------------------------------------------------------------------
# Spécifications
# ---------------------------------------------------------------------------------------
@dc.dataclass
class ModelSpec:
    name: str
    L_moe: int                 # couches MoE
    N: int                     # experts routés par couche
    k: int                     # experts sélectionnés par token
    expert_bytes: float        # s : octets stockés d'un expert (avec échelles)
    always_bytes: float        # A : octets toujours actifs par passe (format stocké)
    total_bytes: float         # taille du checkpoint (format stocké)
    active_params: float       # paramètres actifs par token (pour le calcul)
    mtp_extra_bytes: float     # octets non routés de la couche MTP (attention, eh_proj, partagé)
    rho_routed: float = 1.0    # taux sans perte (compressé/original) des experts routés
    rho_always: float = 1.0    # taux sans perte des poids toujours actifs
    kv_bytes_per_token: float = 0.0

    @property
    def routed_per_token(self):
        return self.L_moe * self.k * self.expert_bytes

    @property
    def full_sweep_bytes(self):
        return self.always_bytes + self.L_moe * self.N * self.expert_bytes


@dc.dataclass
class HardwareSpec:
    name: str
    ram_weights: float               # M : RAM disponible pour les poids résidents
    devices: Sequence[float]         # débits soutenus des périphériques (octets/s)
    phi: float = 4e12                # FLOP/s effectifs (GPU) — hypothèse tant que Metal n'est pas mesuré
    delta: float = 1e12              # débit de décodage sans perte (octets décompressés/s)
    overhead_per_pass: float = 0.0   # bulles/synchronisations non recouvertes (s)

    @property
    def beta_eff(self):
        return float(sum(self.devices))


@dc.dataclass
class RoutingSpec:
    """nu[j] : nouveauté marginale de la j-ième position d'une fenêtre (nu[0]=1).
    union_frac(n) : fraction d'experts touchés par n tokens consécutifs (prefill).
    union_frac_streams(B) : fraction touchée par B tokens de flux indépendants."""
    nu: Sequence[float]
    prefill_frac: Optional[Sequence[float]] = None   # index n-1 -> fraction (n = 1..len)
    streams_frac: Optional[Sequence[float]] = None
    source: str = "hypothèse"

    @staticmethod
    def uniform(N, k, jmax=64, nmax=4096):
        q = 1 - k / N
        nu = [q ** j for j in range(jmax + 1)]
        frac = [1 - q ** n for n in range(1, nmax + 1)]
        return RoutingSpec(nu=nu, prefill_frac=frac, streams_frac=frac, source="uniforme indépendant (pire cas)")

    def union_over_k(self, n):
        """E|U(n)|/k pour n positions consécutives (n>=1)."""
        n = int(n)
        if n <= len(self.nu):
            return float(sum(self.nu[:n]))
        # au-delà : prolonger par la courbe de prefill si disponible
        if self.prefill_frac is not None and n <= len(self.prefill_frac):
            return None
        return float(sum(self.nu)) + (n - len(self.nu)) * self.nu[-1]


@dc.dataclass
class DraftSpec:
    name: str
    alphas: Sequence[float]          # alpha_j : acceptation conditionnelle du j-ième brouillon
    bytes_per_draft_token: float     # octets lus pour proposer un token (ex. experts MTP)

    def pi(self, K):
        p, out = 1.0, [1.0]
        for j in range(K):
            a = self.alphas[min(j, len(self.alphas) - 1)]
            p *= a
            out.append(p)
        return out


# ---------------------------------------------------------------------------------------
# Passes étroites (décodage simple ou spéculation linéaire)
# ---------------------------------------------------------------------------------------
def resident_always(m: ModelSpec, hw: HardwareSpec, compressed_residency=False):
    cap = hw.ram_weights / (m.rho_always if compressed_residency else 1.0)
    return min(m.always_bytes, cap)


def thin_pass(m: ModelSpec, hw: HardwareSpec, r: RoutingSpec, K: int, draft: Optional[DraftSpec],
              lossless=True, residency=True, expert_hit=0.0, compressed_residency=False):
    """Retourne (octets/passe, tokens/passe, secondes/passe) pour une passe à K brouillons."""
    A_res = resident_always(m, hw, compressed_residency) if residency else 0.0
    A_nr = m.always_bytes - A_res
    rho_a = m.rho_always if lossless else 1.0
    rho_r = m.rho_routed if lossless else 1.0
    union_k = sum(r.nu[: K + 1])
    routed = m.L_moe * m.k * m.expert_bytes * union_k * (1 - expert_hit)
    d = 0.0
    if K > 0 and draft is not None:
        d = K * draft.bytes_per_draft_token * rho_r
    B = rho_a * A_nr + rho_r * routed + d
    raw = A_nr + routed + (K * draft.bytes_per_draft_token if (K > 0 and draft) else 0.0)
    tau = sum(draft.pi(K)) if (K > 0 and draft is not None) else 1.0
    F = 2.0 * m.active_params * (K + 1)
    T = max(B / hw.beta_eff, F / hw.phi, (raw / hw.delta) if lossless else 0.0) + hw.overhead_per_pass
    return B, tau, T


def best_depth(m, hw, r, draft, Kmax=8, **kw):
    rows = []
    for K in range(0, Kmax + 1):
        B, tau, T = thin_pass(m, hw, r, K, draft if K > 0 else None, **kw)
        rows.append(dict(K=K, bytes_per_pass=B, tau=tau, ota=B / tau, s_per_token=T / tau))
    best = min(rows, key=lambda x: x["s_per_token"])
    return best, rows


def mediant_bound(r: RoutingSpec, draft: DraftSpec, Kmax=8):
    pis = draft.pi(Kmax)
    return max(pis[j] / r.nu[j] for j in range(Kmax + 1))


# ---------------------------------------------------------------------------------------
# Prefill, balayage spéculatif, multi-flux
# ---------------------------------------------------------------------------------------
def prefill_seconds(m, hw, r: RoutingSpec, P: int, lossless=True, residency=True):
    A_res = resident_always(m, hw) if residency else 0.0
    rho_a = m.rho_always if lossless else 1.0
    rho_r = m.rho_routed if lossless else 1.0
    if r.prefill_frac is not None and P <= len(r.prefill_frac):
        frac = r.prefill_frac[P - 1]
    else:
        frac = 1 - (1 - m.k / m.N) ** P
    routed = m.L_moe * m.N * frac * m.expert_bytes
    B = rho_a * (m.always_bytes - A_res) + rho_r * routed
    F = 2.0 * m.active_params * P
    return max(B / hw.beta_eff, F / hw.phi), B


def sweep_breakeven_tau(m, hw, r, draft, lossless=True):
    """tau minimal par balayage complet pour battre la meilleure passe étroite (Théorème 4)."""
    best, _ = best_depth(m, hw, r, draft, lossless=lossless)
    rho_a = m.rho_always if lossless else 1.0
    rho_r = m.rho_routed if lossless else 1.0
    A_res = resident_always(m, hw)
    C = rho_a * (m.always_bytes - A_res) + rho_r * m.L_moe * m.N * m.expert_bytes
    T_sweep = C / hw.beta_eff
    n_max = T_sweep * hw.phi / (2.0 * m.active_params)
    return dict(T_sweep_s=T_sweep, n_max=n_max, tau_breakeven=T_sweep / best["s_per_token"],
                thin_s_per_token=best["s_per_token"])


def multistream(m, hw, r: RoutingSpec, B: int, lossless=True):
    A_res = resident_always(m, hw)
    rho_a = m.rho_always if lossless else 1.0
    rho_r = m.rho_routed if lossless else 1.0
    if r.streams_frac is not None and B <= len(r.streams_frac):
        frac = r.streams_frac[B - 1]
    else:
        frac = 1 - (1 - m.k / m.N) ** B
    routed = m.L_moe * m.N * frac * m.expert_bytes
    Bp = rho_a * (m.always_bytes - A_res) + rho_r * routed
    T = max(Bp / hw.beta_eff, 2.0 * m.active_params * B / hw.phi)
    return dict(B=B, s_per_pass=T, tokens_per_s=B / T, latency_s_per_token_per_stream=T)
