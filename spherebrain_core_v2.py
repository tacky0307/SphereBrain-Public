from __future__ import annotations

"""SphereBrain Unified Experience Core v2.0 — public research implementation.

This file contains only mechanisms promoted by the current research audit:
structural long-term memory, bounded working state, online directed transition trace,
self-organizing relational identity with fast/slow continuity, relation-conditioned
value memory, maturity-gated plasticity, failure-driven replasticity, and persistence.

It intentionally contains no task-answer lookup table and does not claim held-out
generalization, planning, prediction, or general intelligence.
"""

from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import hashlib
import json
import numpy as np

CORE_VERSION = "2.0-public"


@dataclass
class SignalResult:
    source_nodes: list[int]
    activated_nodes: list[int]
    traversed_edges: list[tuple[int, int]]
    activation_history: list[list[int]]
    final_activation: np.ndarray


@dataclass
class RelationalCluster:
    prototype: np.ndarray
    slow_prototype: np.ndarray
    value_field: np.ndarray
    count: int = 1
    continuity_hits: int = 0


@dataclass
class ReliabilityState:
    success_ema: float = 0.0
    stability_ema: float = 0.0
    volatility_ema: float = 1.0
    maturity_score: float = 0.0
    previous_score: float = 0.0
    previous_outcome: float = 0.0
    observations: int = 0
    mature: bool = False


class RelationalMemory:
    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.clusters: list[RelationalCluster] = []
        self.value_gain = 0.12
        self.value_retention = 0.997
        self.value_clip = 8.0
        self.assign_cosine = 0.86
        self.continuity_cosine = 0.72
        self.new_cluster_cosine = 0.58
        self.prototype_ema = 0.10
        self.slow_prototype_ema = 0.02
        self.membership_power = 6.0
        self.birth_count = 0
        self.direct_assignment_count = 0
        self.continuity_assignment_count = 0

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        x = np.asarray(v, dtype=float).reshape(-1)
        n = float(np.linalg.norm(x))
        return x / n if n > 0.0 else x

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _best(self, sig: np.ndarray) -> tuple[int | None, float, float]:
        if not self.clusters:
            return None, -1.0, -1.0
        scored = []
        for c in self.clusters:
            fast = self.cosine(sig, c.prototype)
            slow = self.cosine(sig, c.slow_prototype)
            scored.append((max(fast, slow), fast, slow))
        idx = int(np.argmax([x[0] for x in scored]))
        _, fast, slow = scored[idx]
        return idx, float(fast), float(slow)

    def _birth(self, sig: np.ndarray) -> int:
        self.clusters.append(RelationalCluster(sig.copy(), sig.copy(), np.zeros(self.size)))
        self.birth_count += 1
        return len(self.clusters) - 1

    def _assign(self, sig: np.ndarray) -> int:
        idx, fast, slow = self._best(sig)
        if idx is None:
            return self._birth(sig)
        identity = max(fast, slow)
        if identity < self.new_cluster_cosine:
            return self._birth(sig)
        continuity = fast < self.assign_cosine
        c = self.clusters[idx]
        fast_alpha = self.prototype_ema * (1.5 if continuity else 1.0)
        c.prototype = self.normalize((1-fast_alpha)*c.prototype + fast_alpha*sig)
        c.slow_prototype = self.normalize((1-self.slow_prototype_ema)*c.slow_prototype + self.slow_prototype_ema*sig)
        c.count += 1
        if continuity:
            c.continuity_hits += 1
            self.continuity_assignment_count += 1
        else:
            self.direct_assignment_count += 1
        return idx

    def membership(self, signature: np.ndarray) -> np.ndarray:
        sig = self.normalize(signature)
        if not self.clusters:
            return np.zeros(0)
        sims = np.asarray([max(0.0, max(self.cosine(sig,c.prototype), self.cosine(sig,c.slow_prototype))) for c in self.clusters])
        if not np.any(sims > 0):
            out = np.zeros(len(self.clusters)); out[0] = 1.0; return out
        weights = np.power(sims, self.membership_power)
        return weights / float(np.sum(weights))

    def score(self, signature: np.ndarray) -> float:
        sig = self.normalize(signature)
        weights = self.membership(sig)
        return float(sum(w*np.dot(c.value_field, sig) for w,c in zip(weights,self.clusters))) if weights.size else 0.0

    def experience(self, signature: np.ndarray, outcome: float) -> None:
        sig = self.normalize(signature)
        self._assign(sig)
        weights = self.membership(sig)
        for c in self.clusters:
            c.value_field *= self.value_retention
        for w,c in zip(weights,self.clusters):
            c.value_field += self.value_gain * float(outcome) * float(w) * sig
            np.clip(c.value_field, -self.value_clip, self.value_clip, out=c.value_field)

    def state_dict(self) -> dict:
        return {"birth_count":self.birth_count,"direct_assignment_count":self.direct_assignment_count,"continuity_assignment_count":self.continuity_assignment_count,"clusters":[{"prototype":c.prototype.tolist(),"slow_prototype":c.slow_prototype.tolist(),"value_field":c.value_field.tolist(),"count":c.count,"continuity_hits":c.continuity_hits} for c in self.clusters]}

    def load_state_dict(self, data: dict) -> None:
        self.birth_count=int(data.get("birth_count",0)); self.direct_assignment_count=int(data.get("direct_assignment_count",0)); self.continuity_assignment_count=int(data.get("continuity_assignment_count",0))
        self.clusters=[RelationalCluster(np.asarray(x["prototype"]),np.asarray(x.get("slow_prototype",x["prototype"])),np.asarray(x["value_field"]),int(x.get("count",1)),int(x.get("continuity_hits",0))) for x in data.get("clusters",[])]


class SphereBrainCore:
    core_version = CORE_VERSION

    def __init__(self, node_count: int = 160, neighbors_per_node: int = 7, seed: int = 42, learning_rate: float = 0.07, decay_rate: float = 0.0008, short_term_capacity: int = 24) -> None:
        self.node_count=node_count; self.neighbors_per_node=neighbors_per_node; self.seed=seed; self.learning_rate=learning_rate; self.decay_rate=decay_rate; self.short_term_capacity=int(short_term_capacity)
        self.rng=np.random.default_rng(seed)
        directions=self.rng.normal(size=(node_count,3)); directions/=np.linalg.norm(directions,axis=1,keepdims=True); radii=self.rng.random(node_count)**(1/3); self.positions=directions*radii[:,None]
        diff=self.positions[:,None,:]-self.positions[None,:,:]; dist=np.linalg.norm(diff,axis=2); np.fill_diagonal(dist,np.inf)
        self.adjacency=np.zeros((node_count,node_count),dtype=bool); self.weights=np.zeros((node_count,node_count)); self.usage=np.zeros((node_count,node_count),dtype=int); self.node_usage=np.zeros(node_count,dtype=int)
        for i in range(node_count):
            for j in np.argsort(dist[i])[:neighbors_per_node]:
                a,b=sorted((i,int(j))); self.adjacency[a,b]=self.adjacency[b,a]=True
                if self.weights[a,b]==0:
                    w=min(.92,.22+.42*np.exp(-2*dist[a,b])+float(self.rng.uniform(0,.07))); self.weights[a,b]=self.weights[b,a]=w
        self.short_term=deque(maxlen=self.short_term_capacity); self.transition_trace=np.zeros(node_count*node_count); self.relational_memory=RelationalMemory(node_count*node_count); self.reliability:list[ReliabilityState]=[]; self.experience_count=0

    def text_to_sources(self,text:str,count:int=2)->list[int]:
        if count < 0 or count > self.node_count:
            raise ValueError(f"count must be between 0 and node_count ({self.node_count})")
        digest=hashlib.sha256(text.encode()).digest(); out=[]; p=0
        while len(out)<count:
            if p+4>len(digest): digest=hashlib.sha256(digest).digest(); p=0
            n=int.from_bytes(digest[p:p+4],"big")%self.node_count; p+=4
            if n not in out: out.append(n)
        return out

    def propagate(self,source_nodes:Iterable[int],*,context_nodes:Iterable[int]|None=None,steps:int=12,threshold:float=.18,learn:bool=False)->SignalResult:
        activation=np.zeros(self.node_count); sources=list(source_nodes)
        for k,n in enumerate(sources): activation[n]=max(activation[n],1-.08*k)
        if context_nodes:
            for n in context_nodes: activation[n]=max(activation[n],.34)
        activated=set(np.flatnonzero(activation>0).tolist()); history=[sorted(activated)]; traversed=set()
        for _ in range(steps):
            candidates={}
            for src in np.flatnonzero(activation>0):
                neigh=np.flatnonzero(self.adjacency[src]); scores=activation[src]*self.weights[src,neigh]
                for li in np.argsort(scores)[-min(2,len(neigh)):]:
                    dst=int(neigh[li]); val=float(scores[li])*.78
                    if val>=threshold and (dst not in candidates or val>candidates[dst][0]): candidates[dst]=(val,int(src))
            if not candidates: break
            nxt=np.zeros(self.node_count)
            for dst,(val,src) in sorted(candidates.items(),key=lambda x:x[1][0],reverse=True)[:72]:
                nxt[dst]=max(nxt[dst],val); traversed.add(tuple(sorted((src,dst))))
            now=np.flatnonzero(nxt>0).tolist()
            if not now: break
            activated.update(now); history.append(now); activation=nxt
            if len(activated)>=100: break
        if learn and traversed:
            self.weights[self.adjacency]*=1-self.decay_rate
            for a,b in traversed:
                w=self.weights[a,b]+self.learning_rate*(1-self.weights[a,b]); self.weights[a,b]=self.weights[b,a]=w; self.usage[a,b]+=1; self.usage[b,a]+=1
            for n in activated: self.node_usage[n]+=1
        return SignalResult(sources,sorted(activated),sorted(traversed),history,activation.copy())

    def transition_signature(self,signal:SignalResult)->np.ndarray:
        sig=np.zeros((self.node_count,self.node_count))
        for before,after in zip(signal.activation_history[:-1],signal.activation_history[1:]):
            if not before or not after: continue
            scale=1/max(1,len(before)*len(after))
            for a in before:
                for b in after:
                    if a!=b: sig[a,b]+=scale
        return RelationalMemory.normalize(sig.reshape(-1))

    def _compose(self,sig:np.ndarray)->np.ndarray:
        t=RelationalMemory.normalize(self.transition_trace)
        return RelationalMemory.normalize(sig+.20*t) if np.any(t) else RelationalMemory.normalize(sig)

    def _sync_reliability(self):
        while len(self.reliability)<len(self.relational_memory.clusters): self.reliability.append(ReliabilityState())

    def score_stimulation(self,source_nodes:Iterable[int],*,context_nodes:Iterable[int]|None=None)->float:
        sig=self.transition_signature(self.propagate(source_nodes,context_nodes=context_nodes,learn=False)); return self.relational_memory.score(self._compose(sig))

    def experience(self,source_nodes:Iterable[int],outcome:float,*,context_nodes:Iterable[int]|None=None)->dict:
        signal=self.propagate(source_nodes,context_nodes=context_nodes,learn=True); sig=self.transition_signature(signal)
        self.transition_trace=.94*self.transition_trace+.10*sig; self.short_term.append({"sources":signal.source_nodes,"signature":sig.tolist(),"outcome":float(outcome)})
        composed=self._compose(sig); pre=self.relational_memory.membership(composed); self._sync_reliability(); mature_weight=sum(float(w) for w,s in zip(pre,self.reliability) if s.mature) if outcome>0 else 0.0; scale=1-mature_weight*.92
        self.relational_memory.experience(composed,float(outcome)*scale); self._sync_reliability(); mem=self.relational_memory.membership(composed)
        for i,w in enumerate(mem):
            if w<=0: continue
            s=self.reliability[i]; alpha=min(1,.12*float(w)*max(1,len(mem))); positive=1.0 if outcome>0 else 0.0; change=abs(float(outcome)-s.previous_outcome); stable=1-min(1,change)
            s.success_ema=(1-alpha)*s.success_ema+alpha*positive; s.stability_ema=(1-alpha)*s.stability_ema+alpha*stable; s.volatility_ema=(1-alpha)*s.volatility_ema+alpha*change; s.previous_score=s.maturity_score; s.maturity_score=s.success_ema*s.stability_ema/(1+max(0,s.volatility_ema)); s.observations+=1
            if outcome<=0: s.mature=False
            elif s.observations>=12: s.mature=bool(s.maturity_score>=s.previous_score and s.maturity_score>=.72)
            s.previous_outcome=float(outcome)
        self.experience_count+=1
        return {"cluster_count":len(self.relational_memory.clusters),"plasticity_scale":float(scale),"mature_clusters":[s.mature for s in self.reliability]}

    def save(self,path:str|Path)->None:
        data={"core_version":self.core_version,"node_count":self.node_count,"neighbors_per_node":self.neighbors_per_node,"seed":self.seed,"learning_rate":self.learning_rate,"decay_rate":self.decay_rate,"short_term_capacity":self.short_term_capacity,"positions":self.positions.tolist(),"adjacency":self.adjacency.astype(int).tolist(),"weights":self.weights.tolist(),"usage":self.usage.tolist(),"node_usage":self.node_usage.tolist(),"short_term":list(self.short_term),"transition_trace":self.transition_trace.tolist(),"reliability":[asdict(x) for x in self.reliability],"experience_count":self.experience_count,"relational_memory":self.relational_memory.state_dict()}; Path(path).write_text(json.dumps(data),encoding="utf-8")

    @classmethod
    def load(cls,path:str|Path)->"SphereBrainCore":
        d=json.loads(Path(path).read_text()); capacity=int(d.get("short_term_capacity",24)); c=cls(d["node_count"],d["neighbors_per_node"],d["seed"],d["learning_rate"],d["decay_rate"],capacity); c.positions=np.asarray(d["positions"]); c.adjacency=np.asarray(d["adjacency"],dtype=bool); c.weights=np.asarray(d["weights"]); c.usage=np.asarray(d["usage"],dtype=int); c.node_usage=np.asarray(d["node_usage"],dtype=int); c.short_term=deque(d["short_term"],maxlen=capacity); c.transition_trace=np.asarray(d["transition_trace"]); c.reliability=[ReliabilityState(**x) for x in d["reliability"]]; c.experience_count=int(d["experience_count"]); c.relational_memory.load_state_dict(d["relational_memory"]); return c
