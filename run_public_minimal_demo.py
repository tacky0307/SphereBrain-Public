from __future__ import annotations

"""Minimal public demo for SphereBrain Unified Experience Core v2.0.

Two states learn one action mapping, then the environment reverses the mapping.
The Core receives only stimuli, candidate actions, and scalar success/failure.
"""

import random
from spherebrain_core_v2 import SphereBrainCore

STATES=("S0","S1"); ACTIONS=("A0","A1")
ACQUISITION={"S0":"A0","S1":"A1"}; REVERSAL={"S0":"A1","S1":"A0"}


def sources(core,state): return core.text_to_sources(f"PUBLIC_STATE_{state}",2)
def action_node(core,action): return [core.text_to_sources(f"PUBLIC_ACTION_{action}",1)[0]]

def choose(core,state):
    scores={a:core.score_stimulation(sources(core,state),context_nodes=action_node(core,a)) for a in ACTIONS}
    return max(ACTIONS,key=lambda a:(scores[a],-ACTIONS.index(a))),scores

def accuracy(core,rule):
    return sum(choose(core,s)[0]==rule[s] for s in STATES)/len(STATES)

def train(core,rule,trials,seed):
    rng=random.Random(seed)
    for i in range(trials):
        state=STATES[i%2]
        action,_=choose(core,state)
        epsilon=.30+(i/max(1,trials-1))*(.04-.30)
        if rng.random()<epsilon: action=rng.choice(ACTIONS)
        core.experience(sources(core,state),1.0 if action==rule[state] else -1.0,context_nodes=action_node(core,action))


def main():
    core=SphereBrainCore(seed=20260822)
    print("initial",accuracy(core,ACQUISITION))
    train(core,ACQUISITION,160,1)
    print("after acquisition",accuracy(core,ACQUISITION))
    print("before reversal",accuracy(core,REVERSAL))
    train(core,REVERSAL,200,2)
    print("after reversal",accuracy(core,REVERSAL))
    print("old rule after reversal",accuracy(core,ACQUISITION))
    print("clusters",len(core.relational_memory.clusters))
    print("experiences",core.experience_count)

if __name__=="__main__": main()
