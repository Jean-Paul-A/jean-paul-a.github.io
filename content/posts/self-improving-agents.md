---
title: "Self-Improving Agents"
date: 2026-02-19
draft: false
math: true
tags: ["AI", "Agents", "Reinforcement Learning"]
---
# Agentic AI

## 1. Why Large Language Models Require AI Agent？

Large language models (LLMs) have achieved remarkable performance on a wide spectrum of language tasks. Yet, when deployed as autonomous problem-solvers for real-world goals,
they exhibit fundamental architectural limitations that preclude direct application. An LLM can be characterized as a **stateless conditional probability estimator**:

$$
\hat{y} = \arg\max_{y} P_\theta(y \mid x)
$$

where $x$ is the input token sequence, $y$ the output sequence, and $\theta$ the frozen parameters obtained at training time. Five structural properties follow directly from this formulation, each giving rise to a corresponding capability gap addressed by agentic frameworks:

1. **Closed-world knowledge**: All knowledge is encoded into $\theta$ at training time and remains immutable post-deployment. The model has no mechanism to query real-time state, update its beliefs from post-cutoff events, or distinguish stale internal knowledge from current ground truth. This is the root cause of the **Structural Gap** (§1).
2. **Statelessness**: Each inference call is independent; no information persists across calls beyond what is re-supplied in the prompt. The accessible context at step $t$ is strictly bounded by the window length $L$, i.e., $h_t = \{x_{t-L}, \ldots, x_t\}$, beyond which all prior observations are irrecoverably lost. This is the root cause of the **Temporal Gap** (§2).
3. **Epistemic opacity**: The model produces outputs without maintaining any explicit representation of uncertainty or evidential support. It cannot distinguish between well-grounded claims and confabulations, and cannot modulate its confidence based on the availability of verifiable evidence — a structural tendency toward *epistemic overconfidence*. This is the root cause of the **Epistemic Gap** (§3).
4. **Parametric rigidity**: The weight matrix $\theta$ is fixed after training. The model cannot incorporate post-deployment experience, correct systematic errors, or refine its strategies through interaction — any adaptation requires full retraining or fine-tuning at significant computational cost. This is the root cause of the **Self-Improvement Gap** (§4).
5. **Passivity and bounded computation**: The model cannot initiate actions, interact with external environments, or modify world state; it solely responds to prompts. Furthermore, each forward pass constitutes a fixed-depth computation $F_\theta: \mathcal{X} \to \mathcal{Y}$ of depth $d$, bounding the set of solvable problems to those expressible within a depth-$d$ circuit — precluding iterative, goal-directed
   reasoning over arbitrary horizons. This is the root cause of the **Computational Gap** (§5).

### 1.1 Structural Gap: From Closed-World Knowledge to Open-Loop Environment

The frozen nature of $\theta$ means the model cannot access real-time data, verify its own outputs against ground truth, or compute non-linguistic results reliably. Agents grant LLMs interfaces to external tools (search engines, code interpreters, APIs), enabling **dynamic knowledge grounding** and **self-verification** via environmental feedback [Yao et al., 2023; Ruan et al., 2025]. The overwhelming majority of practically valuable tasks admit a **closed-loop** structure, formalized as a **Perceive–Act–Observe–Revise (PAOR)** cycle:

$$
s_{t+1} = f(s_t, a_t), \quad a_t = \pi_\theta(s_t)
$$

where $s_t$ is the environment state, $a_t$ the agent action, and $f$ the world transition function. A standalone LLM only approximates the policy $\pi_\theta$ for a single step, but **cannot perceive** $s_t$ from the real world, **cannot execute** $a_t$, and **cannot receive** $s_{t+1}$ to revise its plan. It operates purely in an open-loop fashion, making it structurally insufficient for task execution [Sun et al., 2023; Firoozi et al., 2024].

### 1.2 Temporal Gap: From Stateless Snapshots to Persistent Memory

Because each LLM call is context-independent, multi-turn tasks requiring accumulated information fail without external memory. Agent frameworks address this by maintaining explicit state stores—episodic, semantic, and procedural memory—that persist across inference calls [Zhang et al., 2024; Park et al., 2023]. Formally, the information accessible to a standalone LLM at step $t$ is bounded by its context window of length $L$:

$$
h_t^{\text{LLM}} = \{x_{t-L},\, x_{t-L+1},\, \ldots,\, x_t\}
$$

Any observation outside this window is **irrecoverably discarded**.

An agent, by contrast, maintains an external memory store $\mathcal{M}_t$ with a structured retrieval function $\rho$:

$$
h_t^{\text{agent}} = \rho\!\left(\mathcal{M}_t,\; q_t\right),
\quad \text{where } |\mathcal{M}_t| \text{ is unbounded}
$$

This transforms a finite-horizon Markov process into one with **unbounded effective history**—the agent can retrieve any past observation at any future step, enabling long-term coherence and cross-session continuity unavailable to a stateless LLM [Huang et al., 2026; Zhang et al., 2024].

### 1.3 Epistemic Gap: From Overconfident Generation to Calibrated Grounding

LLMs assign no explicit uncertainty to their outputs and cannot distinguish what they know from what they hallucinate—a condition termed *epistemic overconfidence*. Formally, let $G_{i,t} \in \{0,1\}$ denote whether a reasoning step at turn $t$ is supported by retrieved evidence. Empirical analysis shows that standalone LLMs exhibit near-constant $\mathbb{E}[G \mid E=k]$ across evidence levels $k $meaning they reason with equal confidence regardless of evidential support [Do LLM Agents Know How to Ground..., 2025]:

$$
\mathbb{E}[G \mid E=0] \approx \mathbb{E}[G \mid E=2] \quad \text{(miscalibrated)}
$$

Agents address this by introducing **external grounding**—retrieving verifiable evidence before committing to a claim—and **recovery mechanisms** that revise search strategies upon failure, ensuring reasoning is anchored to verifiable world state rather than parametric priors [Shinn et al., 2023; Asai et al., 2024].

### 1.4 Self-Improvement Gap: From Frozen Parameters to Experiential Adaptation

LLM weights $\theta$ are fixed post-training; the model cannot learn from post-deployment experience without expensive retraining. Agent frameworks decouple **parametric knowledge** (stored in $\theta$) from **experiential knowledge** (stored in $\mathcal{M}$), enabling continuous adaptation through memory accumulation and strategy refinement:

$$
\mathcal{M}_{t+1} = \text{Update}\!\left(\mathcal{M}_t,\, (s_t, a_t, r_t)\right)
$$

where $r_t$ is the reward or feedback signal at step $t$. By accumulating successes, failures, and corrected reasoning traces in $\mathcal{M}$, an agent effectively performs **online learning without gradient updates**, improving task performance over time without modifying the underlying model [Huang et al., 2026; Shinn et al., 2023].

### 1.5 Computational Gap: From Bounded Inference to Unbounded Iterative Reasoning

LLM reasoning is bounded by a fixed computational depth determined by the forward-pass graph. A single inference call computes a deterministic function $F_\theta: \mathcal{X} \to \mathcal{Y}$ of fixed depth $d$, implying that the set of solvable problems is **bounded by the expressivity of a depth-$d$ circuit**. Complex tasks demanding long-horizon reasoning exceed this bound.
Agent loops—such as the ReAct, Thought–Action–Observation cycle—convert fixed-depth inference into an **iterative, feedback-driven process**:

$$
y_k = F_\theta\!\left(x,\, o_1, o_2, \ldots, o_{k-1}\right),
\quad k = 1, 2, \ldots, K
$$

where $o_k$ is the environmental observation after executing $y_k$, and $K$ is not fixed a priori. This recurrent structure enables **adaptive plan refinement** rather than one-shot generation, and has been shown to elevate the composite system to a higher computational complexity class [Yao et al., 2023; Sun et al., 2023; Kambhampati et al., 2026].

> An LLM is a **language-space mapping function** $f: \mathcal{X} \to \mathcal{Y}$. Meaningful real-world tasks, however, require a **goal-to-world-state-change mapping** $g: \mathcal{G} \to \Delta\mathcal{W}$. An agent constitutes the **computational intermediary** that bridges these two domains by embedding the LLM's linguistic reasoning capability within a perceive–act–remember–iterate execution loop, transforming a text predictor into a goal-directed system.

## 2. Formal Definition of an Agentic AI

Let an LLM agent $\mathcal{A}$ be a tuple:

$$
\mathcal{A} = \langle \mathcal{T}, \mathcal{E}, \mathcal{M}, \mathcal{P}, \mathcal{U} \rangle
$$

where each component maps directly to one stage of the agent's single-step data flow:

- $\mathcal{T}$ : perception and tool interface
- $\mathcal{E}$ : epistemic module (evidence grounding + uncertainty)
- $\mathcal{M}$ : persistent memory store
- $\mathcal{P}$ : planner / reasoning engine (backed by LLM)
- $\mathcal{U}$ : update mechanism (experiential adaptation)

A system qualifies as a well-formed agent if and only if all five components are present and form a closed recurrent loop. Any missing component leaves the corresponding structural gap unresolved.

---

### 2.1 Perception & Tool Interface $\mathcal{T}$

*(Input: $s_t$ → Output: $o_t$)*

The interface $\mathcal{T} = \langle \Omega, \Phi \rangle$ constitutes a **bidirectional grounding operator** between the symbolic reasoning space $\mathcal{O}$ of the LLM and the state space $\mathcal{S}$ of the external environment:

$$
\Omega: \mathcal{S} \to \mathcal{O}
\qquad \text{(world-to-language: afferent / grounding)}
$$

$$
\Phi: \mathcal{A} \times \mathcal{S} \to \mathcal{S}
\qquad \text{(language-to-world: efferent / actuation)}
$$

At each step $t$, $\mathcal{T}$ is invoked sequentially in two phases. In the **afferent phase**, $\Omega$ projects the current world state $s_t$ into the LLM's token space, yielding the raw observation:

$$
o_t = \Omega(s_t)
$$

where $s_t \in \mathcal{S}$ is the current environment state (e.g., browser DOM, file system, API response) and $o_t \in \mathcal{O}$ is its tokenized representation.
In the **efferent phase**, $\Phi$ applies action $a_t$—produced downstream by $\mathcal{P}$—to transition the environment:

$$
s_{t+1} = \Phi(a_t,\, s_t), \qquad
a_t \in \mathcal{A} = \{\texttt{search},\, \texttt{exec},\,
\texttt{api},\, \ldots\}
$$

$\mathcal{T}$ itself is **stationary**: its functional form is invariant across $t$, analogous to the observation function $Z$ and transition function $T$ in a POMDP. The time index $t$ belongs to the trajectory $\tau = (s_0, a_0, o_0, s_1, \ldots)$,
not to $\mathcal{T}$.

---

### 2.2 Epistemic Module $\mathcal{E}$

*(Input: $o_t$ → Output: $h_t$)*

Before raw observations enter the reasoning pipeline, the epistemic module gates them against verifiable evidence, converting the unvalidated percept $o_t$ into a **grounded context** $h_t$:

$$
h_t = \mathcal{E}(o_t, \mathcal{M}_t)
$$

where $h_t \in \mathcal{O}$ is the evidence-validated, uncertainty-annotated observation ready for memory retrieval. $\mathcal{E}$ implements two sub-functions:

$$
\mathcal{E} = \langle \text{Ground}, \text{Recover} \rangle
$$

- $\text{Ground}(p, \mathcal{M}_t)$: retrieves supporting evidence for proposition $p$ before asserting it. If $\text{Evidence}(p, \mathcal{M}_t) = \emptyset$, triggers an external retrieval call rather than generating from $\theta$ alone, ensuring $\mathcal{B}(p) \propto \text{Evidence}(p)$.
- $\text{Recover}(\tau_t)$: detects failure states in the current trajectory prefix $\tau_t$ and reformulates the retrieval or reasoning strategy before proceeding.

This is realized concretely by Self-RAG [Asai et al., 2024], which inserts reflection tokens to decide when retrieval is warranted, and Reflexion [Shinn et al., 2023], which uses verbal self-critique to revise failed reasoning traces before re-querying $\mathcal{M}$.

---

### 2.3 Memory System $\mathcal{M}$

*(Input: $h_t$ → Output: $h_t^+$)*

The memory system augments the grounded context $h_t$ with relevant long-term history, yielding the **augmented history** $h_t^+$ that the planner will condition on. Retrieval is performed by a learned or heuristic function $\rho$:

$$
h_t^+ = \rho(\mathcal{M}_t,\; q_t), \qquad
q_t = \text{QueryEmbed}(h_t)
$$

where $q_t$ is a query vector derived from the current grounded context $h_t$, $\mathcal{M}_t$ is the memory store at step $t$, and $h_t^+ \in \mathcal{O}$ is the final context passed to $\mathcal{P}$, enriched with retrieved episodes, facts, and procedures. Unlike
$h_t^{\text{LLM}} = \{x_{t-L}, \ldots, x_t\}$ which is bounded by context window $L$, the effective history $h_t^+$ satisfies $|h_t^+| \not \leq L$ because $|\mathcal{M}_t|$ is unbounded.

Following cognitive science, $\mathcal{M}$ is structured into three stores [Zhang et al., 2024]:

| Store                | Analogue      | Content                             | Horizon    |
| -------------------- | ------------- | ----------------------------------- | ---------- |
| **Episodic**   | Hippocampus   | Raw trajectories$(s_t, a_t, o_t)$ | Long-term  |
| **Semantic**   | Neocortex     | Distilled facts, entity states      | Persistent |
| **Procedural** | Basal ganglia | Successful action sequences         | Persistent |

Recent architectures use graph-structured memory to capture relational dependencies between stored observations, enabling multi-hop retrieval beyond flat vector search [Huang et al., 2026].

---

### 2.4 Planner / Reasoning Engine $\mathcal{P}$

*(Input: $h_t^+$ → Output: $a_t$)*

The planner is the decision-making core of the agent. Given a high-level goal $g$, the augmented history $h_t^+$, and the current grounded observation $h_t$, $\mathcal{P}$ produces the next action:

$$
a_t = \mathcal{P}(g,\; h_t^+)
$$

where $g$ is the user-specified goal, $h_t^+$ encodes the full available context (current percept + retrieved memory), and $a_t \in \mathcal{A}$ is the selected tool action. The number of steps $K$ is not fixed a priori but determined at runtime by a termination predicate:

$$
K = \min\left\{ k \;\middle|\;
\text{Halt}(g,\, o_k,\, r_k) = \texttt{true} \right\}
$$

where $r_k$ is the reward or success signal at step $k$. This unbounded $K$ elevates the composite system beyond any fixed-depth circuit class. Key planning strategies span a complexity spectrum [Huang et al., 2024]:

| Strategy                  | Structure                       | Best For                        |
| ------------------------- | ------------------------------- | ------------------------------- |
| **CoT / ReAct**     | Linear chain                    | Short-horizon tasks             |
| **Tree-of-Thought** | Branching tree with pruning     | Tasks needing exploration       |
| **MCTS-based**      | Monte Carlo rollout + value fn. | Tasks with sparse reward        |
| **Multi-agent**     | Parallel sub-agent graph        | Decomposable long-horizon tasks |

Note that reasoning-augmented LLMs (e.g., o1, DeepSeek-R1) extend computation **within** a single call to $\mathcal{P}$ (intra-step depth), while the agent loop extends computation **across** calls (inter-step breadth). These are complementary, not competing, mechanisms of computational scaling [DeepSeek-R1, 2025].

---

### 2.5 Update Mechanism $\mathcal{U}$

*(Input: $a_t$, $r_t$ → Output: $\mathcal{M}_{t+1}$)*

After action $a_t$ is executed and reward $r_t$ observed, the update mechanism writes post-hoc experience back into $\mathcal{M}$ without modifying $\theta$:

$$
\mathcal{M}_{t+1} = \mathcal{U}(\mathcal{M}_t,\;
(s_t, a_t, r_t, v_t))
$$

where $s_t$ is the pre-action state, $a_t$ the executed action, $r_t \in \mathbb{R}$ the scalar reward or binary success signal, and $v_t$ a verbal critique generated by an LLM-based evaluator [Li et al., 2024]. Three levels of update granularity exist:

1. **Trace-level**: store the full $(s_t, a_t, o_t)$ tuple for future episodic retrieval
2. **Summary-level**: compress the trajectory into a distilled semantic insight and write to semantic memory
3. **Policy-level**: update a soft skill library or few-shot example bank for procedural reuse

The evaluator that produces $r_t$ and $v_t$ can itself be an LLM instance, forming a **self-critique loop** that requires no human labeling [Wang et al., 2024].

---

### 2.6 Architectural Overview

The five components compose into a single recurrent data-flow graph. Within each step $t$, the flow is strictly sequential; across steps, the loop is recurrent and runs for $K$ iterations determined at runtime:

$$
\boxed{
\underbrace{s_t}_{\substack{\text{world state} \\ \in\,\mathcal{S}}}
\xrightarrow{\;\mathcal{T}\;}
\underbrace{o_t}_{\substack{\text{raw percept} \\ \in\,\mathcal{O}}}
\xrightarrow{\;\mathcal{E}\;}
\underbrace{h_t}_{\substack{\text{grounded} \\ \text{context}}}
\xrightarrow{\;\mathcal{M}\;}
\underbrace{h_t^+}_{\substack{\text{augmented} \\ \text{history}}}
\xrightarrow{\;\mathcal{P}\;}
\underbrace{a_t}_{\substack{\text{action} \\ \in\,\mathcal{A}}}
\xrightarrow{\;\mathcal{U}\;}
\underbrace{\mathcal{M}_{t+1}}_{\substack{\text{updated} \\ \text{memory}}}
}
$$

Each arrow denotes one component's transformation:

| Arrow                                               | Transformation                               | Type                                                                              |
| --------------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------- |
| $s_t \xrightarrow{\mathcal{T}} o_t$               | World state → token sequence                | $\Omega: \mathcal{S} \to \mathcal{O}$                                           |
| $o_t \xrightarrow{\mathcal{E}} h_t$               | Raw percept → evidence-grounded context     | $\mathcal{E}: \mathcal{O} \times \mathcal{M} \to \mathcal{O}$                   |
| $h_t \xrightarrow{\mathcal{M}} h_t^+$             | Grounded context → memory-augmented history | $\rho: \mathcal{M} \times \mathcal{O} \to \mathcal{O}$                          |
| $h_t^+ \xrightarrow{\mathcal{P}} a_t$             | Augmented history → action decision         | $\mathcal{P}: \mathcal{O} \times \mathcal{G} \to \mathcal{A}$                   |
| $a_t \xrightarrow{\mathcal{U}} \mathcal{M}_{t+1}$ | Executed action + reward → memory update    | $\mathcal{U}: \mathcal{M} \times \mathcal{A} \times \mathbb{R} \to \mathcal{M}$ |

The efferent phase of $\mathcal{T}$ then closes the loop: $a_t \xrightarrow{\Phi} s_{t+1}$, resetting the world state for the next iteration. This recurrent, environment-coupled structure is precisely what elevates the agent beyond the computational expressivity of any single LLM inference call.




# A Seven-Component, Non-Overlapping Architecture for LLM Agents

This document refines the agent formalization into **seven** non-overlapping components by making (i) **feedback/evaluation** explicit and (ii) **safety enforcement** explicit. This separation matches survey perspectives that treat feedback as a core module for agent self-optimization and adaptation [web:19], and aligns with approaches that enforce provable, runtime-constrained execution for LLM agents [web:163].

---

## 1. Notation (disambiguated)

To avoid the ambiguity where a capital **S** might refer to both *State* and *Safe*, we use distinct symbols:

- World state: $x_t \in \mathcal{X}$
- Raw observation: $o_t \in \mathcal{O}$
- Grounded context: $h_t \in \mathcal{O}$
- Memory-augmented history: $h_t^{+} \in \mathcal{O}$
- Proposed tool action: $a_t \in \mathcal{A}$
- Guarded (safe-filtered) tool action: $\bar{a}_t \in \mathcal{A}$
- Goal: $g \in \mathcal{G}$
- Scalar feedback / reward: $r_t \in \mathbb{R}$
- Optional textual critique: $c_t \in \mathcal{C}$
- Memory store (at time $t$): $\mathcal{M}_t \in \mathfrak{M}$
- Memory update (delta): $\Delta_t \in \mathfrak{D}$

The time index $t$ is a property of the **trajectory** $\tau$ produced at runtime, not of the component implementations.

---

## 2. Agent as a 7-tuple

We define an LLM agent system as:

$$
\mathcal{A}=\langle \mathcal{T},\mathcal{E},\mathcal{M},\mathcal{P},\mathcal{R},\mathcal{G},\mathcal{U}\rangle
$$

where:

- $\mathcal{T}$: perception & actuation interface (tools/environment I/O)
- $\mathcal{E}$: epistemic grounding module (evidence gating, recovery triggers)
- $\mathcal{M}$: persistent memory subsystem (storage + retrieval augmentation)
- $\mathcal{P}$: planner / reasoning engine (LLM-backed decision policy)
- $\mathcal{R}$: evaluator / reward model (produces $r_t, c_t$)
- $\mathcal{G}$: guard / safety monitor (filters or rejects actions)
- $\mathcal{U}$: updater (computes memory deltas $\Delta_t$)

By construction, each component corresponds to a distinct arrow in the execution graph (§4), ensuring **non-overlap of responsibilities**.

---

## 3. Component signatures (non-overlapping)

### 3.1 Environment interface $\mathcal{T}=\langle \Omega,\Phi\rangle$

$$
\Omega:\mathcal{X}\to\mathcal{O},
\qquad
\Phi:\mathcal{A}\times\mathcal{X}\to\mathcal{X}
$$

- $\Omega$ (*world-to-language*): converts the current world state $x_t$ (e.g., DOM, files, API payloads) into a tokenizable observation $o_t$.
- $\Phi$ (*language-to-world*): executes a tool-call action and transitions the world state.

Instantiation along a trajectory:

$$
o_t = \Omega(x_t), \qquad x_{t+1}=\Phi(\bar{a}_t, x_t)
$$

### 3.2 Epistemic grounding $\mathcal{E}$

$$
h_t = \mathcal{E}(o_t,\mathcal{M}_t)
$$

$\mathcal{E}$ transforms raw percepts $o_t$ into a **grounded** context $h_t$ by enforcing evidence requirements, and by triggering recovery behaviors (e.g., “retrieve more evidence”, “disambiguate conflicting sources”). This explicit epistemic gating is consistent with the emphasis on feedback mechanisms for self-correction in LLM-based agents [web:19].

### 3.3 Memory subsystem $\mathcal{M}$

Retrieval augmentation:

$$
h_t^{+} = \rho(\mathcal{M}_t, h_t)
$$

Memory update:

$$
\mathcal{M}_{t+1} = \mathcal{M}_t \oplus \Delta_t
$$

Here, $\rho$ is the retrieval operator (vector/graph/hybrid), and $\oplus$ applies the delta update $\Delta_t$. In this decomposition, $\mathcal{M}$ owns storage semantics, while $\mathcal{U}$ (below) owns the policy for producing updates.

### 3.4 Planner / reasoning engine $\mathcal{P}$

$$
a_t = \mathcal{P}(g, h_t^{+})
$$

$\mathcal{P}$ outputs a **proposed** tool-call $a_t$. It does not execute tools (that is $\Phi$), does not score outcomes (that is $\mathcal{R}$), and does not enforce safety (that is $\mathcal{G}$).

### 3.5 Safety guard $\mathcal{G}$

$$
\bar{a}_t = \mathcal{G}(g, h_t^{+}, a_t)
$$

$\mathcal{G}$ enforces constraints by filtering, rewriting, or rejecting proposed actions before execution. This explicit “action interception” layer is aligned with approaches that aim to prevent harmful or unintended operations by checking each step prior to execution [web:163].

### 3.6 Evaluator / reward model $\mathcal{R}$

$$
(r_t, c_t)=\mathcal{R}(g, x_t, \bar{a}_t, o_{t+1})
$$

$\mathcal{R}$ is the **only** component that produces scalar feedback $r_t$ (and optional critique $c_t$). Making $\mathcal{R}$ explicit cleanly answers “where does feedback come from?”, which is central in feedback-mechanism formulations of LLM agents [web:19].

### 3.7 Updater $\mathcal{U}$

$$
\Delta_t=\mathcal{U}(\mathcal{M}_t,\; x_t,\; \bar{a}_t,\; o_{t+1},\; r_t,\; c_t)
$$

$\mathcal{U}$ converts experience into a **memory delta** $\Delta_t$ (trace/summary/procedure updates). The actual persistence step is performed only by $\mathcal{M}$ via $\oplus$.

---

## 4. Non-overlapping execution graph (one step)

A single step from $t$ to $t+1$ is:

$$
\boxed{
\underbrace{x_t}_{\text{world state}}
\xrightarrow{\;\Omega\;}
\underbrace{o_t}_{\text{raw percept}}
\xrightarrow{\;\mathcal{E}\;}
\underbrace{h_t}_{\text{grounded context}}
\xrightarrow{\;\rho(\mathcal{M}_t,\cdot)\;}
\underbrace{h_t^{+}}_{\text{augmented history}}
\xrightarrow{\;\mathcal{P}\;}
\underbrace{a_t}_{\text{proposed action}}
\xrightarrow{\;\mathcal{G}\;}
\underbrace{\bar{a}_t}_{\text{guarded action}}
\xrightarrow{\;\Phi\;}
\underbrace{x_{t+1}}_{\text{next world}}
\xrightarrow{\;\Omega\;}
\underbrace{o_{t+1}}_{\text{next percept}}
\xrightarrow{\;\mathcal{R}\;}
\underbrace{(r_t,c_t)}_{\text{feedback}}
\xrightarrow{\;\mathcal{U}\;}
\underbrace{\Delta_t}_{\text{memory delta}}
\xrightarrow{\;\oplus\;}
\underbrace{\mathcal{M}_{t+1}}_{\text{updated memory}}
}
$$

Within-step flow is sequential; across steps the system is recurrent and may run for an unbounded number of iterations.

---

## 5. Termination predicate (decoupled from planning)

To avoid semantic coupling between the planner $\mathcal{P}$ and evaluator $\mathcal{R}$, define a stop predicate as an **observable** function of goal and feedback:

$$
\textsc{Stop}_t = \textsc{Stop}(g, o_{t+1}, r_t, c_t)\in\{0,1\}
$$

This makes termination a verifiable condition rather than an internal planner belief, which is particularly important in tasks with externally checkable success criteria [web:19][web:163].

(If desired, $\textsc{Stop}$ may be implemented as part of $\mathcal{R}$ or as a separate runtime controller; the key requirement is that it remains **explicit** and **observable**.)






## References

1. **Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y.** (2023). ReAct: Synergizing reasoning and acting in language models. *ICLR 2023*. arXiv:2210.03629.
2. **Sun, H., Zhuang, Y., Kong, L., Dai, B., & Zhang, C.** (2023). AdaPlanner: Adaptive planning from feedback with language models. *NeurIPS 2023*. arXiv:2305.16653.
3. **Firoozi, R., et al.** (2024). Grounding LLMs for robot task planning using closed-loop state feedback. arXiv:2402.08546.
4. **Zhang, Z., et al.** (2024). A survey on the memory mechanism of large language model based agents. arXiv:2404.13501.
5. **Huang, X., et al.** (2024). Understanding the planning of LLM agents: A survey. arXiv:2402.02716.
6. **Ruan, Y., et al.** (2025). Agentic reasoning and tool integration for LLMs via self-improving transformers(ARTIST). arXiv:2505.01441.
7. **Kambhampati, S., et al.** (2026). Computability of agentic systems. arXiv:2602.13222.
8. **Park, J. S., et al.** (2023). Generative agents: Interactive simulacra of human behavior. *UIST 2023*. arXiv:2304.03442.
9. **Wang, L., et al.** (2024). A survey on large language model based autonomous agents. *Frontiers of Computer Science*, 18(6). arXiv:2308.11432.
10. **Huang et al.** (2026). Graph-based agent memory: Taxonomy, technique and trends. arXiv:2602.05665.
11. **Shinn et al.** (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS 2023*. arXiv:2303.11366.
12. **Asai et al.** (2024). Self-RAG: Learning to retrieve, generate and reflect. *ICLR 2024*. arXiv:2310.11511.
13. **Li et al.** (2024). A review of prominent paradigms for LLM-based agents: Tool use, planning, and feedback learning. arXiv:2406.05804.

## Definitions

At Anthropic, all LLM-driven systems are categorized as  **agentic systems** , with a critical architectural distinction :

> *"Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."* — Schluntz & Zhang (2024)

The key discriminator is  **who holds the control flow** :

| Concept                  | Control Flow Owner         | Complexity                |
| ------------------------ | -------------------------- | ------------------------- |
| **Building Block** | N/A — atomic unit         | Single augmented LLM call |
| **Workflow**       | Engineer (predefined code) | Composable, predictable   |
| **Agent**          | LLM (dynamic, runtime)     | Open-ended, autonomous    |

---

## Building Block: The Augmented LLM

> *"The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain."*

Augmentations:

* **Retrieval** — external knowledge lookup at query time
* **Tools** — executable actions (APIs, code runners, file I/O)
* **Memory** — short-term (context window) and long-term (external store)

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) standardizes third-party tool integration at this layer .

---

## Workflow Patterns

Five composable patterns, ordered by control complexity :

**1. Prompt Chaining** — Sequential decomposition; each LLM call processes the previous output, with optional programmatic *gates* for intermediate validation. Best when subtasks are fixed and cleanly separable.

**2. Routing** — Classify input first, then direct to a specialized sub-prompt. Prevents cross-contamination between task types (e.g., routing customer queries to domain-specific models).

**3. Parallelization** — Two variants:

* *Sectioning* : independent subtasks run concurrently (e.g., guardrail model ∥ response model)
* *Voting* : same task sampled N times, outputs aggregated (e.g., code vulnerability review)

**4. Orchestrator-Workers** — A central LLM dynamically decomposes tasks and delegates to worker LLMs. Differs from parallelization in that subtasks are  **not pre-defined** ; the orchestrator determines them per-input. Suited for tasks where *"the number of files that need to be changed... likely depend on the task."*

**5. Evaluator-Optimizer** — Generate → evaluate → feedback loop. Two fit criteria: (1) LLM output is demonstrably improvable given human-style feedback, and (2) the LLM can itself provide such feedback .

---

## Agents

> *"Agents are typically just LLMs using tools based on environmental feedback in a loop."*

**Runtime lifecycle** :

1. Receive task via command or interactive discussion
2. Plan and operate independently, acquiring **ground truth** at each step (tool call results, code execution output)
3. Pause at checkpoints for human feedback or when blocked
4. Terminate on completion or stopping condition (e.g., max iterations)

**When to use agents** — open-ended problems where *"it's difficult or impossible to predict the required number of steps, and where you can't hardcode a fixed path."* Higher autonomy comes at the cost of latency, API spend, and compounding errors; extensive sandboxed testing is mandatory .

---

## Agent-Computer Interface (ACI)

A concept analogous to HCI, applied to tool design :

> *"We actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths."*

ACI design principles:

* Give the model tokens to *"think before it writes itself into a corner"*
* Keep formats close to naturally occurring internet text
* Eliminate formatting overhead (avoid line-count tracking, string-escaping in JSON code blocks)
* Poka-yoke arguments to make mistakes structurally harder
* Treat tool docstrings with the same rigor as system prompts

---

## Design Philosophy

Three core principles :

> 1. Maintain **simplicity** in your agent's design.
> 2. Prioritize **transparency** by explicitly showing the agent's planning steps.
> 3. Carefully craft your **ACI** through thorough tool documentation and testing.

> *"Success in the LLM space isn't about building the most sophisticated system. It's about building the* right *system for your needs."*

Add multi-step agentic complexity  **only when it demonstrably improves outcomes** ; for many applications, *"optimizing single LLM calls with retrieval and in-context examples is usually enough."*

# Advanced Tool Use on the Claude Developer Platform — Study Notes

---

## Motivation

The future of AI agents requires working across hundreds to thousands of tools simultaneously. Three compounding failure modes arise at this scale :

1. **Context bloat** — five typical MCP servers (GitHub, Slack, Sentry, Grafana, Splunk) consume ~55K tokens  *before any conversation begins* ; Anthropic observed peaks of 134K tokens for tool definitions alone.
2. **Inference overhead** — each natural-language tool call requires a full model inference pass; a five-step workflow means five inference passes plus Claude parsing every intermediate result.
3. **Schema incompleteness** — JSON Schema defines structural validity but cannot express usage conventions: date formats, ID patterns, optional parameter correlations, or which tool to pick when names collide.

Three features address each failure mode independently, then compose :

> *"Tool Search Tool ensures the right tools are found, Programmatic Tool Calling ensures efficient execution, and Tool Use Examples ensure correct invocation."*

---

## Feature 1 — Tool Search Tool (TST)

## Mechanism

Mark tools with `defer_loading: true`; they are excluded from the initial prompt entirely (preserving prompt-cache validity). Claude only sees the `tool_search_tool` (~500 tokens) plus any explicitly pre-loaded critical tools. When it needs a capability, it issues a search; matching tools are expanded into context on demand .

Two built-in search backends are provided: **regex-based** and  **BM25-based** . Custom backends (e.g., dense embeddings) can be substituted .

## Empirical gains

| Metric                     | Before TST  | After TST       |
| -------------------------- | ----------- | --------------- |
| Context at task start      | ~77K tokens | ~8.7K tokens    |
| Context reduction          | —          | **85%**   |
| Opus 4 MCP eval accuracy   | 49%         | **74%**   |
| Opus 4.5 MCP eval accuracy | 79.5%       | **88.1%** |

## Applicability conditions

**Use when:**

* Tool definitions exceed 10K tokens in aggregate
* Encountering tool *selection* errors (especially similar-name collisions like `notification-send-user` vs. `notification-send-channel`)
* Operating 10+ tools, especially multi-server MCP setups

**Skip when:**

* Fewer than 10 tools, all definitions are compact, or all tools are used every session

---

## Feature 2 — Programmatic Tool Calling (PTC)

## Mechanism

Claude writes a Python orchestration script executed in a sandboxed `code_execution` environment. Tools opted into PTC via `allowed_callers: ["code_execution_*"]` are exposed as async Python functions inside the sandbox. Intermediate tool results are processed *within the script* rather than returned to the model's context window. Only `stdout` / final output reaches Claude .

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"><button data-testid="copy-code-button" aria-label="Copy code" type="button" class="focus-visible:bg-subtle hover:bg-subtle text-quiet hover:text-foreground font-sans focus:outline-none outline-none outline-transparent transition duration-300 ease-out select-none items-center relative group/button font-semimedium justify-center text-center items-center rounded-full cursor-pointer active:scale-[0.97] active:duration-150 active:ease-outExpo origin-center whitespace-nowrap inline-flex text-sm h-8 aspect-square" data-state="closed"><div class="flex items-center min-w-0 gap-two justify-center"><div class="flex shrink-0 items-center justify-center size-4"><svg role="img" class="inline-flex fill-current shrink-0" width="16" height="16"><use xlink:href="#pplx-icon-copy"></use></svg></div></div></button></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">python</div></div><div><span><code><span><span class="token token"># Claude generates this; 2000+ expense line items never enter Claude's context</span><span>
</span></span><span><span>team </span><span class="token token operator">=</span><span></span><span class="token token">await</span><span> get_team_members</span><span class="token token punctuation">(</span><span class="token token">"engineering"</span><span class="token token punctuation">)</span><span>
</span></span><span><span>expenses </span><span class="token token operator">=</span><span></span><span class="token token">await</span><span> asyncio</span><span class="token token punctuation">.</span><span>gather</span><span class="token token punctuation">(</span><span class="token token operator">*</span><span class="token token punctuation">[</span><span>get_expenses</span><span class="token token punctuation">(</span><span>m</span><span class="token token punctuation">[</span><span class="token token">"id"</span><span class="token token punctuation">]</span><span class="token token punctuation">,</span><span></span><span class="token token">"Q3"</span><span class="token token punctuation">)</span><span></span><span class="token token">for</span><span> m </span><span class="token token">in</span><span> team</span><span class="token token punctuation">]</span><span class="token token punctuation">)</span><span>
</span></span><span><span>exceeded </span><span class="token token operator">=</span><span></span><span class="token token punctuation">[</span><span>m </span><span class="token token">for</span><span> m</span><span class="token token punctuation">,</span><span> e </span><span class="token token">in</span><span></span><span class="token token">zip</span><span class="token token punctuation">(</span><span>team</span><span class="token token punctuation">,</span><span> expenses</span><span class="token token punctuation">)</span><span>
</span></span><span><span></span><span class="token token">if</span><span></span><span class="token token">sum</span><span class="token token punctuation">(</span><span>x</span><span class="token token punctuation">[</span><span class="token token">"amount"</span><span class="token token punctuation">]</span><span></span><span class="token token">for</span><span> x </span><span class="token token">in</span><span> e</span><span class="token token punctuation">)</span><span></span><span class="token token operator">></span><span> budgets</span><span class="token token punctuation">[</span><span>m</span><span class="token token punctuation">[</span><span class="token token">"level"</span><span class="token token punctuation">]</span><span class="token token punctuation">]</span><span class="token token punctuation">[</span><span class="token token">"travel_limit"</span><span class="token token punctuation">]</span><span class="token token punctuation">]</span><span>
</span></span><span><span></span><span class="token token">print</span><span class="token token punctuation">(</span><span>json</span><span class="token token punctuation">.</span><span>dumps</span><span class="token token punctuation">(</span><span>exceeded</span><span class="token token punctuation">)</span><span class="token token punctuation">)</span><span>
</span></span><span></span></code></span></div></div></div></pre>

> *"Claude's context receives only the final result: the two or three people who exceeded their budget. The 2,000+ line items... do not affect Claude's context."*

## Empirical gains

| Metric                                | Baseline | With PTC                |
| ------------------------------------- | -------- | ----------------------- |
| Avg token consumption (complex tasks) | 43,588   | **27,297**(−37%) |
| Internal knowledge retrieval          | 25.6%    | **28.5%**         |
| GIA benchmark                         | 46.5%    | **51.2%**         |

## Applicability conditions

**Use when:**

* Processing large datasets where only aggregates / summaries are needed downstream
* Multi-step workflows with ≥3 dependent tool calls
* Intermediate data *should not* influence Claude's downstream reasoning (PII, raw logs)
* Parallelizable operations over many items (batch fan-out)

**Skip when:**

* Single-tool lookups with small responses
* Claude must reason over all intermediate results (e.g., chain-of-thought requiring raw evidence)

---

## Feature 3 — Tool Use Examples

## Mechanism

A new `input_examples` field on tool definitions accepts 1–5 concrete JSON call examples. These communicate what JSON Schema *cannot* :

* **Format conventions** — `due_date: "2024-11-06"` (not ISO-8601 with timezone, not natural language)
* **ID conventions** — `reporter.id: "USR-12345"` (not a UUID, not a bare integer)
* **Optional parameter correlations** — `priority: "critical"` co-occurs with tight `sla_hours` and full `reporter.contact`; `priority: "low"` omits escalation entirely
* **Inter-tool disambiguation** — examples on `create_ticket` vs. `create_incident` resolve which to call in ambiguous situations

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"><button data-testid="copy-code-button" aria-label="Copy code" type="button" class="focus-visible:bg-subtle hover:bg-subtle text-quiet hover:text-foreground font-sans focus:outline-none outline-none outline-transparent transition duration-300 ease-out select-none items-center relative group/button font-semimedium justify-center text-center items-center rounded-full cursor-pointer active:scale-[0.97] active:duration-150 active:ease-outExpo origin-center whitespace-nowrap inline-flex text-sm h-8 aspect-square" data-state="closed"><div class="flex items-center min-w-0 gap-two justify-center"><div class="flex shrink-0 items-center justify-center size-4"><svg role="img" class="inline-flex fill-current shrink-0" width="16" height="16"><use xlink:href="#pplx-icon-copy"></use></svg></div></div></button></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">json</div></div><div><span><code><span><span class="token token property">"input_examples"</span><span class="token token operator">:</span><span></span><span class="token token punctuation">[</span><span>
</span></span><span><span></span><span class="token token punctuation">{</span><span class="token token property">"title"</span><span class="token token operator">:</span><span></span><span class="token token">"Login page 500 error"</span><span class="token token punctuation">,</span><span></span><span class="token token property">"priority"</span><span class="token token operator">:</span><span></span><span class="token token">"critical"</span><span class="token token punctuation">,</span><span>
</span></span><span><span></span><span class="token token property">"reporter"</span><span class="token token operator">:</span><span></span><span class="token token punctuation">{</span><span class="token token property">"id"</span><span class="token token operator">:</span><span></span><span class="token token">"USR-12345"</span><span class="token token punctuation">}</span><span class="token token punctuation">,</span><span></span><span class="token token property">"due_date"</span><span class="token token operator">:</span><span></span><span class="token token">"2024-11-06"</span><span class="token token punctuation">,</span><span>
</span></span><span><span></span><span class="token token property">"escalation"</span><span class="token token operator">:</span><span></span><span class="token token punctuation">{</span><span class="token token property">"level"</span><span class="token token operator">:</span><span></span><span class="token token">2</span><span class="token token punctuation">,</span><span></span><span class="token token property">"sla_hours"</span><span class="token token operator">:</span><span></span><span class="token token">4</span><span class="token token punctuation">}</span><span class="token token punctuation">}</span><span class="token token punctuation">,</span><span>
</span></span><span><span></span><span class="token token punctuation">{</span><span class="token token property">"title"</span><span class="token token operator">:</span><span></span><span class="token token">"Update docs"</span><span class="token token punctuation">}</span><span></span><span class="token token">// minimal pattern: title only</span><span>
</span></span><span><span></span><span class="token token punctuation">]</span><span>
</span></span><span></span></code></span></div></div></div></pre>

## Empirical gains

Internal testing: parameter-handling accuracy **72% → 90%** on complex nested schemas .

## Applicability conditions

**Use when:**

* Deeply nested structures with ambiguous optional fields
* Domain-specific format conventions (IDs, dates, enums with implicit semantics)
* Pairs of similar tools where examples clarify selection logic

**Skip when:**

* Single-parameter tools with obvious usage
* Formats Claude already generalizes well (URLs, ISO emails)
* Constraints better expressed as JSON Schema `enum` / `pattern` / `format`

---

## Decision Tree: Which Feature to Reach For

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"><button data-testid="copy-code-button" aria-label="Copy code" type="button" class="focus-visible:bg-subtle hover:bg-subtle text-quiet hover:text-foreground font-sans focus:outline-none outline-none outline-transparent transition duration-300 ease-out select-none items-center relative group/button font-semimedium justify-center text-center items-center rounded-full cursor-pointer active:scale-[0.97] active:duration-150 active:ease-outExpo origin-center whitespace-nowrap inline-flex text-sm h-8 aspect-square" data-state="closed"><div class="flex items-center min-w-0 gap-two justify-center"><div class="flex shrink-0 items-center justify-center size-4"><svg role="img" class="inline-flex fill-current shrink-0" width="16" height="16"><use xlink:href="#pplx-icon-copy"></use></svg></div></div></button></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>Is your bottleneck…
</span></span><span>
</span><span>  Context bloat from too many tool definitions?
</span><span>  └─→ Tool Search Tool  (defer_loading: true)
</span><span>
</span><span>  Large intermediate results or too many round-trips?
</span><span>  └─→ Programmatic Tool Calling  (allowed_callers)
</span><span>
</span><span>  Malformed parameters or wrong tool selection?
</span><span>  └─→ Tool Use Examples  (input_examples: [...])
</span><span>
</span><span>  All three? Layer them — they compose independently.
</span><span></span></code></span></div></div></div></pre>

---

## Citation

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"><button data-testid="copy-code-button" aria-label="Copy code" type="button" class="focus-visible:bg-subtle hover:bg-subtle text-quiet hover:text-foreground font-sans focus:outline-none outline-none outline-transparent transition duration-300 ease-out select-none items-center relative group/button font-semimedium justify-center text-center items-center rounded-full cursor-pointer active:scale-[0.97] active:duration-150 active:ease-outExpo origin-center whitespace-nowrap inline-flex text-sm h-8 aspect-square" data-state="closed"><div class="flex items-center min-w-0 gap-two justify-center"><div class="flex shrink-0 items-center justify-center size-4"><svg role="img" class="inline-flex fill-current shrink-0" width="16" height="16"><use xlink:href="#pplx-icon-copy"></use></svg></div></div></button></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>@article{wu2024advancedtooluse,
</span></span><span>  title   = {Introducing Advanced Tool Use on the Claude Developer Platform},
</span><span>  author  = {Wu, Bin and Jones, Adam and Renault, Artur and Tay, Henry
</span><span>             and Noble, Jake and McCandlish, Nathan and Picard, Noah
</span><span>             and Jiang, Sam and {Claude Developer Platform Team}},
</span><span>  journal = {Anthropic Engineering Blog},
</span><span>  year    = {2024},
</span><span>  month   = {Nov},
</span><span>  url     = {https://www.anthropic.com/engineering/advanced-tool-use}
</span><span>}
</span><span></span></code></span></div></div></div></pre>

**See also:**

* Schluntz & Zhang (2024).  *Building Effective Agents* . Anthropic Engineering Blog.
* Anthropic (2024).  *Code Execution with MCP: Building More Efficient AI Agents* . Anthropic Engineering Blog.
* Schick et al. (2023).  *Toolformer: Language Models Can Teach Themselves to Use Tools* . NeurIPS 2023.
* Yao et al. (2023).  *ReAct: Synergizing Reasoning and Acting in Language Models* . ICLR 2023.[[anthropic](https://www.anthropic.com/research/building-effective-agents)]

## Citation

<pre class="not-prose w-full rounded font-mono text-sm font-extralight"><div class="codeWrapper text-light selection:text-super selection:bg-super/10 my-md relative flex flex-col rounded-lg font-mono text-sm font-medium bg-subtler"><div class="translate-y-xs -translate-x-xs bottom-xl mb-xl flex h-0 items-start justify-end sm:sticky sm:top-xs"><div class="overflow-hidden rounded-full border-subtlest ring-subtlest divide-subtlest bg-base"><div class="border-subtlest ring-subtlest divide-subtlest bg-subtler"><button data-testid="copy-code-button" aria-label="Copy code" type="button" class="focus-visible:bg-subtle hover:bg-subtle text-quiet hover:text-foreground font-sans focus:outline-none outline-none outline-transparent transition duration-300 ease-out select-none items-center relative group/button font-semimedium justify-center text-center items-center rounded-full cursor-pointer active:scale-[0.97] active:duration-150 active:ease-outExpo origin-center whitespace-nowrap inline-flex text-sm h-8 aspect-square" data-state="closed"><div class="flex items-center min-w-0 gap-two justify-center"><div class="flex shrink-0 items-center justify-center size-4"><svg role="img" class="inline-flex fill-current shrink-0" width="16" height="16"><use xlink:href="#pplx-icon-copy"></use></svg></div></div></button></div></div></div><div class="-mt-xl"><div><div data-testid="code-language-indicator" class="text-quiet bg-subtle py-xs px-sm inline-block rounded-br rounded-tl-lg text-xs font-thin">text</div></div><div><span><code><span><span>@article{schluntz2024agents,
</span></span><span>  title   = {Building Effective Agents},
</span><span>  author  = {Schluntz, Erik and Zhang, Barry},
</span><span>  journal = {Anthropic Engineering Blog},
</span><span>  year    = {2024},
</span><span>  month   = {Dec},
</span><span>  url     = {https://www.anthropic.com/engineering/building-effective-agents}
</span><span>}
</span><span></span></code></span></div></div></div></pre>

**See also:**

* Weng, L. (2023).  *LLM-Powered Autonomous Agents* . `lilianweng.github.io`[[anthropic](https://www.anthropic.com/research/building-effective-agents)]
* Wei et al. (2022).  *Chain of Thought Prompting Elicits Reasoning in LLMs* . NeurIPS 2022[[anthropic](https://www.anthropic.com/research/building-effective-agents)]
* Yao et al. (2023).  *ReAct: Synergizing Reasoning and Acting in Language Models* . ICLR 2023[[anthropic](https://www.anthropic.com/research/building-effective-agents)]

Add to follow-up

Check sources
