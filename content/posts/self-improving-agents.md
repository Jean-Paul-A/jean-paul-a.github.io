
---
title: "Self-Improving Agents"
date: 2026-02-19
draft: false
math: true
tags: ["AI", "Agents", "Reinforcement Learning"]
---

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

---

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
