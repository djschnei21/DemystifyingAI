---
eyebrow: CDL • Architecture Reference • 2026
title: A *Mechanical* Agentic Taxonomy
footer: Customer Design Labs • 2026
---

## [Preamble] What This Document Is {#what-this-is}

Anthropomorphic metaphors have their place in how we talk about agentic systems. Calling an agent a *persona*, a skill a *runbook*, or a subagent a *teammate* can speed up early design conversations. But for engineers without a working intuition for these systems, the same vocabulary can feel imprecise or worse, inaccessible. Metaphors must yield to the mechanical and architectural realities of how these primitives actually work; the mechanics are the foundation, and metaphors sit on top of them.

This document is what you get when you commit to that view. It is a mechanical reference for the primitives that make up modern agentic systems: harness, agentic loop, agent, tool, MCP, skill, and subagent. The organizing lens is simple: each primitive is defined by how it interacts with the model's context window. From there, the architectural tradeoffs that follow become surfaceable rather than speculative.

It is not a tutorial, a how-to for any specific harness (Claude Code, Copilot CLI, OpenCode, etc.), or an API reference. The goal is a shared mental model that any team member can use to evaluate, design, or critique an agentic system without getting trapped in vendor-specific metaphor.

## [I · Foundations] The Model {#model}

An LLM is a stateless engine that does one thing: it reads a sequence of tokens and predicts the next sequence of tokens. It retains no memory between calls, executes no code, and possesses no tools, skills, agents, or goals of its own.

When we say a model *has* a tool, *uses* a skill, or *acts* as an agent, we are describing harness behavior around the model, not properties inside the model. The model only receives tokens and emits tokens. Everything else in this document exists because the harness decides what tokens to send, how to interpret the tokens that come back, and what to do before the next call.

## [II · Foundations] The Context Window {#context-window}

The context window is the total set of tokens sent to the model in a single request. It is the model's entire working state for that inference. Anything not present in the context window does not exist to the model; anything present consumes finite token budget and competes for attention.

This is why agentic architecture is largely context architecture. A system prompt, a tool schema, a file attachment, a skill body, a tool result, or a subagent summary becomes operational only when the harness places it into a model request. The mechanical question is always when something enters context, how much space it consumes, and what role it plays in the next inference.

## [III · Foundations] The Harness {#harness}

Because the model cannot act on its own, it requires a runtime environment. This is the harness (sometimes called the host).

Whether it is a coding assistant in your IDE, a chat interface, or a custom Python script calling an API, the harness is responsible for:

- Feeding text to the model.
- Parsing the model's output.
- Executing any code or actions the model requests.
- Feeding the results back into the model's context window.

Every construct in this document is fundamentally a design pattern dictating how the harness manages the context window and routes execution.

## [IV · Foundations] The Agentic Loop {#loop}

A single model inference produces one chunk of text. To accomplish complex tasks, such as looking up documentation, calculating a value, or mutating state, the system must iterate. The agentic loop is how the harness iterates.

1. **Inference:** The harness sends an atomic API request to the model with the current context, and receives a generated text output in response.
2. **Parsing:** The harness parses the output for any requested actions (like a formatted JSON tool call).
3. **Execution:** The harness executes the requested external actions in its own runtime environment, between model requests.
4. **Context Update:** The harness appends the execution results to the context window. Because the model's token capacity is finite, the harness is mechanically responsible for bounded state management, ensuring the accumulated payload remains within limits before the next inference.
5. **Recurse:** The harness sends a new request to the model with the extended context. This repeats until the model produces an output with no further action requests.

The *agent* in *agentic* is simply this loop. Strip it away, and you have a standard chatbot. Add it, and you have a system capable of chaining actions together to achieve a goal.

## [V · Foundations] Agents {#agents}

An agent is not a peer to tools or skills; it is the container.

Mechanically, an agent is a configured agentic loop defined by three things:

1. **A System Prompt:** Defining the model's role, purpose, and behavioral guidance.
2. **A Toolset:** The specific capabilities exposed to the loop, including the permissions and access boundaries enforced by the harness.
3. **A Skill Set:** The procedural knowledge available for the model to read in on demand.

Most harnesses ship with one or more **built-in agents**: predefined system prompts and toolsets exposed as *modes* (e.g., Planner, Coder, Researcher). Some harnesses also support **user-defined agents**, allowing operators to register their own system prompts and curate which tools, MCPs, and skills are available within that agent's loop. The capability to bring your own agent is not universal; it is a deliberate harness feature, and its absence can be a meaningful constraint when evaluating a platform.

Tools, MCPs, skills, and subagents are the primitives that fill this container, whether the container is built-in or user-defined.

## [VI · Primitive] Tools {#tools}

A tool is the smallest functional construct. It consists of a description and a parameter schema.

- **Initialization:** The schema is loaded into the system prompt at the start of the session. Some harnesses with large tool inventories route schemas into context based on relevance rather than loading them all at session start; the payload itself remains opaque schema either way.
- **Execution:** When the model emits a tool call matching the schema, the harness intercepts the request, runs the underlying code in its own runtime environment, and appends the return value as text to the context window before issuing the next model request. The model itself never executes code; it only requests execution.
- **Visibility:** The implementation is opaque to the model. The model only sees the schema going in and the text result coming back; it never sees the underlying code.

```text
# PSEUDOCODE - illustrative only; actual syntax varies by harness
tool {
  name:           "search_docs"
  description:    "Search the IBM Knowledge Center. Use when the user
                   asks about a product, configuration, or how-to."
  parameters: {
    query:        string    # required
    max_results:  integer   # default: 5
    product:      string    # optional filter, e.g. "watsonx"
  }
  implementation: search_docs_impl()   # executed by harness; opaque to model
}
```

## [VII · Primitive] Skills {#skills}

Skills package procedural knowledge (runbooks, templates, conventions, and the supporting files needed to execute them) and inject that knowledge into the context window on demand. Where a tool exposes a *capability*, a skill delivers *instructions* on how to accomplish something, often by orchestrating one or more tool calls along the way.

Two mechanical properties distinguish them from tools:

- **Lazy Loading:** Only the skill's name and a brief description sit in the system prompt. The full body (manifest, instructions, supporting files) is only injected into the context window when the model explicitly decides to read it.
- **Transparent Payload:** A skill is a directory containing a manifest and supporting files (scripts, templates, prose). Unlike a tool, which hides its code, a skill allows the model to open and read its bundled contents *before* invoking them (usually via a generic execution tool provided by the harness).

Crucially, skills compose with tools rather than replacing them. A skill's instructions typically direct the model to invoke specific tools (local or MCP-delivered) in a particular sequence, with branching logic the model interprets at read time. The relationship is hierarchical: skills can orchestrate tools; tools cannot contain skills.

```text
# PSEUDOCODE - illustrative only; actual structure varies by harness
skill {
  name:        "deploy-to-prod"
  description: "Production deployment runbook. Invoke before pushing
                to prod or when a deployment fails."
  body: {        # lazy-loaded only when the model decides to read it
    instructions: "1. Verify CI status.  2. Run canary deploy.  3. Smoke
                   test endpoints.  4. Promote to full prod.  5. On
                   failure, invoke rollback_template.yaml."
    files:        [check_ci.sh, rollback_template.yaml, smoke_tests.py]
  }
}
```

## [VIII · The Harness Extended] MCP {#mcp}

MCP (Model Context Protocol) is not a new primitive. It is a standardized protocol boundary between the harness and external capability servers.

Today, MCP exposed tools are the most common use case. However, the protocol is broader than tool delivery. An MCP server can expose tools, resources, and prompts; the harness can expose capabilities back to the server, such as LLM inference, workspace scope, and user input collection. But none of these introduce new mechanics inside the model. They resolve into patterns this document has already described:

- An MCP tool, once registered by the harness, is a tool: schema enters context, implementation remains opaque, result returns as context.
- An MCP resource or prompt, once injected by the harness, is context: tokens added to the next model call, no different in kind from a local file, a system prompt fragment, or a skill body.

What MCP changes is the source and governance boundary. Authentication, portability, vendor coupling, permissions, and service ownership move out of the local harness and into a protocol relationship with another system. What MCP does not change is the harness's responsibility: deciding what to expose, what to execute, what to inject, and what reaches the model.

In practice, most harnesses fully surface MCP tools. Support for resources and prompts is uneven. Some expose them as attachable context and slash commands; others ignore them entirely. That variance is the point: the specification may define what can cross the boundary, but the harness decides what actually becomes available in the loop.

> **Key Takeaway:** MCP standardizes the boundary, not the primitive. What arrives over MCP still becomes a tool, context, or harness-mediated request before it reaches the model.

## [IX · Primitive] Subagents {#subagents}

Long exploratory tasks eventually bloat the context window, leaving the model with no token space to *think*. Subagents solve this through **context and toolset isolation**.

When a parent agent invokes a subagent, the harness spins up a secondary agentic loop with its own blank context window, system prompt, and toolset. That toolset can be narrower than, or entirely different from, the parent's: useful when a planner subagent should not have write access, or a research subagent should be restricted to read-only operations. The parent's loop pauses. When the subagent finishes, only its final output (a summary or artifact) is returned to the parent's context window. Subagents trade intermediate detail for context preservation and capability scoping.

This isolation comes at a measurable token cost. Anthropic's June 2025 multi-agent research system report found that multi-agent workflows consume roughly 15× more tokens than equivalent chat interactions, with token usage alone explaining ~80% of performance variance on research evaluations. For coding workflows specifically, Claude Code's documentation cites ~7× token consumption for agent teams over single-thread sessions. Despite the higher total spend, the value is keeping the parent's context window clean while expensive work happens elsewhere, and enabling parallel exploration that a single loop cannot achieve. The corollary is that for trivial or tightly-coupled work, the startup cost is not worth paying; stay on the main thread.

```text
# PSEUDOCODE - illustrative only; actual structure varies by harness
subagent {
  name:          "code-reviewer"
  description:   "Invoke for pull request review or pre-commit code audit."
  system_prompt: "You are a senior reviewer. Focus on correctness,
                  security, and style. Cite line numbers and propose
                  concrete diffs when possible."
  tools:         [read_file, grep, run_tests]   # narrower than parent
  model:         "sonnet"   # cheaper than parent's opus
  permissions:   read_only
}
```

## [X · Synthesis] The Three Axes {#axes}

The meaningful differences between these primitives come down to three mechanical axes:

1. **Initialization:** When does it enter the context window?
2. **Execution Context:** Where does the work happen?
3. **Payload:** What is exposed to the model?

| Primitive | Initialization | Execution Context | Payload |
| --- | --- | --- | --- |
| Tool | Session start | Current context | Schema only (opaque code) |
| MCP Tool | Session start | Current context | Schema only (remote code) |
| Skill | Lazy (on-demand) | Current context | Manifest, scripts, files (transparent) |
| Subagent | On parent invocation | Separate context | Isolated prompt & toolset |

Note that this table flattens primitives onto comparable axes but does not capture composition: skills routinely invoke tools (local or MCP), and subagents are themselves agents configured with their own tools and skills.

## [XI · Synthesis] Implications {#implications}

By evaluating these constructs mechanically rather than metaphorically, several architectural tradeoffs become clear:

- **The harness, the loop, and the agent are givens.** You do not choose between a tool and an agent. The tool exists *because* the agentic loop exists.
- **MCP is not a peer to skills.** MCP is a delivery mechanism. You can have a local tool, an MCP-delivered tool, or a skill that utilizes an MCP tool.
- **Skills and tools are not peers; they compose.** A skill typically directs the model through a procedure that invokes one or more tools. The reverse does not hold: tools have no mechanism to contain or invoke skills.
- **Subagents are the only primitive that provides context isolation.** No amount of clever skill engineering can substitute for a fresh context window. If you need to spend tokens on a side investigation without cluttering the main thread (or to scope down capabilities for a delegated task), you need a subagent.
- **Skills optimize for procedural guidance and transparency.** If you need a predictable action with a rigid contract, write a tool. If you have procedural knowledge (a runbook, a template-driven workflow, a sequence of tool calls with branching logic) that the model should pull in only when relevant, write a skill. If you need an execution context with its own system prompt and toolset, write an agent. A skill is content read into a loop, while an agent is the loop itself.
- **Design concepts versus architecture.** Personas, runbooks, and teammates can be useful metaphors while sketching a system, but they do not properly define implementation boundaries or architectural tradeoffs. The mechanical question is always when a construct enters the context window, where execution happens, and what payload is exposed to the model. Once framed that way, primitive selection becomes an architectural decision rather than a naming exercise.
