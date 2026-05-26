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

**Definition:** A model is a stateless token-prediction engine that reads tokens in and emits tokens out, with no memory, code execution, or capabilities of its own between calls.

> **Examples:**
> - GPT-5.5, Claude Opus 4.7, Gemini 2.5 Pro, Llama 4 Maverick, and other named model endpoints.

When we say a model *has* a tool, *uses* a skill, or *acts* as an agent, we are describing harness behavior around the model, not properties inside the model. Everything else in this document exists because the harness decides what tokens to send, how to interpret the tokens that come back, and what to do before the next call.

## [II · Foundations] The Context Window {#context-window}

**Definition:** The context window is the complete set of tokens sent to the model in a single request, constituting its entire working state for that inference; anything outside it does not exist to the model, and everything inside competes for finite space.

This is why agentic architecture is largely context architecture. A system prompt, a tool schema, a file attachment, a skill body, a tool result, or a subagent summary becomes operational only when the harness places it into a model request. The mechanical question is always when something enters context, how much space it consumes, and what role it plays in the next inference.

## [III · Foundations] The Harness {#harness}

**Definition:** The harness is the runtime that wraps the model: it sends tokens in, parses tokens out, executes any requested actions in its own environment, and decides what enters the context window for the next call.

> **Examples:**
> - Claude Code, GitHub Copilot CLI, Cursor, Aider, Cline, OpenHands, and custom LangChain or LangGraph applications.

Whether it is a coding assistant in your IDE, a chat interface, or a custom Python script calling an API, the harness owns the state and side effects the model lacks. The model can request execution only by emitting tokens; the harness decides whether that request maps to an available capability, runs the action if allowed, and formats the result for the next inference.

Every construct in this document is fundamentally a design pattern dictating how the harness manages context and routes execution.

## [IV · Foundations] The Agentic Loop {#loop}

**Definition:** The agentic loop is the iterative cycle in which a harness repeatedly sends context to the model, executes the actions the model requests, appends the results back into context, and calls the model again until no further actions are requested.

A single inference can only produce an output. To accomplish complex tasks, such as looking up documentation, calculating a value, or mutating state, the harness must turn that output into the next input after any external work completes.

1. **Inference:** The harness sends an atomic API request to the model with the current context, and receives a generated text output in response.
2. **Parsing:** The harness parses the output for any requested actions (like a formatted JSON tool call).
3. **Execution:** The harness executes the requested external actions in its own runtime environment, between model requests.
4. **Context Update:** The harness appends the execution results to the context window. Because the model's token capacity is finite, the harness is mechanically responsible for bounded state management, ensuring the accumulated payload remains within limits before the next inference.
5. **Recurse:** The harness sends a new request to the model with the extended context.

The *agent* in *agentic* is simply this loop. Strip it away, and you have a standard chatbot. Add it, and you have a system capable of chaining actions together to achieve a goal.

## [V · Foundations] Agents {#agents}

**Definition:** An agent is a configured agentic loop defined by a system prompt, a toolset, and a skill set; it is the container that drives the repeated cycle of calling a model and executing what the model asks for.

> **Examples:**
> - Planner, Builder, Researcher, Code Reviewer, QA, and Security Reviewer agents configured with different prompts, tools, skills, and permissions.

An agent is not a peer to tools or skills; it is the container that gives those primitives a specific operating shape. Mechanically, that configuration is defined by three things:

1. **A System Prompt:** Defining the model's role, purpose, and behavioral guidance.
2. **A Toolset:** The specific capabilities exposed to the loop, including the permissions and access boundaries enforced by the harness.
3. **A Skill Set:** The procedural knowledge available for the model to read in on demand.

Most harnesses ship with one or more **built-in agents**: predefined system prompts and toolsets exposed as *modes* (e.g., Planner, Coder, Researcher). Some harnesses also support **user-defined agents**, allowing operators to register their own system prompts and curate which tools, MCPs, and skills are available within that agent's loop. The capability to bring your own agent is not universal; it is a deliberate harness feature, and its absence can be a meaningful constraint when evaluating a platform.

Tools, MCPs, skills, and subagents are the primitives that fill this container, whether the container is built-in or user-defined.

## [VI · Primitive] Tools {#tools}

**Definition:** A tool is a named capability exposed to the model through a description and parameter schema; the model emits a structured call matching that schema, and the harness executes the underlying opaque code before returning the result as context.

> **Examples:**
> - `read_file`, `write_file`, `execute_shell`, `read_web`, `search_code`, and `mcp_read_jira_task`.

- **Initialization:** The tool's name, description, and schema are loaded into the system prompt at the start of the session. Some harnesses with large tool inventories route schemas into context based on relevance rather than loading them all at session start; the payload itself remains an opaque contract either way.
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

**Definition:** A skill is a bundle of procedural knowledge, usually instructions plus supporting files, whose name and description sit in the system prompt while its full transparent body enters the context window only when read or invoked.

> **Examples:**
> - PDF, DOCX, XLSX, and PPTX document-processing skills; code-review runbooks; release or deployment runbooks; incident-response playbooks; and organization-specific style-guide skills.

Where a tool exposes a *capability*, a skill delivers *instructions* for accomplishing something, often by orchestrating one or more tool calls along the way.

Two mechanical properties distinguish them from tools:

- **Lazy Loading:** Only the skill's name and a brief description sit in the system prompt. The full body (manifest, instructions, supporting files) is only injected into the context window when the model explicitly decides to read or invoke it.
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

**Definition:** MCP is a standardized protocol boundary that lets a harness connect to external servers offering tools, resources, or prompts; it changes where capabilities come from and how they are governed, but anything delivered over it still resolves into a tool or context before reaching the model.

> **Examples:**
> - GitHub MCP server, filesystem MCP server, Playwright MCP server, PostgreSQL MCP server, Atlassian/Jira MCP server, and Terraform MCP server.

MCP is therefore not a new kind of model capability. Like the other primitives in this taxonomy, it exists in the harness layer: it extends the harness's perimeter by moving authentication, portability, vendor coupling, permissions, and service ownership out of the local runtime and into a protocol relationship with another system.

Today, MCP-exposed tools are the most common use case, but the protocol is broader than tool delivery. An MCP server can expose tools, resources, and prompts; the harness can expose capabilities back to the server, such as LLM inference, workspace scope, and user input collection. Once registered or injected by the harness, these resolve into patterns this document has already described:

- An MCP tool, once registered by the harness, is a tool: schema enters context, implementation remains opaque, result returns as context.
- An MCP resource or prompt, once injected by the harness, is context: tokens added to the next model call, no different in kind from a local file, a system prompt fragment, or a skill body.

What MCP does not change is the harness's responsibility: deciding what to expose, what to execute, what to inject, and what reaches the model.

In practice, most harnesses fully surface MCP tools. Support for resources and prompts is uneven. Some expose them as attachable context and slash commands; others ignore them entirely. That variance is the point: the specification may define what can cross the boundary, but the harness decides what actually becomes available in the loop.

> **Key Takeaway:** MCP standardizes the boundary, not the primitive. What arrives over MCP still becomes a tool, context, or harness-mediated request before it reaches the model.

## [IX · Primitive] Subagents {#subagents}

**Definition:** A subagent is a secondary agentic loop invoked by a parent with its own fresh context window, system prompt, and usually narrower toolset; only its final output returns to the parent, trading higher token cost for context isolation and capability scoping.

> **Examples:**
> - Research, code-review, test-failure triage, security-review, documentation, migration-planning, and dependency-upgrade subagents spawned by a parent coding agent.

The core benefit is **context and toolset isolation**. Long exploratory tasks can spend tokens in the subagent's separate loop without bloating the parent's context window.

When a parent invokes a subagent, the parent loop pauses while the secondary loop performs bounded work. The child may have narrower or different tools: useful when a planner should not have write access, or a research worker should be restricted to read-only operations. When it finishes, the parent receives a summary or artifact rather than the child's full intermediate history.

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
