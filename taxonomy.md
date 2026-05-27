---
eyebrow: CDL • Architecture Reference • 2026
title: A *Mechanical* Agentic Taxonomy
footer: Customer Design Labs • 2026
---

## [Preamble] What This Document Is {#what-this-is}

Anthropomorphic metaphors have their place in how we talk about agentic systems. Calling an agent a *persona*, a skill a *runbook*, or a subagent a *teammate* can speed up early design conversations. But for engineers without a working intuition for these systems, the same vocabulary can feel imprecise or worse, inaccessible. Metaphors must yield to the mechanical and architectural realities of how these primitives actually work; the mechanics are the foundation, and metaphors sit on top of them.

This document is what you get when you commit to that view. It is a mechanical explanation of the various components that make up modern agentic systems: tokens, models, inference, context windows, harnesses, agentic loops, agents, tools, skills, MCP, and subagents. The organizing claim is simple: agentic architecture is context architecture. Once you ask what the model can see, what the harness can execute, and what happens to context between model requests, the tradeoffs become surfaceable rather than speculative.

It is not a tutorial, a how-to for any specific harness (Claude Code, Copilot CLI, OpenCode, etc.), or an API reference. The goal is a shared mental model that any team member can use to evaluate, design, or critique an agentic system without getting trapped in vendor-specific metaphor.

## [I · Foundations] Tokens {#tokens}

**Definition:** A token is a chunk of text used as a unit of input and output. Tokens are not necessarily words; they may be words, parts of words, punctuation, whitespace, or other text fragments.

## [II · Foundations] The Model {#model}

**Definition:** A model is a stateless service that receives tokens and returns predicted tokens. It does not remember previous requests, execute code, inspect your environment, or take action; it only returns text.

> **Examples:**
>
> - GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, Llama 4 Maverick, and other named model endpoints.

When we say a model *has* a tool, *uses* a skill, or *acts* as an agent, we are describing harness behavior around the model, not properties inside the model. Everything else in this document exists because the harness decides what tokens to send, how to interpret the tokens that come back, and what to do before the next model request.

## [III · Foundations] Inference {#inference}

**Definition:** An inference is a single operation in which a model receives tokens and returns predicted tokens. It is what the model does when a request reaches its API: tokens in, predicted tokens out.

## [IV · Foundations] The Context Window {#context-window}

**Definition:** The context window is the complete set of tokens sent to the model in a request, constituting its entire working state for that inference; anything outside the context window does not exist to the model, and everything inside competes for finite attention and space.

This is why agentic architecture is largely context architecture. A system prompt, a tool schema, a file attachment, a skill body, a tool result, or a subagent summary becomes operational only when the harness places it into the model request happening now. The mechanical question is always when something enters context, how much space it consumes, and what role it plays in the next inference.

## [V · Foundations] The Harness {#harness}

**Definition:** The harness is the runtime or application responsible for operating the model: it constructs each request, sends tokens to the model, parses tokens returned by the model, executes any requested actions in its own environment, and decides what populates the context window for the next model request.

> **Examples:**
>
> - Claude Code, GitHub Copilot CLI, Cursor, Aider, Cline, OpenHands, and custom LangChain or LangGraph applications.

Whether it is a coding assistant in your IDE, a chat interface, or a custom Python script calling an API, the harness owns the state and side effects the model lacks. The model can request execution only by emitting tokens; the harness decides whether that request maps to an available capability, runs the action if allowed, and places the result in the context window for the next model request.

Every construct in this document is fundamentally a design pattern dictating how the harness manages context and takes action; not the model, because it cannot.

## [VI · Foundations] The Agentic Loop {#loop}

**Definition:** The agentic loop is the cycle a harness runs for every prompt: it sends a model request, parses the returned tokens for requested actions, executes any allowed actions, places results back into context, and repeats until no further actions are requested.

A single inference can only produce an output. For a simple prompt, the loop may complete after one model request. For a more complex prompt, such as looking up documentation, calculating a value, or mutating state, the harness must turn that output into the next input after any external work completes.

1. **Model Request:** The harness sends the current context window to the model.
2. **Inference:** The model receives the tokens and returns predicted tokens.
3. **Parsing:** The harness parses the returned tokens for requested actions, such as a formatted JSON tool call.
4. **Execution:** If an allowed action was requested, the harness executes it in its own runtime environment, outside the model.
5. **Context Update:** If execution occurred, the harness places the result in the context window for the next model request.
6. **Repeat or Stop:** If further action is needed, the harness sends another model request with the updated context. If no action is requested, the loop ends with the model's output.

The *agent* in *agentic* is simply this loop. Strip it away, and you have a standard chatbot. Add it, and you have a system capable of chaining actions together to achieve a goal.

## [VII · Foundations] Agents {#agents}

**Definition:** An agent is a configured agentic loop defined by a system prompt, a toolset, and a skill set; it is the container that the other primitives fill.

> **Examples:**
>
> - Planner, Builder, Researcher, Code Reviewer, QA, and Security Reviewer agents configured with different prompts, tools, skills, and permissions.

An agent is not a peer to tools or skills; it is the container that determines which instructions, capabilities, and procedural knowledge are available inside a particular loop. Mechanically, that configuration is defined by three things:

1. **A System Prompt:** Defining the model's role, purpose, and behavioral guidance.
2. **A Toolset:** The specific capabilities exposed to the loop, including the permissions and access boundaries enforced by the harness.
3. **A Skill Set:** The procedural knowledge available for the model to read in on demand.

Most harnesses ship with one or more **built-in agents**: predefined system prompts and toolsets exposed as *modes* (e.g., Planner, Coder, Researcher). Some harnesses also support **user-defined agents**, allowing operators to register their own system prompts and curate which tools, MCPs, and skills are available within that agent's loop. The capability to bring your own agent is not universal; it is a deliberate harness feature, and its absence can be a meaningful constraint when evaluating a platform.

Tools, MCPs, skills, and subagents are the primitives that fill this container, whether the container is built-in or user-defined.

## [VIII · Lens] The Three Axes {#axes}

Before comparing tools, skills, MCP, and subagents, it helps to use the same mechanical questions for each one:

1. **Initialization:** When does its model-visible payload enter the context window?
2. **Execution Context:** Where does the work happen?
3. **Payload:** What is exposed to the model?

These axes are not primitives themselves. They are the lens for comparing primitives: when each one becomes available, where execution happens, and what payload the model can see.

| Primitive | Initialization | Execution Context | Payload |
| --- | --- | --- | --- |
| Tool | Session initialization | Current context | Schema only (opaque code) |
| Skill | On skill request | Current context | Manifest, scripts, files (transparent) |
| Subagent | On parent delegation | Separate context | Isolated prompt & toolset |

Note that this table flattens primitives onto comparable axes but does not capture composition: skills routinely invoke tools, including MCP-delivered tools, and subagents are themselves agents configured with their own tools and skills.

## [IX · Primitive] Tools {#tools}

**Definition:** A tool is a named operation made available to the model through a description and parameter schema. When the model returns a structured tool call matching that schema, the harness executes the underlying code outside the model and places the result in the context window for the next model request.

> **Examples:**
>
> - `read_file`, `write_file`, `execute_shell`, `read_web`, `search_code`, and `mcp-atlassian-jira_get_issue`.

- **Initialization:** The tool's name, description, and schema are loaded into the system prompt at the start of the session. Some harnesses with large tool inventories route schemas into context based on relevance rather than loading them all at session start (for example, Claude Code defers MCP tool definitions by default, initially exposing only tool names); the payload itself remains an opaque contract either way.
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

## [X · Primitive] Skills {#skills}

**Definition:** A skill is a named bundle of procedural knowledge made available to the model through a description. When the model requests the skill, the harness reads its full body, usually instructions plus supporting files, into the context window for the next model request.

> **Examples:**
>
> - Code-review runbooks; release or deployment runbooks; incident-response playbooks; organization-specific style-guide skills; and Anthropic's shipped document-processing skills for PDF, DOCX, XLSX, and PPTX.

Where a tool exposes an operation the harness can execute, a skill exposes instructions the model can follow, often by orchestrating one or more tool calls along the way.

Mechanically, skill invocation resembles tool invocation at the model boundary: the model sees a name and description, then emits tokens the harness parses as a request to load that skill. The difference is the harness response. A tool request executes opaque code and returns a result; a skill request reads transparent procedural content into the context window.

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

## [XI · The Harness Extended] MCP {#mcp}

**Definition:** MCP is a standardized protocol that lets a harness connect to external servers that expose tools, resources, or prompts. MCP does not give the model a new capability directly; the harness must register or inject what the server provides before it can appear in the context window or be requested by the model.

> **Examples:**
>
> - GitHub MCP server, filesystem MCP server, Playwright MCP server, PostgreSQL MCP server, Atlassian/Jira MCP server, and Terraform MCP server.

MCP is therefore not a new kind of model capability. Like the other primitives in this taxonomy, it exists in the harness layer: it extends the harness's perimeter by moving authentication, portability, vendor coupling, permissions, and service ownership out of the local runtime and into a protocol relationship with another system.

Today, MCP-exposed tools are the most common use case, but the protocol is broader than tool delivery. An MCP server can expose tools, resources, and prompts; the harness can expose capabilities back to the server, such as LLM inference, workspace scope, and user input collection. Once registered or injected by the harness, these resolve into patterns this document has already described:

- An MCP tool, once registered by the harness, is a tool: schema enters context, implementation remains opaque, result returns as context.
- An MCP resource or prompt, once injected by the harness, is context: tokens added to the next model request, no different in kind from a local file, a system prompt fragment, or a skill body.

What MCP does not change is the harness's responsibility: deciding what to expose, what to execute, what to inject, and what reaches the model.

In practice, most harnesses fully surface MCP tools. Support for resources and prompts is uneven. Some expose them as attachable context and slash commands; others ignore them entirely. That variance is the point: the specification may define what can cross the boundary, but the harness decides what actually becomes available in the loop.

> **Key Takeaway:** MCP standardizes the boundary between a harness and external capability servers. It does not create a new model primitive; what arrives over MCP still becomes a tool, context, or harness-mediated request before it reaches the model.

## [XII · Primitive] Subagents {#subagents}

**Definition:** A subagent is a separate agentic loop started by the harness in response to a parent agent's request. It runs with its own context window, system prompt, and toolset; when it finishes, the harness places only its final output back into the parent's context window.

> **Examples:**
>
> - Research, code-review, test-failure triage, security-review, documentation, migration-planning, and dependency-upgrade subagents spawned by a parent coding agent.

The core tradeoff is **context and toolset isolation**. A subagent helps when a task can be bounded, needs many tokens, can run independently, or should operate with narrower permissions. It hurts when the task depends tightly on the parent's surrounding context, is small, or would require an expensive summary to be useful.

When the harness starts a subagent, the parent loop waits while the child loop works. The parent receives a summary or artifact, not the child's full intermediate history. This keeps the parent context cleaner, but detail can be lost at the handoff.

The token cost is material. A parent-child workflow spends tokens in both loops: the parent frames the task and absorbs the result, while the child spends context on investigation, tool use, and summarization. Anthropic's June 2025 multi-agent research system report found roughly 15× more token use than comparable single-agent baselines. The question is whether isolation, scoped permissions, or parallel exploration justify the added coordination and spend.

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

## [XIII · Synthesis] Implications {#implications}

By evaluating these constructs mechanically rather than metaphorically, several architectural tradeoffs become clear:

- **The harness, loop, and agent are the operating frame.** You do not choose between a tool and an agent. A tool is useful only because a harness can run an agentic loop: expose a schema, parse a request, execute code, and place the result back into context.
- **MCP is not a primitive.** MCP is a delivery boundary. You can have a local tool, an MCP-delivered tool, or a skill that instructs the model to use an MCP-delivered tool.
- **Skills and tools compose rather than compete.** A tool exposes an operation the harness can execute. A skill exposes procedural knowledge the model can follow, often by invoking one or more tools. The reverse does not hold: tools have no mechanism to contain or invoke skills.
- **Subagents trade isolation for coordination cost.** A subagent provides a fresh context window and scoped toolset, but introduces delegation, summarization, and token overhead. It is useful when that isolation is worth more than the added cost.
- **Primitive selection is an architectural decision.** Personas, runbooks, and teammates can be useful metaphors while sketching a system, but they do not define implementation boundaries. The mechanical questions are what the model can see, what the harness can execute, and what happens to context between model requests.
