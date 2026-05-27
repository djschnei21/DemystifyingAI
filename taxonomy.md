---
eyebrow: CDL • Architecture Reference • 2026
title: A *Mechanical* Agentic Taxonomy
footer: Customer Design Labs • 2026
---

## [Preamble] What This Document Is {#what-this-is}

Anthropomorphic metaphors have their place in how we talk about agentic systems. Calling an agent a *persona*, a skill a *runbook*, or a subagent a *teammate* can speed up early design conversations. But for engineers without a working intuition for these systems, the same vocabulary can feel imprecise or worse, inaccessible. Metaphors must yield to the mechanical and architectural realities of how these constructs actually work; the mechanics are the foundation, and metaphors sit on top of them.

This document is what you get when you commit to that view. It is a mechanical explanation of the components that make up modern agentic systems, from the model request boundary to the harness patterns built around it. The organizing claim is simple: agentic architecture is context architecture. Once you ask what the model can see, what the harness can execute, and what happens to context between model requests, the tradeoffs become surfaceable rather than speculative.

It is not a tutorial, a how-to for any specific harness (Claude Code, Copilot CLI, OpenCode, etc.), or an API reference. The goal is a shared mental model that any team member can use to evaluate, design, or critique an agentic system without getting trapped in vendor-specific metaphor.

## [I · Foundations] Tokens {#tokens}

**Definition:** A token is a model-side chunk of text used as a unit for processing, prediction, counting, and billing. Tokens are not necessarily words; they may be words, parts of words, punctuation, whitespace, or other text fragments.

## [II · Foundations] The Model {#model}

**Definition:** A model is a stateless service that receives text, processes it as tokens, and returns predicted text. It does not remember previous requests, execute code, inspect your environment, or take action; it only returns text.

![Tokens in, the model, tokens out](diagrams/model.png)

> **Examples:**
>
> - GPT-5.5, Claude Opus 4.7, Gemini 3.1 Pro, Llama 4 Maverick, and other named model endpoints.

When we say a model *has* a tool, *uses* a skill, or *acts* as an agent, we are describing harness behavior around the model, not properties inside the model. Everything else in this document exists because the harness decides what text and context to send, how to interpret the text that comes back, and what to do before the next model request.

## [III · Foundations] Inference {#inference}

**Definition:** An inference is a single model-side operation in which input text is processed as tokens and returned as predicted text. It is what the model does when a request reaches its API: text in, tokenization and prediction inside, text out.

## [IV · Foundations] The Context Window {#context-window}

**Definition:** The context window is the set of tokens the model can use during a single inference. Anything outside the context window does not exist to the model; everything inside competes for finite attention and space.

![The context window contains the system prompt, tool schemas, skill descriptions, user request, and remaining free space](diagrams/contextwindow.png)

This is why agentic architecture is largely context architecture. A system prompt, a tool schema, a file attachment, a skill body, a tool result, or a subagent summary can affect the next inference only if the harness includes it in the next model request. The mechanical question is always when something enters context, how much space it consumes, and what role it plays in the next model request.

## [V · Foundations] The Harness {#harness}

**Definition:** The harness is the runtime or application responsible for operating the model: it constructs each model request, sends it to the model, parses text returned by the model, executes any requested actions in its own environment, and decides what populates the context window for the next model request.

![The harness runs in the environment and mediates requests to and from the model API](diagrams/harness.png)

> **Examples:**
>
> - Claude Code, GitHub Copilot CLI, Cursor, Aider, Cline, OpenHands, and custom LangChain or LangGraph applications.

Whether it is a coding assistant in your IDE, a chat interface, or a custom Python script calling an API, the harness owns the state and side effects the model lacks. The model can request execution only by returning text the harness can parse; the harness decides whether that request maps to an available capability, runs the action if allowed, and places the result in the context window for the next model request.

Every construct in this document is fundamentally a design pattern dictating how the harness manages context and takes action; not the model, because it cannot.

## [VI · Foundations] The Agentic Loop {#loop}

**Definition:** The agentic loop is the cycle a harness runs for every prompt: it sends a model request, parses the returned text for requested actions, executes any allowed actions, places results back into context, and repeats until no further actions are requested.

A single inference can only produce an output. For a simple prompt, the loop may complete after one model request. For a prompt that requires external work, such as looking up documentation, calculating a value, or editing a file, the harness performs that work and places the result into context for another model request.

1. **Model Request:** The harness sends a model request containing the current context.
2. **Inference:** The model processes the request text (context window) and returns predicted text.
3. **Parsing:** The harness parses the returned text for requested actions, such as a formatted JSON tool call.
4. **Execution:** If an allowed action was requested, the harness executes it in its own runtime environment, outside the model.
5. **Context Update:** If execution occurred, the harness places the result in the context window for the next model request.
6. **Repeat or Stop:** If further action is needed, the harness sends another model request with the updated context. If no action is requested, the loop ends with the model's output.

The loop is what makes a harness agentic. Strip it away, and you have a standard chatbot. Add it, and you have a system capable of chaining actions together to achieve a goal.

## [VII · Analytical Lens] The Three Axes {#axes}

Before comparing agents, tools, skills, and subagents, it helps to use the same mechanical questions for each one:

1. **Initialization:** When does its model-visible payload enter the context window?
2. **Execution Context:** Where does the work happen?
3. **Payload:** What is exposed to the model?

The point is not to classify by name, but to compare by mechanics: when something becomes available, where the work happens, and what the model can see.

| Construct | Initialization | Execution Context | Payload |
| --- | --- | --- | --- |
| Agent | On session start or agent selection | Configured loop | System prompt, toolset, skill set, permissions |
| Tool | Session initialization | Current context | Schema only (opaque code) |
| Skill | On skill request | Current context | Manifest, scripts, files (transparent) |
| Subagent | On parent delegation | Separate context | Isolated prompt & toolset |

Note that this table flattens constructs onto comparable axes but does not capture composition: agents contain tools and skills, skills routinely invoke tools, including MCP-delivered tools, and subagents are themselves agents configured with their own tools and skills.

## [VIII · Agentic Construct] Agents {#agents}

**Definition:** An agent is the container construct: a configured agentic loop defined by a system prompt, a toolset, a skill set, and permissions. It determines which instructions, capabilities, procedural knowledge, and delegation paths are available for a task.

> **Examples:**
>
> - Planner, Builder, Researcher, Code Reviewer, QA, and Security Reviewer agents configured with different prompts, tools, skills, and permissions.

An agent is not a peer to tools or skills in the sense of doing the same kind of work. It is the construct that composes them into a particular operating configuration. Mechanically, that configuration is defined by four things:

1. **A System Prompt:** Defining the model's role, purpose, and behavioral guidance.
2. **A Toolset:** The specific capabilities exposed to the loop, including the permissions and access boundaries enforced by the harness.
3. **A Skill Set:** The procedural knowledge available for the model to read in on demand.
4. **Permissions and Delegation Paths:** The boundaries that determine what the agent may execute directly and when it may hand work to another loop.

Most harnesses ship with one or more **built-in agents**: predefined system prompts and toolsets exposed as *modes* (e.g., Planner, Coder, Researcher). Some harnesses also support **user-defined agents**, allowing operators to register their own system prompts and curate which tools, skills, MCP connections, and subagents are available within that agent's loop. The capability to bring your own agent is not universal; it is a deliberate harness feature, and its absence can be a meaningful constraint when evaluating a platform.

## [IX · Agentic Construct] Tools {#tools}

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

## [X · Agentic Construct] Skills {#skills}

**Definition:** A skill is a named bundle of procedural knowledge made available to the model through a description. When the model requests the skill, the harness reads its full body, usually instructions plus supporting files, into the context window for the next model request.

> **Examples:**
>
> - Code-review runbooks; release or deployment runbooks; incident-response playbooks; organization-specific style-guide skills; and Anthropic's shipped document-processing skills for PDF, DOCX, XLSX, and PPTX.

Where a tool exposes an operation the harness can execute, a skill exposes instructions the model can follow, often by orchestrating one or more tool calls along the way.

Mechanically, skill invocation resembles tool invocation at the model boundary: the model sees a name and description, then returns text the harness parses as a request to load that skill. The difference is the harness response. A tool request executes opaque code and returns a result; a skill request reads transparent procedural content into the context window.

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

## [XI · Agentic Construct] Subagents {#subagents}

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

## [XII · The Harness Extended] MCP {#mcp}

**Definition:** MCP is a standardized protocol that lets a harness connect to external servers that expose tools, resources, or prompts. MCP does not give the model a new capability directly; the harness must register or inject what the server provides before it can appear in the context window or be requested by the model.

> **Examples:**
>
> - GitHub MCP server, filesystem MCP server, Playwright MCP server, PostgreSQL MCP server, Atlassian/Jira MCP server, and Terraform MCP server.

MCP is therefore not a new kind of model capability. It exists in the harness layer: it extends the harness's perimeter by moving authentication, portability, vendor coupling, permissions, and service ownership out of the local runtime and into a protocol relationship with another system.

Today, MCP-exposed tools are the most common use case, but the protocol is broader than tool delivery. An MCP server can expose tools, resources, and prompts; the harness can expose capabilities back to the server, such as LLM inference, workspace scope, and user input collection. Once registered or injected by the harness, these resolve into patterns this document has already described:

- An MCP tool, once registered by the harness, is a tool: schema enters context, implementation remains opaque, result returns as context.
- An MCP resource or prompt, once injected by the harness, is context: content added to the next model request, no different in kind from a local file, a system prompt fragment, or a skill body.

What MCP does not change is the harness's responsibility: deciding what to expose, what to execute, what to inject, and what reaches the model.

In practice, most harnesses fully surface MCP tools. Support for resources and prompts is uneven. Some expose them as attachable context and slash commands; others ignore them entirely. That variance is the point: the specification may define what can cross the boundary, but the harness decides what actually becomes available in the loop.

> **Key Takeaway:** MCP standardizes the boundary between a harness and external capability servers. It does not create a new model-side capability; what arrives over MCP still becomes a tool, context, or harness-mediated request before it reaches the model.

## [XIII · Synthesis] Implications {#implications}

By evaluating these constructs mechanically rather than metaphorically, several architectural tradeoffs become clear:

- **The harness and loop are the operating frame.** Agents, tools, skills, and subagents are useful only because a harness can run an agentic loop: construct model requests, parse requested actions, execute allowed work, and place results back into context.
- **MCP is not an agentic construct.** MCP is a delivery boundary. You can have a local tool, an MCP-delivered tool, or a skill that instructs the model to use an MCP-delivered tool.
- **Skills and tools compose rather than compete.** A tool exposes an operation the harness can execute. A skill exposes procedural knowledge the model can follow, often by invoking one or more tools. The reverse does not hold: tools have no mechanism to contain or invoke skills.
- **Subagents trade isolation for coordination cost.** A subagent provides a fresh context window and scoped toolset, but introduces delegation, summarization, and token overhead. It is useful when that isolation is worth more than the added cost.
- **Construct selection is an architectural decision.** Personas, runbooks, and teammates can be useful metaphors while sketching a system, but they do not define implementation boundaries. The mechanical questions are what the model can see, what the harness can execute, and what happens to context between model requests.
