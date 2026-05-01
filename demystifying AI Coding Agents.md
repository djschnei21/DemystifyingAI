# Demystifying AI Coding Agents: A Technical Research Report for IT Professionals

This report compiles vendor-neutral, first-principles research on how modern AI coding agents actually work, organized to support a presentation outline that builds from the underlying inference service up through full multi-agent orchestration. It is written for an audience that understands HTTP, REST, microservices, control loops (Kubernetes, Ansible), and CI/CD — and uses those concepts as anchor analogies throughout.

---

## 1. The LLM as a Hosted Service

### What is physically running

A large language model in production is **a set of binary weight files (tens to hundreds of GB) loaded into the VRAM of one or more GPUs**, sitting behind a network service. That service is almost always one of three families of inference engines, all of which expose an HTTP API and are conceptually equivalent for our purposes:

- **vLLM** — an open-source engine originally from UC Berkeley. Its key innovation is *PagedAttention*, which manages the KV (key/value) cache the way an OS manages virtual memory pages, allowing many concurrent requests to share GPU memory efficiently.
- **TGI (Text Generation Inference)** — Hugging Face's serving stack, built as a Rust router plus model server. Notable for prefix caching of long shared contexts (e.g., the same system prompt across many users) and for supporting multiple backends (TRT-LLM, vLLM, llama.cpp).
- **TensorRT-LLM** — NVIDIA's optimized engine for NVIDIA GPUs. Performs aggressive kernel fusion, quantization (FP8, FP4, INT4/INT8), and hardware-specific compilation. Often paired with NVIDIA Triton Inference Server for multi-model serving.
- Adjacent implementations: **SGLang**, **LMDeploy**, **llama.cpp** for CPU inference, and proprietary stacks behind hosted APIs like OpenAI, Anthropic, and Google.

These engines all do the same job: receive an HTTP/JSON request, run a forward pass through transformer weights on a GPU, and stream tokens back. They differ mainly in throughput, latency, supported quantizations, and memory-management cleverness.

### Why inference is stateless

Each forward pass is mathematically a pure function: `f(weights, input_tokens) → probability_distribution_over_next_token`. The weights are immutable at inference time (training is a separate, expensive offline process). The model has no internal mutable state that persists between requests. Any "memory" of a previous request must be supplied again, as text, in the next request.

The KV cache that engines like vLLM maintain is **not** a memory of past conversations. It is a per-request scratchpad of intermediate attention computations that lets the model avoid re-processing tokens it has already seen *within a single generation*. Engines may reuse cached prefixes across requests as a *performance optimization* (prompt caching), but this is invisible to the protocol: dropping the cache has no effect on correctness, only on cost and latency.

### Mapping to familiar IT concepts

An LLM endpoint is a stateless microservice that happens to do matrix multiplication on a GPU instead of database I/O on a CPU. Conceptually:

- It's behind a load balancer (often dozens or hundreds of GPU replicas).
- Each replica has the model loaded into VRAM at startup (cold-start can take minutes — analogous to a JVM warm-up but with 70 GB of weights to page in).
- Requests are routed to any healthy replica; sticky sessions are unnecessary because there is no per-session state.
- Horizontal scaling looks identical to any stateless service — except your "compute" is bound by GPU memory bandwidth and tensor cores, not CPU/RAM.

### What a single forward pass does

The model's atomic operation is **next-token prediction**:

1. **Tokenization** — input text is converted to integer token IDs (e.g., `"The cat sleeps" → [1012, 2305, 4512]`).
2. **Prefill** — the entire prompt is fed through the transformer in parallel. Every layer computes attention weights linking every token to every other token, building a KV cache. This is the latency component called *time to first token (TTFT)*; it grows roughly linearly with prompt length (with quadratic attention cost hidden inside).
3. **Decode** — the model emits a probability distribution over its entire vocabulary (typically 30k–200k tokens) for the *next* token.
4. **Sample** — a sampling step (controlled by `temperature`, `top_p`, `top_k`) picks one token from that distribution.
5. **Append and repeat** — the chosen token is appended to the sequence and step 3 repeats. Each iteration is one full forward pass at decode time, taking roughly 1–3 ms per token on modern GPUs.

Generation stops when the model emits an end-of-sequence token, when `max_tokens` is hit, or when a stop string matches. Crucially: **the model does not "plan" a whole answer.** It commits one token at a time. The illusion of coherent reasoning emerges from the conditional probability machinery and the fact that each sampled token is then visible in context for the next prediction.

---

## 2. The API Request, Conceptually

### Anatomy of a chat-completions request

Almost all production LLM APIs (OpenAI, Anthropic, vLLM's OpenAI-compatible server, TGI, Together, Databricks, etc.) accept a JSON POST that looks something like this:

```json
POST /v1/chat/completions
{
  "model": "some-model-id",
  "messages": [
    {"role": "system",    "content": "You are a careful coding assistant..."},
    {"role": "user",      "content": "Fix the failing test in auth.py"},
    {"role": "assistant", "content": "I'll start by reading the file."},
    {"role": "tool",      "tool_call_id": "...", "content": "<file contents>"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read a file from the workspace",
        "parameters": {  /* JSON Schema */ }
      }
    }
  ],
  "temperature": 0.2,
  "max_tokens": 4096,
  "stream": true
}
```

Standard fields:

- **`model`** — string identifier for which weights to invoke.
- **`messages`** — an ordered list of role-tagged turns: `system` (operator instructions), `user` (human input), `assistant` (model's prior outputs), and `tool` (tool execution results to be fed back in).
- **`tools`** — an optional array of tool definitions, each carrying a name, a natural-language description, and a JSON Schema describing its arguments.
- **Sampling parameters** — `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `frequency_penalty`, etc.

### Two kinds of response

The model returns either:

1. **A text completion** — a normal assistant message, typically signaling either an answer or a request for clarification.
2. **A structured tool-call request** — a JSON object stating "please call function `X` with arguments `Y`." Critically, the model itself does not invoke anything. It just emits a structured request. (Databricks documentation puts it bluntly: "The LLM itself does not call these functions, but instead it creates a JSON object that users can use to call the functions in their code.") A response can contain multiple tool calls (parallel tool-calling) on supported models.

A single response may also mix free text ("Let me check the file…") with one or more tool-call objects.

### Streaming vs non-streaming

- **Non-streaming** — the server waits until generation is complete and returns one JSON object. Easier to handle; higher perceived latency.
- **Streaming** — typically Server-Sent Events (SSE). Each token (or small token group) is emitted as a `data:` chunk as soon as it is produced. The harness assembles them client-side. Useful for UI responsiveness and for early termination, but tool-call assembly across chunks is fiddly.

### The API itself is stateless — and "memory" is an illusion

This is the single most important fact for the audience to internalize:

> **Every API call is independent. The full conversation history must be re-sent on every single request. The "memory" of a chat is maintained entirely on the client side (the harness).**

A chat with 50 turns sends all 50 turns on the 51st request. From the model server's perspective, request #51 is its first time seeing this conversation. As Armin Ronacher puts it in *LLM APIs are a Synchronization Problem*: the bytes you send each turn grow linearly, but cumulative bytes-on-the-wire over the conversation grow quadratically because each linear-sized history is retransmitted every step.

There are exceptions: OpenAI's *Responses API*, LM Studio's `/api/v1/chat`, and similar "stateful" endpoints store conversation state server-side and let you reference it via a `previous_response_id`. These are layered abstractions on top of the same stateless underlying inference — the server is doing the resending for you. They do not change the underlying physical reality.

This statelessness is a feature, not a bug:
- **Scalability** — any replica can handle any request.
- **Debuggability** — you can replay any turn by replaying its exact request body.
- **Cost & data control** — well-architected stateless APIs (e.g., zero-data-retention deployments) hold no user content after the response is returned, which simplifies GDPR and CLOUD Act analysis.
- **Caching is opt-in and transparent** — prefix caching is a server-side optimization, not a hidden state-sharing channel.

### Token limits and context windows — what they actually mean

The **context window** is the maximum sequence length (in tokens) the model can attend to in a single forward pass. It is a hard architectural property fixed at training time: the positional-encoding scheme (typically RoPE — Rotary Position Embeddings) and the position indices used during pretraining define what the model has ever "seen." Beyond that length, position embeddings are out-of-distribution and quality collapses.

Practical points for the audience:

- The window is a *shared budget* covering system prompt + conversation history + retrieved documents + tool definitions + the model's output. Output and input both compete for the same tokens.
- Self-attention scales **quadratically** with sequence length: doubling context roughly quadruples the attention compute. This is why long-context models are expensive.
- **Effective context ≪ advertised context.** Empirical research (e.g., HKU's STRING paper, plus widely-replicated "lost-in-the-middle" findings) shows most open models are reliable across only ~50% of their advertised window, with the strongest attention concentrated at the very beginning and the very end. A 128k-token model might give you 20–40k tokens of *useful* context.
- A token is roughly 0.75 English words (varies by tokenizer).
- Token cost is the dominant runtime expense for agents — and because input tokens are re-sent each turn, costs compound.

---

## 3. The Harness — What It Actually Is and Does

This is the section that most demystifies the field. **The "AI coding agent" is not the model.** The model is a hosted service somewhere on the internet. The agent is a *program running on your machine* — the **harness** — that orchestrates calls to that service.

### The minimal agent loop

Anthropic's Hannah Moran defined an agent at an Anthropic developer conference (May 2025) as "**LLMs autonomously using tools in a loop**." That is the entire core. The pseudo-code is six lines long:

```
history = [system_prompt, user_message]
while True:
    response = llm_api.call(model, history, tools=available_tools)
    history.append(response)
    if response.has_tool_calls:
        for call in response.tool_calls:
            result = execute_locally(call)
            history.append({"role": "tool", "content": result})
    else:
        return response.text  # final answer
```

Everything else — context management, safety, parallelism, sub-agents, skills — is engineering on top of this skeleton.

### What the harness is responsible for

The harness owns *all* of the state and side effects. Specifically:

1. **System prompt assembly** — composing identity, behavior rules, environment description, and policy from multiple sources (built-in template, project config like `CLAUDE.md`/`AGENTS.md`, skill metadata, tool list).
2. **Context curation** — deciding what files, snippets, and prior turns to include. This is now widely called **context engineering** (Anthropic's term) and is considered the dominant lever for output quality.
3. **Tool registration** — converting available capabilities into JSON-Schema tool definitions and passing them in the request.
4. **Tool execution** — when the model emits a tool-call request, actually running the bash command, opening the file, fetching the URL, etc., on the user's machine, container, or sandbox.
5. **Conversation-history management** — appending tool outputs as `tool` messages, truncating or compacting older turns when the window fills, optionally writing summaries.
6. **Permissioning and sandboxing** — gating which tool calls require user approval, blocking dangerous commands, isolating filesystem changes (e.g., Claude Code's git-worktree isolation).
7. **Error and retry handling** — catching tool failures, returning structured errors back to the model so it can adjust, and handling transient API errors with backoff.
8. **Verification loops** — running tests, linters, type-checkers automatically and feeding their output back in (Martin Fowler's "harness engineering" framing distinguishes *guides* — feedforward controls — from *sensors* — feedback controls).
9. **Sub-agent dispatch** — spawning child agents and merging their outputs.
10. **Telemetry and tracing** — recording every turn for debugging and evaluation.

### Mapping to familiar IT orchestration patterns

- **Kubernetes controller** — observe (gather context) → diff (model decides what to change) → reconcile (execute tool). The agent loop is Boyd's OODA loop (Observe, Orient, Decide, Act) at machine speed; multiple sources draw the parallel explicitly.
- **CI/CD pipeline runner** — a workflow runner that, instead of executing a static YAML graph, asks an LLM at each step what to do next.
- **Ansible** — the playbook (skill) declares intent; the runner (harness) executes modules (tools); the controller node (model) plans tasks based on facts gathered from the target. Skills are the closest direct analog to Ansible *roles*.
- **Webhook/RPC dispatcher** — the model emits structured events; the harness routes them to handlers.

The Daily Dose of DS "Anatomy of an Agent Harness" piece captures this neatly: "The runtime is a 'dumb loop.' All intelligence lives in the model." Claude Code's variant is described as a *Gather–Act–Verify* cycle. VS Code's docs describe agent mode as *Understand → Act → Validate*.

### Why this distinction matters

> **The model is a commodity. The harness is the product.**

This shows up in benchmarks: same underlying model, different harnesses, very different SWE-bench scores. Academic work on the open-source OpenDev terminal agent explicitly frames the design as a "compound AI system": separation of concerns between scaffolding, harness, context engineering, and the LLM, with the LLM swappable via configuration.

### Examples of harnesses in the wild (vendor-neutral framing)

All of the following are *different harnesses* that call largely overlapping LLM APIs underneath:

- **Claude Code** — terminal-native, opinionated multi-tool harness from Anthropic. ~19–40 permission-gated tools depending on configuration. Pioneered many of the patterns now widespread (sub-agents, skills, hooks).
- **Cursor** — IDE-embedded harness; large investment in tuned system prompts and "apply" models that translate model outputs into precise edits.
- **GitHub Copilot agent mode** — VS Code-integrated; recent versions also host third-party harnesses (Anthropic's Claude harness, OpenAI's Codex harness) side-by-side under the same UI.
- **Codex CLI** — OpenAI's terminal harness.
- **Aider** — open-source CLI; deep Git integration (auto-commits, repo maps).
- **Cline** (formerly Claude Dev) — VS Code extension; step-by-step approval UX; MCP-native.
- **Continue.dev**, **Goose** (Block), **OpenDevin / OpenHands**, **Devin** (cloud agent), **Gemini CLI**, **Codex Cloud**, etc.

The split worth highlighting: **pair-programming harnesses** (Aider, Codex CLI, basic Cline) assume a developer at the keyboard giving direction; **agent-orchestrator harnesses** (Claude Code, Devin, agent-mode Cline) assume long-running autonomous execution.

---

## 4. Tools / Function Calling

### The model never executes anything

This is the cleanest demystification. A "tool" is just a function definition you describe to the model in JSON. When the model "uses" a tool, it emits a JSON object naming the tool and supplying arguments. **The harness sees that JSON and runs the actual code.** The tool result (stdout, file contents, HTTP response, etc.) is then appended to the conversation history as a `tool` message and sent back to the model on the next API call.

Anthropic's docs phrase the contract precisely: "Tool use is a contract between your application and the model. You specify what operations are available and what shape their inputs and outputs take; Claude decides when and how to call them. The model never executes anything on its own."

### How tool definitions are conveyed

Tool schemas ride along in the request as JSON Schema. A typical definition:

```json
{
  "type": "function",
  "function": {
    "name": "run_bash",
    "description": "Run a bash command in the project workspace and return stdout/stderr.",
    "parameters": {
      "type": "object",
      "properties": {
        "command": {"type": "string", "description": "The bash command to run."},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 600}
      },
      "required": ["command"]
    }
  }
}
```

The `description` field is genuinely important — it is part of the prompt; the model uses it to decide *when* to invoke the tool. Tool selection is fundamentally a soft, language-driven decision.

### How models are made to call tools reliably

Two complementary techniques:

- **Fine-tuning** — frontier models are trained on large datasets of tool-call traces so they emit well-formed JSON when given tool definitions.
- **Constrained decoding / structured output** — at the inference engine level (vLLM, TGI, TensorRT-LLM all support this), the next-token logits can be masked at every step so only tokens consistent with the active JSON Schema are sampleable. Baseten's writeup on Triton-based logit biasing is a good reference: a state machine over the schema computes a token mask, which is applied during sampling. The model literally cannot emit invalid JSON for a given schema. Libraries like Outlines, Instructor, and BAML provide higher-level wrappers; some (BAML's SAP — Schema-Aligned Parsing) handle parsing more permissively.

### Common tools in coding agents

Across Claude Code, Cursor, Aider, Cline, Codex, etc., the recurring tool surface is small and predictable:

- `read_file(path)`
- `write_file(path, content)` / `apply_edit(path, diff)` (often using a structured diff format)
- `list_directory(path)` / `glob(pattern)`
- `grep(pattern, path)` / semantic code search
- `run_bash(command)` / shell execution
- `web_fetch(url)` / `web_search(query)`
- `run_tests` / `run_linter`
- Subagent dispatch (e.g., Claude Code's `Task` tool)
- Optionally: image inspection, browser automation, MCP-bridge calls

Claude Code has roughly 19 permission-gated tools in the standard configuration, with tool counts reaching 40+ when MCP servers are added.

### Mapping to familiar IT concepts

- **RPC** — function-calling is essentially RPC where the dispatcher is a language model. The schema is the IDL, JSON is the wire format, the model is the caller, the harness is the server.
- **Webhooks** — the model emits an event ("call this function with these args"); the harness is the receiver that performs the side effect.
- **Stored procedures vs application code** — the model is the planner, the harness is the executor. The model decides *what*; the harness controls *how* and *whether*.

---

## 5. Skills

### The newest piece of the puzzle

**Agent Skills** are a packaging format introduced by Anthropic in October 2025 and released as an open standard on December 18, 2025. Within weeks, OpenAI (Codex CLI, ChatGPT), Google (Gemini CLI), GitHub Copilot, and Cursor all adopted the format. By February 2026, public skill marketplaces indexed 40,000+ skills (Bosch Research / CMU analysis, arXiv:2602.08004).

### What a skill actually is

A skill is **a folder on disk** containing a `SKILL.md` file with YAML frontmatter, plus optional supporting files:

```
my-skill/
├── SKILL.md            # required: YAML frontmatter + markdown instructions
├── scripts/            # optional: executable Python, Bash, or JS
├── references/         # optional: detailed docs loaded on demand
└── assets/             # optional: templates, schemas, data files
```

A minimal `SKILL.md`:

```markdown
---
name: pdf-processing
description: Extract text and tables from PDF files, fill forms, merge documents.
  Use when working with PDFs or when the user mentions PDFs, forms, or document
  extraction.
---
# PDF Processing
## Quick start
Use pdfplumber to extract text from PDFs:
```python
import pdfplumber
with pdfplumber.open("document.pdf") as pdf: ...
```
For form filling, see forms.md. For advanced use, see reference.md.
```

### Progressive disclosure — the architectural trick

Skills are loaded in three stages, *only paying for context as needed*:

1. **Discovery** — at startup, only the `name` and `description` from the YAML frontmatter are injected into the system prompt. Median per-skill cost is ~80 tokens (across Anthropic's official skills, 55 to 235). You can install hundreds of skills without flooding the window.
2. **Activation** — when the model judges that a skill matches the task, it reads the full body of `SKILL.md` (typically a few thousand tokens).
3. **Execution** — only if needed, the model loads referenced files (`references/forms.md`) or runs bundled scripts.

Some practitioners argue "progressive *discovery*" is the better term, since the model is actively searching, not the system pushing.

### Skills vs tools vs prompts

| | What it is | When loaded |
|---|---|---|
| **Tool** | A callable function with a JSON-Schema signature | Always (definitions in every request) |
| **Skill** | A bundle of instructions + scripts + references | Lazily, when description matches |
| **System prompt** | Static identity/behavior rules | Always, at the top |

A skill can *reference* tools but isn't itself a tool. Conceptually: **tools are verbs; skills are know-how; prompts are policies.**

### Mapping to familiar IT concepts

Several useful analogies for the audience:

- **Ansible roles** — a directory of tasks, files, templates, and metadata that gets loaded by name. Almost a 1:1 mapping.
- **Man pages** — the model knows a man page exists (it sees the name and one-liner) and chooses whether to `man pdf-processing` for the full text.
- **Loadable kernel modules** — small headers always present; large modules only paged in when needed.
- **Plugin index + lazy loading** in IDEs.
- **Intensional vs extensional databases** — the `SKILL.md` is an intensional index over an extensional reference set (the bundled files).

### Why this format won so fast

- It's **just a directory of markdown** — version-controllable in Git, editable by non-engineers.
- It encodes **discovery, activation, and execution in plain text**, not framework-specific code.
- It collapses a previously-fragmented category — prompt registries, prompt-management SaaS, custom retrieval pipelines — into one filesystem-native primitive that any agent runtime can implement.

---

## 6. Agents and Sub-Agents

### "Agent" demystified

An agent, per Anthropic's working definition, is "a model using tools in a loop, deciding its own trajectory." That's it. The buzzword does not denote anything more magical than the loop in section 3.

It's worth noting Anthropic's distinction between *task*, *workflow*, and *agent*:

- **Task** — a single model call (summarize this email).
- **Workflow** — multiple model calls in a *predefined* control flow (you decide steps; the model fills nodes).
- **Agent** — a model in a loop where the *model* decides the trajectory.

### Sub-agents

A **sub-agent** is what you get when the parent harness spawns a fresh instance of the loop with:

- A **separate, isolated context window**.
- A **scoped task** described by the parent.
- Possibly a **reduced tool set** and stricter permissions.
- Possibly a different model.

Mechanically, this looks like an OS spawning a child process: the parent passes initial state (the task), the child runs its own loop until completion, and only the final summary is returned to the parent's context. Intermediate noise — verbose tool output, exploratory reads — stays in the child and never bloats the parent's window.

In Claude Code, sub-agent dispatch is handled by a `Task` tool (or, in the newer fork-mode, `CLAUDE_CODE_FORK_SUBAGENT=1`). Up to ~10 sub-agents can run in parallel within a single Claude Code session.

### Why you'd want sub-agents

1. **Context isolation** — running tests can produce 50k tokens of output you don't need in the main window. Delegate to a child whose only job is to run them and report the failures.
2. **Parallelism** — investigate authentication, database, and API modules simultaneously and synthesize.
3. **Specialization** — a "code-reviewer" sub-agent with read-only tools and a very specific system prompt (no Edit/Write); a "test-runner" sub-agent that can't touch source code.
4. **Cost reasoning** — Anthropic has published rough scaling: a chat call is 1× tokens, a single-agent loop is ~4×, a multi-agent system is ~15×. Use sub-agents when the gain justifies the cost.

### Common orchestration patterns

- **Manager / planner-executor** — a top-level agent decomposes the task and delegates pieces (used by OpenAI's Agents SDK).
- **Orchestrator-worker (split-and-merge / map-reduce)** — fan out independent subtasks, merge results (used by Anthropic's Research feature).
- **Handoff** — peers transfer control by specialty.
- **Builder-validator chain** — one agent writes, another reviews.
- **Agent teams** — multiple long-lived agents communicating via shared task lists.

### Trade-offs and constraints

- Sub-agents cannot easily share context with the parent except through the explicit return value. This is a known anti-pattern when the parent needs to see the file the child wrote (community calls this the "Implementer Model" problem).
- They consume parallel API quota and parallel tokens — for a team running 10 sub-agents you are running 10 inference streams concurrently.
- Coordination overhead grows nonlinearly; nesting beyond 1–2 levels rarely pays off.

### Mapping to familiar IT concepts

- **Spawning a child process** with a constrained env and scoped permissions.
- **Worker queues** — main agent enqueues tasks, workers process in parallel, results flow back.
- **Microservices calling microservices** — except every "service" is the same LLM with different prompts and tools.
- **Map-reduce** — the merge step is performed by the orchestrator's next LLM call.

---

## 7. Putting It All Together — A Concrete Walkthrough

Trace a single user request: **"Fix the failing test in auth.py"**

1. **Harness construction.** The harness assembles a single API request payload:
   - System prompt: identity, behavior rules, environment description (CWD, OS, git status), Skill metadata (~80 tokens × N installed skills).
   - Tool definitions: `read_file`, `write_file`, `run_bash`, `grep`, `apply_edit`, `task` (subagent), MCP-exposed tools, etc., all as JSON Schema.
   - Conversation: `[system, user("Fix the failing test in auth.py")]`.
   - Sampling params: `temperature=0.2`, `max_tokens=4096`, `stream=true`.
2. **`POST /v1/messages`** to the LLM API.
3. **Model response.** A tool call: `read_file(path="auth.py")`. The harness sees this in the SSE stream, parses it, and pauses generation.
4. **Tool execution.** The harness reads `auth.py` from disk (perhaps after prompting the user for permission, depending on policy). It captures the content.
5. **Append result.** A new `tool` message containing the file content is appended to the in-memory history.
6. **Second API call.** The harness re-sends the *entire* updated history. The model sees the file and emits another tool call: `run_bash(command="pytest tests/test_auth.py -x")`.
7. **Tool execution.** The harness runs pytest in the sandbox, captures stdout/stderr (1,200 tokens of traceback), and appends it as a tool message.
8. **Possible sub-agent dispatch.** If the test output is huge, the model may call the `task` tool: "Investigate the traceback and identify the root-cause line; return a one-paragraph summary." A child agent is spawned with isolated context, runs its own mini-loop, and returns a 200-token summary that the parent appends instead of the raw output.
9. **Skill activation.** Suppose a `python-debugging` skill exists. The model, recognizing the task fits its description, emits `read_file("python-debugging/SKILL.md")`. The harness loads the full skill body. Now the model has procedural guidance ("first reproduce, then narrow, then patch").
10. **Apply edit.** Model emits `apply_edit(path="auth.py", diff=…)`. Harness validates the diff, optionally requests user confirmation, applies it.
11. **Verify.** Model emits another `run_bash("pytest")`. Harness runs it. Tests pass. Output appended.
12. **Final response.** Model emits a plain assistant message — no tool call — summarizing the fix. The loop terminates and the harness surfaces the summary to the user.

The N round-trips to the LLM API are all stateless. Each one re-sends an ever-growing history. The harness, the disk, the sandbox, and any sub-agents are where state lives.

---

## 8. Where the Intelligence Actually Lives

The audience's biggest takeaway should be **the division of responsibilities**:

| Layer | Responsibility | Property |
|---|---|---|
| **Model (weights on a GPU)** | Pattern-matching, code understanding, deciding the next token / next tool call / final answer | Stateless, commoditizing, swappable via API |
| **Harness (program on disk)** | State management, history replay, tool execution, sandboxing, retry, sub-agent dispatch, observability | Where all the engineering and most of the bugs live |
| **Skills + system prompt** | Domain knowledge injection, behavior shaping, policy enforcement | Versioned, code-reviewable, composable |
| **Tools + MCP servers** | Side effects on the world (filesystem, shell, network, APIs) | Where security boundaries are enforced |

### Why this composability matters

- **Models are swappable.** Drop in a different model behind the same harness and most things keep working. SWE-bench leaderboards show the harness contribution is large and persistent.
- **Harnesses are buildable.** A capable agent loop fits in a few hundred lines of Python (Temporal, LangGraph, OpenAI Agents SDK, smolagents all expose this). You can build your own for an internal use case.
- **Skills and prompts are forkable.** A team can clone a skill, tweak it, version it in Git, and redeploy without touching either the model or the harness.
- **The "AI agent" is a compound system, not a monolith.** Treat it like a 4-tier app: storage layer (history), orchestration layer (harness), policy layer (prompts/skills), and a remote inference service.

---

## Cross-Cutting Topic: The Model Context Protocol (MCP)

### What MCP is

**MCP** (Model Context Protocol) is an open protocol introduced by Anthropic in November 2024 that standardizes how harnesses (clients) discover and call external tools and data sources (servers). Inspired by the Language Server Protocol (LSP), it solves the "M × N integration" problem: instead of every harness writing custom integrations to every system, both sides implement MCP and any compliant client works with any compliant server.

### Architecture

- **Host** — the user-facing application (Claude Desktop, Cursor, VS Code, ChatGPT, etc.).
- **Client** — an MCP client embedded in the host; one client per server.
- **Server** — a process exposing tools, resources (read-only context), and prompts via the protocol. Can run locally or remotely.

### Wire format

- **JSON-RPC 2.0** for all messages.
- **Stateful sessions**: an `initialize` handshake negotiates protocol version and capabilities; the connection then carries requests, responses, and notifications until closed.
- **Transports**: 
  - **stdio** — host launches the server as a subprocess, wires stdin/stdout. Best for local tools.
  - **Streamable HTTP** (the modern transport, replacing the older HTTP+SSE design) — server-sent events for streaming, HTTP POST for client-to-server.
- **Auth** — for remote servers, OAuth 2.1 with dynamic client registration, plus protected resource metadata.

### Primitives

- **Tools** — callable functions with JSON-Schema input/output (the same shape as native function-calling).
- **Resources** — read-only context (files, DB rows, API documents) the model can reference.
- **Prompts** — server-supplied prompt templates the host can offer.
- **Sampling** — server can request the host to perform an LLM call on its behalf (for agentic workflows initiated server-side).

### How it fits into the agent picture

MCP is essentially a **plug-in bus for tools**. From the harness's point of view, an MCP-discovered tool is indistinguishable from a built-in tool: the schema is fetched from the server during initialization and passed through to the model in the next API call. When the model emits a tool call, the harness routes it back through the MCP client to the appropriate server.

Practical implication: most major agent products (Claude Code, Cursor, Cline, Goose, ChatGPT, Codex CLI, Gemini CLI) consume MCP servers natively, and SaaS vendors (Atlassian, Cloudflare, Notion, Stripe, Sentry, Zapier, Figma, etc.) ship official MCP servers. The protocol has effectively become the integration layer for the agent ecosystem in roughly the same way LSP became one for IDEs.

### Caveats

- Implementing a robust MCP server is currently rough — the spec has changed several times during 2025; documentation has been criticized as terse; the SSE/Streamable HTTP transition broke some serverless deployments.
- **Security is a real concern** — tools represent arbitrary code execution; tool descriptions are *part of the prompt* and can be used for prompt-injection attacks (a malicious MCP server can craft a description that subverts the model's behavior). The spec explicitly says clients SHOULD treat tool annotations as untrusted unless the server is trusted. OpenAI and Anthropic both publish security guidance recommending OAuth, scope limits, and human-in-the-loop confirmation for high-impact tool calls.

---

## Cross-Cutting Topic: Why Stateless API + Stateful Harness Matters

The split has direct, practical consequences across security, cost, and debugging — all areas an IT audience will care about.

### Security

- **Auditability** — every model interaction is a complete, replayable JSON document. There's no hidden server-side memory. Logging the request bodies gives you a full trace.
- **Data residency / compliance** — with zero-data-retention deployments, no user content persists past the response. The harness controls all storage.
- **Prompt-injection blast radius** — because the harness controls what's in the prompt, it is the layer where input sanitization, tool gating, and output validation must happen. Schneier's *OODA Loop Problem* analysis is the canonical reference: the riskiest layer in agentic AI is the harness's *Observe* step ingesting untrusted content (web pages, file contents, MCP tool outputs).
- **Permission enforcement** — Claude Code's design separates model reasoning from tool execution: the model decides what to attempt, the tool system decides what's allowed. This is the right architectural pattern: never trust the model with privileged action; gate on the harness side.

### Cost

- Token costs scale **linearly with each turn but quadratically over a conversation** because the full history is re-sent each time. Long sessions get expensive fast even if the model is fast.
- **Prompt caching** at the inference engine level (vLLM, TGI v3, Anthropic's prompt caching, OpenAI's automatic caching) can reduce prefill cost by an order of magnitude when prefixes are stable — but is invisible behavior; design for it but don't depend on it.
- **Sub-agents and verification loops are the dominant cost knobs.** Anthropic's published numbers (1× / 4× / 15×) are useful planning anchors.

### Debugging

- Reproducibility is excellent at the API level (deterministic with `temperature=0`; replayable with the exact request body) but poor at the system level (filesystem, network, time-of-day all differ across runs).
- The hardest bugs are *context-window pathologies* — important info pushed out by truncation, "lost-in-the-middle" forgetting around the 50% mark of a long context, or context contamination from earlier tool failures.
- Good harnesses log every API request/response, every tool call/result, and every sub-agent transcript. This is non-negotiable in production.

---

## Cross-Cutting Topic: Common Misconceptions to Address

A vendor-neutral demystification talk should explicitly call out:

1. **"The agent is the AI."** No — the model is one stateless function call away; the agent is the program orchestrating it.
2. **"The model remembers our conversation."** No — your harness re-sends the entire history every turn. The model has no continuity beyond the bytes of the current request.
3. **"The model executes my tools."** No — the model emits a structured request; your code runs the tool. The model can ask for rm -rf /, but only your harness can run it.
4. **"Bigger context window = better." ** Diminishing returns are real. Effective context is typically a fraction of advertised context; quadratic attention cost grows fast; tokens compete for attention.
5. **"More agents are always better."** Multi-agent systems are ~15× the token cost of a single chat. Use them when you genuinely need parallelism, isolation, or specialization — not because they sound impressive.
6. **"Skills are tools" / "tools are skills."** Skills are instruction+resource bundles loaded by description match; tools are JSON-Schema functions called by name. Different mechanisms, different lifecycles.
7. **"MCP is a model thing."** MCP is a protocol between hosts and external services. The model never speaks MCP directly — the harness translates between MCP and the model's tool-calling API.
8. **"If I subscribe to a chat product, I'm paying for the model."** When you build your own agent, you pay per-token at API rates and you bear the runaway-loop risk. Cost engineering is the new thing the AI engineering profession exists to do.
9. **"LLMs are deterministic if I set temperature to 0."** Mostly, but kernel-level non-determinism (CUDA atomics, batching) means even temp-0 calls can vary slightly across replicas. Don't depend on bit-for-bit reproducibility.
10. **"Hallucinations are a model problem."** Often they are a *context* problem — the model wasn't given the file it needed, or was given conflicting information, or important info fell out of the window. Most "hallucination" complaints in coding agents are root-caused in the harness's context-curation logic.

---

## Useful Practitioner Analogies (Curated)

Concrete mappings the audience already knows, ready to drop into slides:

- **The model is a stateless Lambda function. The harness is the surrounding orchestration code.**
- **The agent loop is Kubernetes' reconciliation loop with a language model in the diff step.**
- **Tool calling is RPC where the language model is the dispatcher and JSON Schema is the IDL.**
- **Function calling is a webhook system where the LLM is the decision-maker emitting events and the harness is the event handler executing them.**
- **Skills are Ansible roles for agents — discoverable on disk, loaded by name, bundling instructions + scripts + assets.**
- **Progressive disclosure of skills works like a `man` index: the model sees titles and one-liners, then "reads the man page" only when it needs to.**
- **Sub-agents are child processes with isolated address spaces. The parent gets the exit summary, not the stack traces.**
- **MCP is LSP for AI tools** — Anthropic explicitly invokes the analogy.
- **A coding agent is "models using tools in a loop"** (Anthropic's official one-liner — quote it).
- **Context engineering is the new prompt engineering. The model is a CPU, the context window is RAM, the harness is the OS deciding what to page in.**
- **LLM APIs are a synchronization problem, not a request/response problem** (Ronacher) — useful when explaining why long agent sessions get weird.
- **The runtime is a "dumb loop"; all intelligence lives in the model** (Daily Dose of DS) — useful summary slide.

---

## Suggested Narrative Flow for the Presentation

The eight-section arc the user requested already lands the demystification cleanly. To strengthen the IT-audience framing, the presentation can hammer one through-line on every slide:

> *Everything we are about to look at is either (a) a stateless service over HTTP, (b) a control loop on a workstation, or (c) JSON Schema between the two. There is no magic.*

Open with a slide that strips away marketing language and shows just three boxes: **Model (GPU service, stateless)** ↔ **Harness (your machine, stateful)** ↔ **Tools (filesystem, shell, network, MCP)**. End with the same three boxes annotated with where the engineering, the cost, and the security live. Everything in between — sections 1–8 — fills in the details.

The audience will leave with the right mental model: the agent is not a magical entity but a familiar shape — a stateless inference microservice consumed by a stateful orchestration program that calls out to the same kinds of tools they already manage in Kubernetes, Ansible, and CI/CD.