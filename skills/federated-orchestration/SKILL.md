---
name: federated-orchestration
description: |
  Coordinate work across multiple servers, machines, repos, or disconnected workspaces when the whole project cannot live in one folder. Use this skill whenever the user mentions different servers, remote machines, split codebases, cross-workspace integration, handoff between assistant sessions, relay workflows, SSH/Git forwarding, or says they cannot put everything in one directory / 不能放在一个文件夹 / 不同服务器 / 多台机器 / 需要跨机器协作 / 需要交接上下文. This skill standardizes a shared coordination layer with contracts, work items, handoffs, patch bundles, and environment manifests so local assistant sessions can collaborate without direct shared filesystem access.
---

# Federated Orchestration

Use this skill when the real problem is not "write code in one repo" but "coordinate work across several machines that each have only part of the code or runtime."

## Core Idea

Do not pretend multiple servers are one workspace.

Instead, treat each server as a local execution island and connect them through a **coordination workspace** that stores:
- server manifests
- workspace summaries
- interface contracts
- task/work-item definitions
- patch manifests and patch bundles
- handoff notes
- test/build reports

This skill is about **protocol and workflow**, not magical cross-server file access.

## Default Architecture

Prefer this default unless the user explicitly wants something else:

1. Keep the real code on each server.
2. Create one small coordination workspace tracked with Git when possible.
3. Share **key interface mirrors**, not full code mirrors.
4. Keep large weights, datasets, and environment-specific artifacts local.
5. Share only metadata, commands, contracts, and outputs needed for integration.

Use full code mirroring only when the user explicitly prefers it and the repo size/runtime constraints make it practical.

## Non-Negotiable Rules

1. Never claim to have read remote code that is not in the current workspace.
2. Never invent missing interfaces; create or update a contract instead.
3. Never ask one server-local assistant session to reason from the entire distributed project if a focused contract or summary is enough.
4. When network access is partial, model that explicitly with relay steps instead of treating sync as implicit.
5. Keep coordination artifacts compact and structured so future sessions can resume without reloading entire repos.

## Choose a Role First

At the start of the task, determine which role applies:

- `coordinator`: Owns the coordination workspace, updates contracts, creates work items, aggregates results, decides next actions.
- `worker`: Operates inside one server-local codebase, reads a work item, changes local code, emits a patch bundle and handoff.
- `relay`: Moves coordination artifacts through a reachable machine when a server cannot access the shared Git remote directly.

If the user did not state the role, infer it from the task and say which role you are acting as.

## Startup Workflow

### 1. Check whether a coordination workspace already exists

Look for the standard directories:

```text
contracts/
handoffs/
patches/
reports/
servers/
work-items/
```

If it does not exist, initialize it with:

```bash
python scripts/init_coordination_workspace.py <path> --server-id <server-name>
```

Read `references/protocol.md` before making changes to the workspace structure.

### 2. Materialize the current machine state

Every participating server should maintain:
- `servers/<server_id>/env-manifest.json`
- `servers/<server_id>/workspace-summary.md`

The manifest is the machine-readable truth about repos, runtimes, artifacts, and connectivity limits.
The summary is the human/assistant-readable distillation of the local workspace.

### 3. Lock contracts before broad implementation

If the task changes behavior across servers, create or update a contract in `contracts/` before asking workers to implement.

Use contracts for:
- request/response schemas
- CLI or config shapes
- adapter boundaries
- environment expectations
- sample inputs/outputs

## Coordinator Workflow

Use this role when the user is integrating distributed work or wants a plan that several server-local sessions can execute.

Sequence:

1. Read relevant server manifests and workspace summaries.
2. Identify which machine owns each code or runtime dependency.
3. Create or update the minimal contract set needed for the change.
4. Write one `work-items/<task_id>.json` per independently assignable unit of work.
5. For each work item, specify owner server, dependencies, acceptance checks, local commands, and expected outputs.
6. As worker outputs arrive, review `patch-manifest.json`, reports, and handoffs.
7. Update task status and either merge, redirect, or issue follow-up work items.

Coordinator outputs should stay explicit and bounded:
- which server owns the next step
- which files in the coordination workspace changed
- whether contracts are stable or still provisional
- what is blocked by missing access, missing artifacts, or unresolved API decisions

## Worker Workflow

Use this role when you are inside one server-local workspace doing real code work.

Sequence:

1. Read the assigned work item and only the contracts relevant to it.
2. Read the local repo just enough to implement the task safely.
3. Make local changes only in the server-owned codebase.
4. Run local checks that the work item requires.
5. Emit:
   - a patch or diff bundle
   - `patches/<task_id>/patch-manifest.json`
   - `handoffs/<task_id>/<timestamp>.md`
   - any referenced report files in `reports/`
6. If the task cannot proceed because the contract is wrong or incomplete, stop coding and update the contract or escalate back to the coordinator.

Worker mode should never depend on seeing the full distributed system.

## Relay Workflow

Use this role when a server cannot directly push or pull the coordination workspace.

Sequence:

1. Treat the coordination workspace as transportable data.
2. Move only the updated coordination artifacts and patch bundles through the relay path.
3. Preserve provenance: source server, timestamp, base commit, and transferred files.
4. Update status once the relay has completed so other sessions do not wait on invisible work.

Relay mode is a transport concern. Do not mutate contracts or task intent unless explicitly asked.

## What to Load From References

Read `references/protocol.md` when you need:
- directory layout
- JSON field definitions
- file responsibilities
- default rules for key interface mirrors

Read `references/templates.md` when you need:
- starter content for `workspace-summary.md`
- starter content for contracts
- starter content for handoffs
- starter content for work items

## Key Interface Mirrors

When the user cannot or should not mirror full code, share only the smallest slices that unblock coordination:
- repo tree excerpts
- public function/class signatures
- config keys
- entrypoint commands
- adapter stubs
- sample input/output payloads
- artifact manifests for weights and datasets

This is the default compromise between "thin coordination repo" and "full code mirror."

## Output Expectations

When using this skill, keep the response practical. State:
- your role
- which coordination artifacts you created or updated
- which local or shared files must be read next
- what is still unresolved

Prefer structured artifacts over long prose.

## Failure Handling

If something is missing:

- Missing contract: create one before implementation continues.
- Missing base commit: record the uncertainty in the patch manifest.
- Missing network path: switch to relay mode and describe the transfer boundary.
- Missing runtime artifact: add it to the artifact manifest; do not fake reproducibility.
- Conflicting server outputs: compare against the contract first, not against memory.

## References

- `references/protocol.md` - Coordination workspace structure and file schemas
- `references/templates.md` - Canonical starter templates
- `scripts/init_coordination_workspace.py` - Deterministically initialize a coordination workspace
