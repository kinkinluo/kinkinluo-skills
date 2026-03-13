# Coordination Protocol

This reference defines the default coordination workspace for distributed assistant work.

## Directory Layout

```text
coordination-workspace/
├── README.md
├── .gitignore
├── contracts/
├── handoffs/
├── patches/
├── reports/
├── servers/
└── work-items/
```

Use this layout unless the user already has an established equivalent.

## Directory Responsibilities

- `contracts/`: Shared truth about interfaces, schemas, commands, environment assumptions, and adapter boundaries.
- `handoffs/`: Session-to-session summaries. One task can have many handoffs over time.
- `patches/`: Patch bundles and patch manifests emitted by worker sessions.
- `reports/`: Build/test summaries or structured execution results.
- `servers/`: Per-server environment manifests and workspace summaries.
- `work-items/`: Task definitions, ownership, dependencies, acceptance checks, and status.

## Required Files

### `servers/<server_id>/env-manifest.json`

Purpose: machine-readable facts about a server-local workspace.

Recommended shape:

```json
{
  "server_id": "train-gpu-01",
  "repos": [
    {
      "name": "model-service",
      "path": "/srv/model-service",
      "default_branch": "main",
      "base_commit": ""
    }
  ],
  "entrypoints": [
    {
      "name": "trainer",
      "command": "python train.py --config configs/train.yaml"
    }
  ],
  "runtimes": ["python3.11", "cuda-12.1"],
  "artifacts": [
    {
      "name": "llama-weights",
      "kind": "weights",
      "location": "/mnt/models/llama",
      "share_mode": "manifest-only"
    }
  ],
  "restricted_network": true,
  "relay_required": true,
  "notes": ""
}
```

### `servers/<server_id>/workspace-summary.md`

Purpose: compact summary that lets another assistant session reason about the local workspace without loading the entire repo.

It should include:
- local repositories and ownership
- important entrypoints
- key interfaces exposed to other servers
- local commands for build/test/run
- large artifacts and why they stay local
- minimal file set to read for the next task

### `work-items/<task_id>.json`

Purpose: the authoritative task packet for one assignable unit of work.

Recommended shape:

```json
{
  "task_id": "TASK-001",
  "goal": "Add an adapter so the coordinator can call the ranking service with the new payload.",
  "owner_server": "rank-api-01",
  "depends_on": ["CONTRACT-rank-request-v2"],
  "input_contracts": ["contracts/rank-request-v2.md"],
  "expected_outputs": [
    "patches/TASK-001/rank-api.patch",
    "patches/TASK-001/patch-manifest.json",
    "handoffs/TASK-001/2026-03-13T10-00-00Z.md"
  ],
  "acceptance_checks": [
    "pytest tests/test_adapter.py",
    "payload matches contracts/rank-request-v2.md"
  ],
  "local_commands": [
    "pytest tests/test_adapter.py"
  ],
  "status": "todo"
}
```

Status values:
- `todo`
- `in_progress`
- `blocked`
- `done`

### `patches/<task_id>/patch-manifest.json`

Purpose: structured metadata for a patch bundle produced on a worker machine.

Recommended shape:

```json
{
  "task_id": "TASK-001",
  "server_id": "rank-api-01",
  "target_repo": "rank-api",
  "base_commit": "abc1234",
  "changed_paths": [
    "src/adapter.py",
    "tests/test_adapter.py"
  ],
  "patch_files": [
    "rank-api.patch"
  ],
  "local_checks": [
    {
      "command": "pytest tests/test_adapter.py",
      "status": "passed",
      "report": "../../reports/TASK-001/pytest.txt"
    }
  ],
  "manual_steps": []
}
```

### `handoffs/<task_id>/<timestamp>.md`

Purpose: bounded narrative handoff between assistant sessions or humans.

It should answer:
- what changed
- why it changed
- what did not change
- what remains blocked
- what the next session should read first

## Key Interface Mirrors

Use key interface mirrors to reduce coordination drift without mirroring entire repos.

Recommended contents:
- short repo tree excerpt for the affected module
- public signatures
- config keys
- request/response examples
- adapter boundaries
- entrypoint command
- artifact manifest entries

Avoid storing:
- large binary artifacts
- full model weights
- full datasets
- generated caches
- broad snapshots of unrelated code

## Contract Guidance

Keep contracts narrow and concrete. A good contract file usually contains:
- problem statement
- ownership
- version or date
- schema or CLI shape
- sample inputs and outputs
- compatibility constraints
- open questions

If two servers disagree, update the contract first. Do not keep coding against conflicting memories.

## Relay Rules

When direct Git is unavailable:
- record that `relay_required` is true in `env-manifest.json`
- treat the laptop or jump host as a transport boundary, not as the source of truth
- preserve timestamps and source server IDs on transferred artifacts
- never let relay steps silently mutate task status
