# Canonical Templates

Use these templates when creating coordination artifacts from scratch.

## `servers/<server_id>/workspace-summary.md`

~~~md
# Workspace Summary: <server_id>

## Repositories
- `<repo-name>`: <path> - <what this repo owns>

## Entrypoints
- `<name>`: `<command>`

## Key Interfaces Exposed To Other Servers
- `<interface-or-api>`: <short description>

## Large Artifacts Kept Local
- `<artifact>`: <why it stays local>

## Local Commands
- Build: `<command>`
- Test: `<command>`
- Run: `<command>`

## Constraints
- Network: <restricted or unrestricted>
- Relay needed: <yes or no>
- Other limits: <notes>

## Minimal Read Set For The Next Session
- `<path-or-doc>`
- `<path-or-doc>`
~~~

## `contracts/<name>.md`

~~~md
# Contract: <name>

## Purpose
<what coordination problem this contract solves>

## Owners
- Primary: <server or repo owner>
- Related: <other server or repo owner>

## Version
- Date: <YYYY-MM-DD>
- Status: draft | stable

## Interface
<schema, command shape, config keys, or request/response format>

## Sample Input
```json
{}
```

## Sample Output
```json
{}
```

## Compatibility Notes
- <backward compatibility or migration note>

## Open Questions
- <question>
~~~

## `work-items/<task_id>.json`

~~~json
{
  "task_id": "TASK-001",
  "goal": "",
  "owner_server": "",
  "depends_on": [],
  "input_contracts": [],
  "expected_outputs": [],
  "acceptance_checks": [],
  "local_commands": [],
  "status": "todo"
}
~~~

## `handoffs/<task_id>/<timestamp>.md`

~~~md
# Handoff: <task_id>

## What Changed
- <change>

## Why
- <reason>

## What Did Not Change
- <explicit non-change>

## Evidence
- `<report-path>`
- `<patch-path>`

## Blockers
- <blocker or "none">

## Next Recommended Read Order
1. `<path>`
2. `<path>`
3. `<path>`
~~~

## `patches/<task_id>/patch-manifest.json`

~~~json
{
  "task_id": "TASK-001",
  "server_id": "",
  "target_repo": "",
  "base_commit": "",
  "changed_paths": [],
  "patch_files": [],
  "local_checks": [],
  "manual_steps": []
}
~~~
