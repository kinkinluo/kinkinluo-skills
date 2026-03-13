#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_README = """# Coordination Workspace

This repository is the shared coordination layer for federated multi-workspace work.

## Purpose

Use this workspace to coordinate work across multiple servers or repos when the
full project cannot live in one local workspace.

## Workflow

1. Each participating server updates `servers/<server_id>/env-manifest.json`.
2. Each participating server updates `servers/<server_id>/workspace-summary.md`.
3. The coordinator writes or updates contracts in `contracts/`.
4. The coordinator creates work items in `work-items/`.
5. Workers emit patch manifests under `patches/`, handoffs under `handoffs/`,
   and reports under `reports/`.
"""


GITIGNORE = """*.log
*.orig
*.rej
*.tmp
"""


WORKSPACE_SUMMARY_TEMPLATE = """# Workspace Summary: {server_id}

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
"""


ENV_MANIFEST_TEMPLATE = {
    "server_id": "",
    "repos": [],
    "entrypoints": [],
    "runtimes": [],
    "artifacts": [],
    "restricted_network": False,
    "relay_required": False,
    "notes": "",
}


DIRECTORY_READMES = {
    "contracts": "# Contracts\n\nStore shared interface and environment contracts here.\n",
    "handoffs": "# Handoffs\n\nStore per-task session handoffs here.\n",
    "patches": "# Patches\n\nStore patch bundles and patch manifests here.\n",
    "reports": "# Reports\n\nStore build, test, and execution reports here.\n",
    "servers": "# Servers\n\nStore per-server manifests and workspace summaries here.\n",
    "work-items": "# Work Items\n\nStore assignable task packets here.\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize a coordination workspace for federated orchestration."
    )
    parser.add_argument("root", help="Path to the coordination workspace to create.")
    parser.add_argument(
        "--server-id",
        action="append",
        default=[],
        help="Create starter manifest and summary files for this server ID. Repeat as needed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite starter files if they already exist.",
    )
    return parser.parse_args()


def write_text(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ensure_structure(root: Path, force: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_text(root / "README.md", ROOT_README, force)
    write_text(root / ".gitignore", GITIGNORE, force)

    for dirname, readme in DIRECTORY_READMES.items():
        directory = root / dirname
        directory.mkdir(parents=True, exist_ok=True)
        write_text(directory / "README.md", readme, force)


def seed_server(root: Path, server_id: str, force: bool) -> None:
    server_dir = root / "servers" / server_id
    server_dir.mkdir(parents=True, exist_ok=True)

    manifest = dict(ENV_MANIFEST_TEMPLATE)
    manifest["server_id"] = server_id

    write_json(server_dir / "env-manifest.json", manifest, force)
    write_text(
        server_dir / "workspace-summary.md",
        WORKSPACE_SUMMARY_TEMPLATE.format(server_id=server_id),
        force,
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    ensure_structure(root, args.force)
    for server_id in args.server_id:
        seed_server(root, server_id, args.force)

    print(f"Initialized coordination workspace at {root}")


if __name__ == "__main__":
    main()
