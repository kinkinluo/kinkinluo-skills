# kinkinluo-skills

Reusable Codex skills for engineering and ML projects.

## Skills

- **engineering-code** — Modular, professional code structure for long-running projects.
- **ml-infra** — Config-driven ML/LLM training infrastructure with clear stable/experimental layers.
- **model-reproduction** — Strict model reproduction workflow: engineering adapters allowed, algorithms unchanged.

## Install (using Codex skill-installer)

From any machine with Codex:

```bash
cd ~/.codex/skills/.system/skill-installer
scripts/install-skill-from-github.py --repo kinkinluo/kinkinluo-skills --path skills/engineering-code
scripts/install-skill-from-github.py --repo kinkinluo/kinkinluo-skills --path skills/ml-infra
scripts/install-skill-from-github.py --repo kinkinluo/kinkinluo-skills --path skills/model-reproduction
```

Options:
- Pin a version: add `--ref <tag>` after the repo.
- Install multiple skills at once: pass multiple `--path` arguments in a single command.

After installation, restart Codex to load the new skills.

## Layout

```
skills/
  engineering-code/
    SKILL.md
    references/
  ml-infra/
    SKILL.md
    references/
  model-reproduction/
    SKILL.md
```

Each `SKILL.md` explains when to trigger the skill and the workflow to follow. The `references/` folders contain naming conventions, templates, and patterns reused across projects.

## Contributing / Updates

- Open issues or PRs on GitHub.
- Keep skills in English for compatibility with Codex triggers.
- For breaking changes, bump a tag and note it in the commit/PR title.

## License

Add your preferred license file (e.g., MIT) at the repo root so others can legally reuse these skills.
