---
name: engineering-code
description: |
  Engineering-grade code development with modular architecture, clear interfaces, and professional project structure. Use this skill when: (1) Starting any new coding project, (2) Building features that may grow in complexity, (3) User requests maintainable/scalable/professional code, (4) Projects expected to span multiple conversation turns, (5) Code that needs to be readable, testable, or reusable. This skill ensures code is modular (small focused files), consistent (predictable patterns), and navigable (clear project structure) - enabling efficient long-context development.
---

# Engineering Code

This skill produces modular, well-structured code that remains maintainable across long conversations and complex projects.

## Core Principle: Modular by Default

**Every file should have ONE clear responsibility.** This enables:
- Understanding code without loading entire project
- Modifying one feature without touching others
- Reusing components across projects
- Continuing development across conversation turns

## Project Initialization

When starting a new project:

1. **Determine project type** → Select appropriate template from `references/templates.md`
2. **Create directory structure first** → Establish skeleton before writing code
3. **Define interfaces early** → Specify how modules communicate

## File Size Rules

| File Type | Max Lines | Action if Exceeded |
|-----------|-----------|-------------------|
| Module/Component | 150 | Split by responsibility |
| Utility file | 100 | Group related utils separately |
| Type definitions | 200 | Split by domain |
| Test file | 200 | One test file per module |
| Config file | 50 | Use config composition |

## Module Design Pattern

Every module follows this structure:

```
module/
├── index.ts          # Public exports only
├── types.ts          # Interfaces and types
├── [name].ts         # Core implementation
├── [name].test.ts    # Tests
└── utils.ts          # Internal helpers (optional)
```

**index.ts pattern** - Only re-exports, no logic:
```typescript
export { UserService } from './user-service';
export type { User, UserConfig } from './types';
```

## Interface-First Development

Before implementing any feature:

1. **Define the interface** in `types.ts`
2. **Write usage example** in comments
3. **Then implement**

This ensures clear contracts between modules.

## Naming Conventions

See `references/naming.md` for comprehensive conventions. Key rules:

- Files: `kebab-case.ts` (e.g., `user-service.ts`)
- Classes/Types: `PascalCase`
- Functions/Variables: `camelCase`
- Constants: `UPPER_SNAKE_CASE`
- Test files: `[name].test.ts` or `[name].spec.ts`

## Project Structure by Type

See `references/templates.md` for complete templates. Quick reference:

**Backend API:**
```
src/
├── modules/           # Feature modules
│   └── [feature]/
├── shared/            # Cross-cutting concerns
│   ├── types/
│   ├── utils/
│   └── middleware/
├── config/            # Configuration
└── index.ts           # Entry point
```

**Frontend App:**
```
src/
├── components/        # UI components
│   └── [Component]/
├── hooks/             # Custom hooks
├── services/          # API/business logic
├── types/             # Shared types
└── utils/             # Helpers
```

**Library/Package:**
```
src/
├── core/              # Core functionality
├── types/             # Public types
├── utils/             # Internal utilities
└── index.ts           # Public API
```

## Code Organization Rules

### Imports
Order imports consistently:
1. External packages
2. Internal modules (absolute paths)
3. Relative imports
4. Types (with `type` keyword)

### Exports
- Use named exports (not default)
- Export types separately with `export type`
- Keep public API minimal

### Dependencies
- Each module declares its dependencies explicitly
- No circular dependencies
- Shared code goes in `shared/` or `common/`

## Error Handling Pattern

```typescript
// Define domain errors in types.ts
class DomainError extends Error {
  constructor(
    message: string,
    public code: string,
    public context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'DomainError';
  }
}

// Use Result type for recoverable errors
type Result<T, E = Error> = 
  | { success: true; data: T }
  | { success: false; error: E };
```

## Configuration Pattern

```typescript
// config/index.ts
export const config = {
  database: loadDatabaseConfig(),
  api: loadApiConfig(),
  // ...
} as const;

// Type-safe config access
export type Config = typeof config;
```

## Testing Strategy

- One test file per module
- Test public interface, not internals
- Use descriptive test names: `should [action] when [condition]`
- Mock external dependencies only

## Long-Context Development

This structure enables efficient development across long conversations:

1. **Adding features**: Identify target module → Read only that module → Implement
2. **Fixing bugs**: Locate affected module → Read module + its tests → Fix
3. **Refactoring**: Understand interface → Modify internals freely

The modular structure means Claude only needs to load relevant files, not the entire project.

## Workflow

1. **New project**: Read `references/templates.md` → Initialize structure
2. **New feature**: Create module skeleton → Define types → Implement → Test
3. **Modification**: Identify module → Load minimal context → Change
4. **Review**: Check file sizes → Verify interface clarity → Ensure test coverage

## References

- `references/templates.md` - Complete project templates by type
- `references/naming.md` - Comprehensive naming conventions
- `references/patterns.md` - Common code patterns and examples
