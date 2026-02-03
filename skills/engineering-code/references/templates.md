# Project Templates

## Table of Contents
1. [Backend API (Node.js/TypeScript)](#backend-api)
2. [Frontend React App](#frontend-react)
3. [Python Backend](#python-backend)
4. [Library/Package](#library-package)
5. [Full-Stack Monorepo](#full-stack-monorepo)
6. [CLI Tool](#cli-tool)

---

## Backend API (Node.js/TypeScript) {#backend-api}

```
project-name/
├── src/
│   ├── modules/                    # Feature modules
│   │   ├── user/
│   │   │   ├── index.ts            # Public exports
│   │   │   ├── types.ts            # User types/interfaces
│   │   │   ├── user.service.ts     # Business logic
│   │   │   ├── user.controller.ts  # HTTP handlers
│   │   │   ├── user.repository.ts  # Data access
│   │   │   ├── user.routes.ts      # Route definitions
│   │   │   └── user.test.ts        # Tests
│   │   └── auth/
│   │       ├── index.ts
│   │       ├── types.ts
│   │       ├── auth.service.ts
│   │       ├── auth.controller.ts
│   │       └── auth.test.ts
│   │
│   ├── shared/                     # Cross-cutting concerns
│   │   ├── types/
│   │   │   ├── index.ts
│   │   │   ├── common.ts           # Shared types
│   │   │   └── api.ts              # API response types
│   │   ├── utils/
│   │   │   ├── index.ts
│   │   │   ├── validation.ts
│   │   │   └── logger.ts
│   │   ├── middleware/
│   │   │   ├── index.ts
│   │   │   ├── auth.middleware.ts
│   │   │   └── error.middleware.ts
│   │   └── errors/
│   │       ├── index.ts
│   │       └── domain-error.ts
│   │
│   ├── config/
│   │   ├── index.ts                # Config aggregator
│   │   ├── database.ts
│   │   ├── server.ts
│   │   └── env.ts                  # Environment validation
│   │
│   ├── database/
│   │   ├── index.ts
│   │   ├── connection.ts
│   │   └── migrations/
│   │
│   └── index.ts                    # Application entry
│
├── tests/
│   ├── integration/
│   └── e2e/
│
├── package.json
├── tsconfig.json
└── .env.example
```

**Module template:**
```typescript
// modules/user/types.ts
export interface User {
  id: string;
  email: string;
  name: string;
  createdAt: Date;
}

export interface CreateUserInput {
  email: string;
  name: string;
  password: string;
}

export interface UserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  create(input: CreateUserInput): Promise<User>;
  update(id: string, data: Partial<User>): Promise<User>;
  delete(id: string): Promise<void>;
}

export interface UserService {
  getUser(id: string): Promise<User>;
  createUser(input: CreateUserInput): Promise<User>;
}
```

```typescript
// modules/user/index.ts
export { UserService } from './user.service';
export { UserController } from './user.controller';
export { userRoutes } from './user.routes';
export type { User, CreateUserInput } from './types';
```

---

## Frontend React App {#frontend-react}

```
project-name/
├── src/
│   ├── components/                 # Reusable UI components
│   │   ├── Button/
│   │   │   ├── index.ts
│   │   │   ├── Button.tsx
│   │   │   ├── Button.test.tsx
│   │   │   └── Button.module.css
│   │   ├── Input/
│   │   └── Modal/
│   │
│   ├── features/                   # Feature-specific components
│   │   ├── auth/
│   │   │   ├── index.ts
│   │   │   ├── LoginForm.tsx
│   │   │   ├── SignupForm.tsx
│   │   │   └── hooks/
│   │   │       └── useAuth.ts
│   │   └── dashboard/
│   │       ├── index.ts
│   │       ├── Dashboard.tsx
│   │       └── components/
│   │
│   ├── hooks/                      # Shared custom hooks
│   │   ├── index.ts
│   │   ├── useLocalStorage.ts
│   │   └── useDebounce.ts
│   │
│   ├── services/                   # API/external services
│   │   ├── index.ts
│   │   ├── api.ts                  # API client setup
│   │   ├── user.service.ts
│   │   └── auth.service.ts
│   │
│   ├── stores/                     # State management
│   │   ├── index.ts
│   │   ├── user.store.ts
│   │   └── ui.store.ts
│   │
│   ├── types/                      # Shared types
│   │   ├── index.ts
│   │   ├── user.ts
│   │   └── api.ts
│   │
│   ├── utils/                      # Helper functions
│   │   ├── index.ts
│   │   ├── format.ts
│   │   └── validation.ts
│   │
│   ├── styles/                     # Global styles
│   │   ├── globals.css
│   │   └── variables.css
│   │
│   ├── App.tsx
│   └── main.tsx
│
├── public/
├── package.json
└── vite.config.ts
```

**Component template:**
```typescript
// components/Button/types.ts
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}
```

```typescript
// components/Button/Button.tsx
import { type ButtonProps } from './types';
import styles from './Button.module.css';

export function Button({
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  onClick,
  children,
}: ButtonProps) {
  return (
    <button
      className={`${styles.button} ${styles[variant]} ${styles[size]}`}
      disabled={disabled || loading}
      onClick={onClick}
    >
      {loading ? <Spinner /> : children}
    </button>
  );
}
```

---

## Python Backend {#python-backend}

```
project_name/
├── src/
│   └── project_name/
│       ├── __init__.py
│       │
│       ├── modules/                # Feature modules
│       │   ├── __init__.py
│       │   ├── user/
│       │   │   ├── __init__.py     # Public exports
│       │   │   ├── models.py       # Data models
│       │   │   ├── schemas.py      # Pydantic schemas
│       │   │   ├── service.py      # Business logic
│       │   │   ├── repository.py   # Data access
│       │   │   ├── routes.py       # API routes
│       │   │   └── tests/
│       │   │       ├── __init__.py
│       │   │       └── test_service.py
│       │   └── auth/
│       │       └── ...
│       │
│       ├── shared/
│       │   ├── __init__.py
│       │   ├── types.py            # Common types
│       │   ├── errors.py           # Custom exceptions
│       │   ├── utils/
│       │   │   ├── __init__.py
│       │   │   └── validation.py
│       │   └── middleware/
│       │       ├── __init__.py
│       │       └── auth.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py         # Pydantic settings
│       │
│       ├── database/
│       │   ├── __init__.py
│       │   ├── connection.py
│       │   └── migrations/
│       │
│       └── main.py                 # Application entry
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures
│   └── integration/
│
├── pyproject.toml
└── .env.example
```

**Module template:**
```python
# modules/user/schemas.py
from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    name: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True
```

```python
# modules/user/__init__.py
from .service import UserService
from .routes import router as user_router
from .schemas import UserCreate, UserResponse

__all__ = ['UserService', 'user_router', 'UserCreate', 'UserResponse']
```

---

## Library/Package {#library-package}

```
package-name/
├── src/
│   ├── core/                       # Core functionality
│   │   ├── index.ts
│   │   ├── parser.ts
│   │   └── transformer.ts
│   │
│   ├── types/                      # Public types
│   │   ├── index.ts
│   │   └── options.ts
│   │
│   ├── utils/                      # Internal utilities
│   │   ├── index.ts
│   │   └── helpers.ts
│   │
│   └── index.ts                    # Public API (minimal surface)
│
├── tests/
│   ├── core/
│   └── integration/
│
├── examples/
│   └── basic-usage.ts
│
├── package.json
├── tsconfig.json
└── README.md
```

**Public API pattern:**
```typescript
// src/index.ts - Keep minimal
export { parse } from './core/parser';
export { transform } from './core/transformer';
export type { ParseOptions, TransformOptions } from './types';
```

---

## Full-Stack Monorepo {#full-stack-monorepo}

```
project-name/
├── apps/
│   ├── web/                        # Frontend app
│   │   ├── src/
│   │   └── package.json
│   ├── api/                        # Backend API
│   │   ├── src/
│   │   └── package.json
│   └── admin/                      # Admin panel
│       └── ...
│
├── packages/
│   ├── shared-types/               # Shared TypeScript types
│   │   ├── src/
│   │   │   ├── user.ts
│   │   │   ├── api.ts
│   │   │   └── index.ts
│   │   └── package.json
│   ├── shared-utils/               # Shared utilities
│   │   └── ...
│   └── ui/                         # Shared UI components
│       └── ...
│
├── package.json                    # Workspace root
├── turbo.json                      # Turborepo config
└── tsconfig.base.json              # Shared TS config
```

---

## CLI Tool {#cli-tool}

```
cli-name/
├── src/
│   ├── commands/                   # Command implementations
│   │   ├── index.ts
│   │   ├── init.ts
│   │   ├── build.ts
│   │   └── deploy.ts
│   │
│   ├── core/                       # Core logic
│   │   ├── index.ts
│   │   ├── config-loader.ts
│   │   └── executor.ts
│   │
│   ├── utils/
│   │   ├── index.ts
│   │   ├── logger.ts
│   │   └── prompt.ts
│   │
│   ├── types/
│   │   └── index.ts
│   │
│   └── index.ts                    # Entry point
│
├── bin/
│   └── cli.js                      # Executable
│
└── package.json
```
