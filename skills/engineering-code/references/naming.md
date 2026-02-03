# Naming Conventions

## Table of Contents
1. [File Naming](#file-naming)
2. [Code Naming](#code-naming)
3. [Directory Naming](#directory-naming)
4. [Language-Specific](#language-specific)

---

## File Naming {#file-naming}

### TypeScript/JavaScript

| Type | Convention | Example |
|------|------------|---------|
| Module file | kebab-case | `user-service.ts` |
| Component file | PascalCase | `UserProfile.tsx` |
| Test file | [name].test.ts | `user-service.test.ts` |
| Type definition | kebab-case | `user-types.ts` or `types.ts` |
| Index/barrel | index.ts | `index.ts` |
| Config | kebab-case | `database-config.ts` |
| Constant | kebab-case | `error-codes.ts` |
| Hook | camelCase with use prefix | `useAuth.ts` |
| Utility | kebab-case | `string-utils.ts` |

### Python

| Type | Convention | Example |
|------|------------|---------|
| Module file | snake_case | `user_service.py` |
| Test file | test_[name].py | `test_user_service.py` |
| Package | snake_case | `my_package/` |
| Config | snake_case | `database_config.py` |

---

## Code Naming {#code-naming}

### Variables and Functions

```typescript
// Variables - camelCase, descriptive
const userId = '123';
const isAuthenticated = true;
const maxRetryCount = 3;

// Functions - camelCase, verb prefix
function getUserById(id: string) {}
function validateEmail(email: string) {}
function calculateTotalPrice(items: Item[]) {}

// Boolean variables - is/has/should prefix
const isLoading = true;
const hasPermission = false;
const shouldRefetch = true;

// Arrays - plural nouns
const users: User[] = [];
const selectedItems: Item[] = [];

// Maps/Records - descriptive
const userById: Record<string, User> = {};
const priceByProductId: Map<string, number> = new Map();
```

### Constants

```typescript
// Module-level constants - UPPER_SNAKE_CASE
const MAX_RETRY_COUNT = 3;
const DEFAULT_TIMEOUT_MS = 5000;
const API_VERSION = 'v1';

// Enum values - UPPER_SNAKE_CASE
enum Status {
  PENDING = 'pending',
  ACTIVE = 'active',
  COMPLETED = 'completed',
}

// Object constants - PascalCase for object, UPPER_SNAKE_CASE for keys
const ErrorCodes = {
  NOT_FOUND: 'E001',
  UNAUTHORIZED: 'E002',
  VALIDATION_FAILED: 'E003',
} as const;
```

### Classes and Types

```typescript
// Classes - PascalCase, noun
class UserService {}
class AuthenticationManager {}
class DatabaseConnection {}

// Interfaces - PascalCase, noun (no I prefix)
interface User {}
interface CreateUserInput {}
interface UserRepository {}

// Type aliases - PascalCase
type UserId = string;
type UserRole = 'admin' | 'user' | 'guest';
type ApiResponse<T> = { data: T; status: number };

// Generic types - single uppercase letter or descriptive
function identity<T>(value: T): T {}
function map<TInput, TOutput>(arr: TInput[], fn: (item: TInput) => TOutput): TOutput[] {}
```

### Event Handlers and Callbacks

```typescript
// Event handlers - handle[Event] or on[Event]
function handleClick() {}
function handleSubmit(event: FormEvent) {}
function onUserCreated(user: User) {}

// Callbacks - descriptive action
const onSuccess = (data: Data) => {};
const onError = (error: Error) => {};
const afterSave = () => {};
```

---

## Directory Naming {#directory-naming}

```
src/
├── components/           # Lowercase, plural
├── hooks/                # Lowercase, plural
├── services/             # Lowercase, plural
├── types/                # Lowercase, plural
├── utils/                # Lowercase, plural
├── config/               # Lowercase, singular (conceptual)
├── database/             # Lowercase, singular
└── modules/              # Lowercase, plural
    ├── user/             # Lowercase, singular (domain name)
    └── auth/             # Lowercase, singular
```

**Rules:**
- Use lowercase always
- Category folders are plural (components, services, utils)
- Domain/feature folders are singular (user, auth, payment)
- No abbreviations unless universally known (api, ui, db)

---

## Language-Specific {#language-specific}

### Python

```python
# Variables and functions - snake_case
user_id = '123'
is_authenticated = True

def get_user_by_id(user_id: str) -> User:
    pass

def validate_email(email: str) -> bool:
    pass

# Constants - UPPER_SNAKE_CASE
MAX_RETRY_COUNT = 3
DEFAULT_TIMEOUT = 5000

# Classes - PascalCase
class UserService:
    pass

class DatabaseConnection:
    pass

# Private members - leading underscore
class User:
    def __init__(self):
        self._internal_state = {}
    
    def _validate(self):
        pass
```

### React Components

```typescript
// Component - PascalCase
function UserProfile({ user }: UserProfileProps) {}

// Component file - PascalCase.tsx
// UserProfile.tsx

// Props interface - [Component]Props
interface UserProfileProps {
  user: User;
  onEdit?: () => void;
}

// Hooks - use[Name]
function useUserProfile(userId: string) {}

// Context - [Name]Context, [Name]Provider
const UserContext = createContext<UserContextValue>(defaultValue);
function UserProvider({ children }: PropsWithChildren) {}
```

### API Routes

```typescript
// RESTful - lowercase, hyphenated, plural resources
// GET    /api/users
// GET    /api/users/:id
// POST   /api/users
// PATCH  /api/users/:id
// DELETE /api/users/:id

// Nested resources
// GET    /api/users/:userId/posts
// GET    /api/users/:userId/posts/:postId

// Actions (when CRUD doesn't fit)
// POST   /api/users/:id/activate
// POST   /api/orders/:id/cancel
```

### Database

```sql
-- Tables - snake_case, plural
users
user_profiles
order_items

-- Columns - snake_case
user_id
created_at
is_active

-- Foreign keys - [table]_id
user_id
order_id

-- Indexes - idx_[table]_[columns]
idx_users_email
idx_orders_user_id_created_at
```

---

## Quick Reference

| Element | TypeScript | Python |
|---------|------------|--------|
| File | `kebab-case.ts` | `snake_case.py` |
| Variable | `camelCase` | `snake_case` |
| Function | `camelCase` | `snake_case` |
| Class | `PascalCase` | `PascalCase` |
| Interface | `PascalCase` | `PascalCase` |
| Constant | `UPPER_SNAKE_CASE` | `UPPER_SNAKE_CASE` |
| Private | `_camelCase` | `_snake_case` |
| Directory | `lowercase` | `lowercase` |
