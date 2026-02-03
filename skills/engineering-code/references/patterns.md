# Code Patterns

## Table of Contents
1. [Error Handling](#error-handling)
2. [Dependency Injection](#dependency-injection)
3. [Repository Pattern](#repository-pattern)
4. [Service Layer](#service-layer)
5. [Configuration](#configuration)
6. [API Response](#api-response)
7. [Validation](#validation)
8. [Logging](#logging)

---

## Error Handling {#error-handling}

### Domain Errors (TypeScript)

```typescript
// shared/errors/domain-error.ts
export class DomainError extends Error {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number = 400,
    public readonly context?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'DomainError';
    Error.captureStackTrace(this, this.constructor);
  }
}

// Usage in modules
export class UserNotFoundError extends DomainError {
  constructor(userId: string) {
    super(
      `User not found: ${userId}`,
      'USER_NOT_FOUND',
      404,
      { userId }
    );
  }
}

export class ValidationError extends DomainError {
  constructor(field: string, message: string) {
    super(message, 'VALIDATION_ERROR', 400, { field });
  }
}
```

### Result Type Pattern

```typescript
// shared/types/result.ts
export type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

export function ok<T>(data: T): Result<T, never> {
  return { success: true, data };
}

export function err<E>(error: E): Result<never, E> {
  return { success: false, error };
}

// Usage
async function createUser(input: CreateUserInput): Promise<Result<User, UserError>> {
  const existingUser = await userRepo.findByEmail(input.email);
  if (existingUser) {
    return err(new EmailAlreadyExistsError(input.email));
  }
  
  const user = await userRepo.create(input);
  return ok(user);
}

// Handling
const result = await createUser(input);
if (result.success) {
  console.log('Created:', result.data);
} else {
  console.error('Failed:', result.error.message);
}
```

### Python Error Handling

```python
# shared/errors.py
from dataclasses import dataclass
from typing import Any

@dataclass
class DomainError(Exception):
    message: str
    code: str
    status_code: int = 400
    context: dict[str, Any] | None = None

class UserNotFoundError(DomainError):
    def __init__(self, user_id: str):
        super().__init__(
            message=f"User not found: {user_id}",
            code="USER_NOT_FOUND",
            status_code=404,
            context={"user_id": user_id}
        )
```

---

## Dependency Injection {#dependency-injection}

### Constructor Injection (TypeScript)

```typescript
// modules/user/user.service.ts
export class UserService {
  constructor(
    private readonly userRepository: UserRepository,
    private readonly emailService: EmailService,
    private readonly logger: Logger
  ) {}

  async createUser(input: CreateUserInput): Promise<User> {
    this.logger.info('Creating user', { email: input.email });
    
    const user = await this.userRepository.create(input);
    await this.emailService.sendWelcome(user.email);
    
    return user;
  }
}

// Factory function for composition root
export function createUserService(deps: {
  userRepository: UserRepository;
  emailService: EmailService;
  logger: Logger;
}): UserService {
  return new UserService(deps.userRepository, deps.emailService, deps.logger);
}
```

### Python Dependency Injection

```python
# modules/user/service.py
from dataclasses import dataclass
from .repository import UserRepository
from shared.services import EmailService, Logger

@dataclass
class UserService:
    user_repository: UserRepository
    email_service: EmailService
    logger: Logger

    async def create_user(self, input: CreateUserInput) -> User:
        self.logger.info("Creating user", email=input.email)
        
        user = await self.user_repository.create(input)
        await self.email_service.send_welcome(user.email)
        
        return user

# Factory
def create_user_service(
    user_repository: UserRepository,
    email_service: EmailService,
    logger: Logger
) -> UserService:
    return UserService(
        user_repository=user_repository,
        email_service=email_service,
        logger=logger
    )
```

---

## Repository Pattern {#repository-pattern}

### TypeScript Repository

```typescript
// modules/user/types.ts
export interface UserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  findMany(filter: UserFilter): Promise<User[]>;
  create(input: CreateUserInput): Promise<User>;
  update(id: string, data: Partial<User>): Promise<User>;
  delete(id: string): Promise<void>;
}

// modules/user/user.repository.ts
export class PostgresUserRepository implements UserRepository {
  constructor(private readonly db: Database) {}

  async findById(id: string): Promise<User | null> {
    const row = await this.db.query(
      'SELECT * FROM users WHERE id = $1',
      [id]
    );
    return row ? this.toUser(row) : null;
  }

  async create(input: CreateUserInput): Promise<User> {
    const row = await this.db.query(
      `INSERT INTO users (email, name, password_hash) 
       VALUES ($1, $2, $3) 
       RETURNING *`,
      [input.email, input.name, await hash(input.password)]
    );
    return this.toUser(row);
  }

  private toUser(row: UserRow): User {
    return {
      id: row.id,
      email: row.email,
      name: row.name,
      createdAt: row.created_at,
    };
  }
}
```

---

## Service Layer {#service-layer}

```typescript
// modules/order/order.service.ts
export class OrderService {
  constructor(
    private readonly orderRepo: OrderRepository,
    private readonly productRepo: ProductRepository,
    private readonly paymentService: PaymentService,
    private readonly notificationService: NotificationService
  ) {}

  async createOrder(input: CreateOrderInput): Promise<Order> {
    // 1. Validate products exist and have stock
    const products = await this.validateProducts(input.items);
    
    // 2. Calculate totals
    const totals = this.calculateTotals(products, input.items);
    
    // 3. Process payment
    const payment = await this.paymentService.charge({
      amount: totals.total,
      customerId: input.customerId,
    });
    
    // 4. Create order
    const order = await this.orderRepo.create({
      customerId: input.customerId,
      items: input.items,
      totals,
      paymentId: payment.id,
    });
    
    // 5. Send notification (async, don't await)
    this.notificationService.sendOrderConfirmation(order).catch(console.error);
    
    return order;
  }

  private async validateProducts(items: OrderItem[]): Promise<Product[]> {
    // Validation logic
  }

  private calculateTotals(products: Product[], items: OrderItem[]): OrderTotals {
    // Calculation logic
  }
}
```

---

## Configuration {#configuration}

### TypeScript Config

```typescript
// config/env.ts
import { z } from 'zod';

const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']),
  PORT: z.coerce.number().default(3000),
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  LOG_LEVEL: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
});

export const env = envSchema.parse(process.env);

// config/index.ts
import { env } from './env';

export const config = {
  server: {
    port: env.PORT,
    env: env.NODE_ENV,
  },
  database: {
    url: env.DATABASE_URL,
    poolSize: env.NODE_ENV === 'production' ? 20 : 5,
  },
  auth: {
    jwtSecret: env.JWT_SECRET,
    tokenExpiry: '7d',
  },
  logging: {
    level: env.LOG_LEVEL,
  },
} as const;

export type Config = typeof config;
```

### Python Config

```python
# config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_env: str = "development"
    port: int = 8000
    database_url: str
    jwt_secret: str
    log_level: str = "info"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

---

## API Response {#api-response}

```typescript
// shared/types/api.ts
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  meta?: {
    page?: number;
    pageSize?: number;
    total?: number;
  };
}

// shared/utils/response.ts
export function successResponse<T>(data: T, meta?: ApiResponse<T>['meta']): ApiResponse<T> {
  return { success: true, data, meta };
}

export function errorResponse(code: string, message: string, details?: Record<string, unknown>): ApiResponse<never> {
  return {
    success: false,
    error: { code, message, details },
  };
}

export function paginatedResponse<T>(
  data: T[],
  page: number,
  pageSize: number,
  total: number
): ApiResponse<T[]> {
  return {
    success: true,
    data,
    meta: { page, pageSize, total },
  };
}
```

---

## Validation {#validation}

### Zod Schemas

```typescript
// modules/user/schemas.ts
import { z } from 'zod';

export const createUserSchema = z.object({
  email: z.string().email('Invalid email format'),
  name: z.string().min(2, 'Name must be at least 2 characters'),
  password: z
    .string()
    .min(8, 'Password must be at least 8 characters')
    .regex(/[A-Z]/, 'Password must contain uppercase letter')
    .regex(/[0-9]/, 'Password must contain number'),
});

export const updateUserSchema = createUserSchema.partial().omit({ password: true });

export type CreateUserInput = z.infer<typeof createUserSchema>;
export type UpdateUserInput = z.infer<typeof updateUserSchema>;
```

### Validation Middleware

```typescript
// shared/middleware/validation.ts
import { Request, Response, NextFunction } from 'express';
import { ZodSchema } from 'zod';

export function validate(schema: ZodSchema) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);
    
    if (!result.success) {
      return res.status(400).json({
        success: false,
        error: {
          code: 'VALIDATION_ERROR',
          message: 'Invalid request data',
          details: result.error.flatten(),
        },
      });
    }
    
    req.body = result.data;
    next();
  };
}
```

---

## Logging {#logging}

```typescript
// shared/utils/logger.ts
export interface Logger {
  debug(message: string, context?: Record<string, unknown>): void;
  info(message: string, context?: Record<string, unknown>): void;
  warn(message: string, context?: Record<string, unknown>): void;
  error(message: string, error?: Error, context?: Record<string, unknown>): void;
}

export function createLogger(module: string): Logger {
  const log = (level: string, message: string, context?: Record<string, unknown>) => {
    console.log(JSON.stringify({
      timestamp: new Date().toISOString(),
      level,
      module,
      message,
      ...context,
    }));
  };

  return {
    debug: (msg, ctx) => log('debug', msg, ctx),
    info: (msg, ctx) => log('info', msg, ctx),
    warn: (msg, ctx) => log('warn', msg, ctx),
    error: (msg, err, ctx) => log('error', msg, { error: err?.message, stack: err?.stack, ...ctx }),
  };
}

// Usage in modules
const logger = createLogger('UserService');
logger.info('User created', { userId: user.id });
```
