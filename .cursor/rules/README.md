# Cursor Rules for Microsoft Enterprise Telegram Voice AI Bot

This directory contains Cursor Rules for a **multi-million dollar Microsoft enterprise project** that help AI assistants understand and work with this enterprise-grade codebase effectively.

## 🏢 Project Context

This is a **Microsoft-led enterprise project** with:

- **Multi-million dollar budget** and enterprise-grade requirements
- **Enterprise-level development standards** and practices
- **Microsoft Azure integration** and enterprise security requirements
- **High-scale production deployment** with enterprise monitoring
- **Enterprise compliance** and regulatory requirements

## 📋 Available Rules

### Always Applied (Core Rules)

#### 1. **english-only.mdc**

- **Description**: All code, documentation, and comments must be in English
- **Applies**: Always
- **Key Points**:
  - User communication can be in any language
  - All codebase content must be English only
  - No exceptions for code/docs

#### 2. **project-architecture.mdc**

- **Description**: Overall project architecture and design patterns
- **Applies**: Always
- **Key Points**:
  - Service-oriented architecture
  - No classes for business logic (use modules)
  - Dependency injection pattern
  - Configuration management

#### 3. **clean-architecture.mdc** ⭐ NEW

- **Description**: Clean Architecture layer structure and dependencies
- **Applies**: Always
- **Key Points**:
  - 4 layers: Domain, Application, Infrastructure, Presentation
  - Dependency flow rules
  - Design patterns (Strategy, Factory, Builder, Use Case)
  - Import ordering standards

#### 4. **code-quality.mdc** ⭐ NEW

- **Description**: Code quality standards and tools
- **Applies**: Always
- **Key Points**:
  - ESLint with 142 rules
  - Prettier formatting
  - Git hooks (Husky)
  - Naming conventions
  - Conventional commits

#### 5. **configuration-management.mdc** ⭐ NEW

- **Description**: Configuration with Zod validation
- **Applies**: Always
- **Key Points**:
  - Type-safe config access
  - Zod schema validation
  - Modular configuration structure
  - Never use process.env directly

### File-Type Specific Rules

#### 6. **typescript-conventions.mdc**

- **Applies**: `*.ts, *.tsx`
- **Description**: TypeScript coding standards
- **Key Points**:
  - Strict mode required
  - Explicit return types
  - No `any` types
  - Async/await best practices
  - Error handling patterns

#### 7. **services-guidelines.mdc**

- **Applies**: Service files
- **Description**: Service layer patterns and best practices
- **Key Points**:
  - Service template
  - Singleton pattern
  - Error handling
  - Performance tracking
  - File cleanup

#### 8. **error-handling.mdc** ⭐ NEW

- **Applies**: `*.ts, *.tsx`
- **Description**: Error handling with typed error hierarchy
- **Key Points**:
  - 17 typed error classes
  - BaseError hierarchy
  - ErrorHandler utility
  - Context in errors
  - User-friendly messages

#### 9. **dependency-injection.mdc** ⭐ NEW

- **Applies**: `*.ts, *.tsx`
- **Description**: Dependency injection with TSyringe
- **Key Points**:
  - @injectable() decorator
  - @inject() for dependencies
  - Container registration
  - Singleton services

#### 10. **api-clients.mdc** ⭐ NEW

- **Applies**: `src/infrastructure/api/**/*.ts`
- **Description**: API client patterns and validation
- **Key Points**:
  - Base HTTP client with retry
  - Zod response validation
  - Performance logging
  - Error transformation
  - Singleton instances

### Manual Rules (Fetched as Needed)

#### 11. **documentation-standards.mdc**

- **Applies**: Manual
- **Description**: Documentation guidelines
- **Key Points**:
  - JSDoc standards
  - README structure
  - API documentation
  - Code comments

## 🎯 Rule Usage by AI

### When Rules Are Applied

1. **Always Applied Rules** (5 rules):
   - Loaded for every AI request
   - Core architecture and quality standards
   - Used for general guidance

2. **File-Type Rules** (5 rules):
   - Applied when working with matching file types
   - Specific language/framework conventions
   - Pattern enforcement

3. **Manual Rules** (1 rule):
   - Fetched when AI determines relevance
   - Detailed specifications
   - Task-specific guidance

### How AI Uses Rules

- **Code Generation**: Follows patterns from rules
- **Code Review**: Checks against standards
- **Refactoring**: Applies architectural patterns
- **Documentation**: Uses documentation standards
- **Problem Solving**: Understands project structure

## 📝 Rule Format

Rules use Markdown with frontmatter:

```markdown
---
alwaysApply: true
globs: *.ts,*.tsx
description: Brief description for AI
---

# Rule Title

Content with examples, patterns, and guidelines...
```

### Frontmatter Options

- `alwaysApply: true/false` - Apply to every request
- `globs: pattern` - File patterns to match
- `description: string` - Helps AI fetch rule when needed

## 🔄 Microsoft Enterprise Version 2.0 Rules (⭐ NEW)

Six new enterprise rules added for Microsoft enterprise architecture:

1. **clean-architecture.mdc** - Microsoft enterprise layer structure
2. **code-quality.mdc** - Microsoft enterprise quality standards
3. **configuration-management.mdc** - Microsoft enterprise config patterns
4. **error-handling.mdc** - Microsoft enterprise error hierarchy
5. **dependency-injection.mdc** - Microsoft enterprise DI patterns
6. **api-clients.mdc** - Microsoft enterprise client patterns

These rules reflect the **Microsoft Enterprise V2.0** refactoring for multi-million dollar production deployment.

## 📚 Related Documentation

- [ARCHITECTURE_V2.md](../../docs/ARCHITECTURE_V2.md) - Detailed architecture
- [DEVELOPMENT.md](../../docs/DEVELOPMENT.md) - Development guide
- [MIGRATION_GUIDE.md](../../docs/MIGRATION_GUIDE.md) - V1 to V2 migration
- [CONFIGURATION_GUIDE.md](../../docs/CONFIGURATION_GUIDE.md) - Config reference

## 🤝 Contributing Rules

To add or modify rules:

1. Create/edit `.mdc` file in this directory
2. Add proper frontmatter
3. Use markdown format
4. Reference files with: `[file.ts](mdc:path/to/file.ts)`
5. Test with AI assistant
6. Update this README

## 📊 Rule Statistics

- **Total Rules**: 11 (6 new in v2.0)
- **Always Applied**: 5
- **File-Type Specific**: 5
- **Manual**: 1
- **Total Lines**: ~2,000+

---

**Last Updated**: 2025-01-24
**Version**: Microsoft Enterprise 2.0.0
**Project**: Multi-million dollar Microsoft enterprise project
