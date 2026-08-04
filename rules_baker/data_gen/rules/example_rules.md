# House Style & Architecture Rules

> This is the file you REPLACE with your own. Each `## RULE` block is parsed into
> one atomic, id'd rule. Keep each rule single-purpose and checkable — the eval
> harness later verifies the baked model actually obeys these.

## RULE naming-private-underscore
Private/internal helper functions and methods MUST be prefixed with a single
underscore (`_fetch_user`, not `fetchUser`). Public API functions use no prefix.

## RULE errors-no-bare-except
Never use a bare `except:` or `except Exception:` that swallows errors silently.
Catch specific exception types and either re-raise with context or log via the
module logger. No `print()` for error reporting.

## RULE typing-required
All function signatures MUST have complete type hints on parameters and return
values. Use `from __future__ import annotations` at the top of every module.

## RULE logging-module-logger
Modules MUST create a logger with `logger = logging.getLogger(__name__)` at
module scope and use it. Never call `print()` for diagnostics.

## RULE layering-no-db-in-handlers
HTTP route handlers / controllers MUST NOT contain database queries directly.
They call a service-layer function, which calls a repository. Handlers stay thin.

## RULE docstrings-google-style
Every public function and class MUST have a Google-style docstring with Args,
Returns, and Raises sections where applicable.

## RULE constants-uppercase-module
Magic numbers and strings MUST be lifted to module-level UPPER_SNAKE_CASE
constants. No inline literals for timeouts, limits, or config keys.
