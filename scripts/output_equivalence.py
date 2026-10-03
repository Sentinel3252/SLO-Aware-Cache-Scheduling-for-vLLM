"""Source-checkout entry point; use the installed CacheFlow command as well."""

from cacheflow.tools.output_equivalence import (
    build_parser,
    capture,
    compare,
    main,
)

__all__ = ["build_parser", "capture", "compare", "main"]

if __name__ == "__main__":
    main()
