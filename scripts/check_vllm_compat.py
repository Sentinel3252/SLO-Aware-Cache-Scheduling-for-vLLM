"""Source-checkout entry point; use the installed CacheFlow command as well."""

from cacheflow.tools.check_vllm_compat import (
    main,
)

__all__ = ["main"]

if __name__ == "__main__":
    main()
