"""Source-checkout entry point; use the installed CacheFlow command as well."""

from cacheflow.tools.run_simulation_study import (
    METRICS,
    _mean_ci95,
    main,
    render_report,
    run_study,
)

__all__ = ["METRICS", "_mean_ci95", "render_report", "run_study", "main"]

if __name__ == "__main__":
    main()
