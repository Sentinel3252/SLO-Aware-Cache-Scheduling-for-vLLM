# Preparing a release

A release is ready when users can install the artifact, follow the documented
workflow, and understand its validation scope. Version `0.1.0` is currently package
metadata; these instructions do not publish anything automatically.

1. Update `cacheflow.__version__`, `CHANGELOG.md`, and the compatibility matrix.
2. Run CPU tests, lint, and formatting checks on the supported CI matrix.
3. Run `python -m build`, `python -m twine check dist/*`, and
   `python scripts/check_distribution.py --dist-dir dist` against fresh artifacts.
4. Review archive contents: exclude local results, credentials, and `interview/`.
5. Verify the pinned vLLM target separately on GPU before claiming integration
   support; attach the environment, output comparison, raw metrics, and limitations.
6. Prepare release notes identifying changes and validation levels. Tag and publish
   only after maintainer review of the exact artifacts.

CI creates validation artifacts only. It does not upload to PyPI, push tags, or
create GitHub releases. Credentials and publishing policy remain maintainer-managed.
