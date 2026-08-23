# Releasing

How `cafe-core` is published to PyPI. This is a **maintainer** guide — you don't need
it to *use* CAFE.

## How it works

Publishing is automatic: when you publish a **GitHub Release**, the
[`.github/workflows/publish.yml`](https://github.com/fabian-lu/Cafe/blob/main/.github/workflows/publish.yml)
workflow builds `packages/cafe-core` and uploads it to PyPI via **Trusted Publishing**
(OIDC). No API tokens are stored anywhere — PyPI verifies the workflow's identity
directly.

## Versioning (SemVer)

The version lives in `packages/cafe-core/pyproject.toml` — that file is the **single
source of truth**; the build reads the version from it (the git tag is just a label and
the trigger).

| Bump | When | Example |
|---|---|---|
| **PATCH** (`Z`) | bug fix, no API change | `0.1.0 → 0.1.1` |
| **MINOR** (`Y`) | new feature, backward-compatible | `0.1.1 → 0.2.0` |
| **MAJOR** (`X`) | breaking change (renamed/removed API) | `0.9.0 → 1.0.0` |

!!! note "Pre-1.0 (`0.x.y`)"
    While the version is `0.x`, the API carries no stability promise — breaking changes
    are allowed. Convention: features bump the **minor**, fixes bump the **patch**.
    Reserve **`1.0.0`** for when you commit to a stable API.

## Cutting a release

1. **Bump the version** in `packages/cafe-core/pyproject.toml`.
2. **Commit and push** to `main`.
3. On GitHub: **Releases → Draft a new release → Choose a tag →** type `vX.Y.Z`
   (matching the pyproject version) **→ "Create new tag on publish" → Publish release.**
4. The **Publish to PyPI** workflow runs (~1–2 min) and uploads the build.
5. **Verify** it's live:

   ```bash
   python -m venv /tmp/cafe-check && /tmp/cafe-check/bin/pip install cafe-core
   /tmp/cafe-check/bin/cafe version        # prints the new version
   ```

The same thing from the terminal instead of the browser, if you prefer:

```bash
# after bumping pyproject + pushing
gh release create v0.2.0 --title "v0.2.0" --notes "..."
```

## Rules that bite

- **PyPI is append-only.** A version can never be re-uploaded or reused — not even after
  yanking. **Always bump `pyproject.toml` first**; if you re-release the same number the
  workflow rebuilds it and PyPI rejects it (*"file already exists"*).
- **Keep the tag and the pyproject version equal** (`v0.2.0` ↔ `version = "0.2.0"`).
- **Broke a release?** You **yank** it on PyPI (hides it from new installs; existing pins
  still resolve) and ship a fixed higher version. You don't delete.

## If the publish step fails

- **OIDC / "trusted publisher" auth error** → the PyPI publisher isn't set up or a value
  doesn't match (see below). Fix it, then **re-run the failed job** — no version is
  consumed, so there's nothing to bump.
- **"file already exists"** → the version wasn't bumped. Bump `pyproject.toml`, then cut a
  new tag/release.

## One-time setup (already done — for reference)

Trusted Publishing was configured once on PyPI (Account → Publishing → *pending
publisher*), with values that must match the workflow exactly:

| Field | Value |
|---|---|
| PyPI project name | `cafe-core` |
| Owner | `fabian-lu` |
| Repository | `Cafe` |
| Workflow filename | `publish.yml` |
| Environment | *(blank)* |

PyPI also requires **2FA** on the account to publish. If the trusted publisher is ever
removed or the repo is renamed, re-add it with the matching values.
