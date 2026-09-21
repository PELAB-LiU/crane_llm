# Releasing CRANE-LLM

How to publish a version so that `pip install crane-llm` gives people your
latest code. This document is for maintainers of this repository; users do not
need it.

Releases are built by GitHub Actions, not on your machine. **Pushing a tag is
the entire release action.** Everything else here is either one-time setup or
a check.

## What triggers what

| You do | `build.yml` | `release.yml` | Reaches PyPI |
|---|---|---|---|
| push to `dev` or `master` | runs | — | no |
| open a pull request | runs | — | no |
| push a tag `v*` | — | runs | **yes** |

Ordinary pushes can never publish: `build.yml` builds the wheel, verifies it
and discards it, and has no publish step. Only a tag reaches PyPI.

A tag push does not run `build.yml`, because a tag is not a branch.
`release.yml` repeats the same verification itself before publishing. This also
means the tag chooses what is released: the workflow checks out the tagged
commit, so the code and the workflow files both come from there rather than
from the tip of the branch.

---

## The short version

Once the one-time setup below is done, every release is four commands:

```bash
# 1. bump both version numbers to the new version, e.g. 0.2.0 (see below)
# 2. commit the bump
git commit -am "release 0.2.0"
git push

# 3. tag it and push the tag -- this is what triggers publishing
git tag v0.2.0
git push origin v0.2.0
```

Then watch the run under the repository's **Actions** tab. When it finishes,
the new version is on PyPI and attached to a GitHub Release.

---

## One-time setup

None of this is done yet. A release cannot succeed until all three are.

### 1. Claim the name on PyPI

The distribution is named `crane-llm` in `pyproject.toml`. Check whether that
name is free at <https://pypi.org/project/crane-llm/>. If it is taken, pick
another name and change `name` in `pyproject.toml`; this does not affect the
`import crane_llm` name, only what people type after `pip install`.

### 2. Authorise this repository to publish

The release workflow uses PyPI **trusted publishing**, so no API token is
stored anywhere. Configure it once at
<https://pypi.org/manage/account/publishing/> with:

| Field | Value |
|---|---|
| PyPI project name | `crane-llm` |
| Owner | `yarinamomo` |
| Repository name | `crane_llm` |
| Workflow name | `release.yml` |
| Environment | *(leave empty)* |

If the project does not exist on PyPI yet, use the *pending publisher* form on
the same page, which creates the project on the first successful upload.

### 3. Push the workflows

The workflows have to exist on GitHub before a tag can trigger them. They are
in `.github/workflows/`, so pushing the branch that contains them is enough.

---

## Bumping the version

The version lives in **two** files and they must agree:

| File | What it sets |
|---|---|
| `crane_llm/__init__.py` | `__version__`, which `pyproject.toml` reads for the wheel |
| `crane_llm/nb_extension/package.json` | the frontend version that `jupyter labextension list` prints |

Both workflows fail if they differ, so a mismatch cannot reach PyPI. The build
workflow checks it on every push, which is the cheap place to find out; the
release workflow also checks both against the tag.

Use [semantic versioning](https://semver.org): `0.1.0` → `0.1.1` for a fix,
`0.2.0` for new behaviour, `1.0.0` when the interface is stable.

**A version number can never be reused on PyPI**, even after deleting a
release. If you publish something broken, fix it and publish the next patch
version; you cannot re-upload the same one.

---

## What the release workflow does

From `.github/workflows/release.yml`, on any tag starting with `v`:

1. installs Node and Python
2. builds the frontend with `jlpm build:prod` (minified)
3. builds the wheel and sdist
4. checks the tag matches both version numbers
5. installs the wheel into a clean virtual environment and confirms JupyterLab
   reports `crane-llm-jlab ... enabled ok`, then runs the backend checks
6. creates a GitHub Release with generated notes and attaches the wheel
7. uploads to PyPI

If any step fails, nothing is published. Step 5 is the one that matters most:
it is what stops a wheel shipping without its frontend bundle, which would
install cleanly and then show no button.

---

## Before you tag

Worth doing, since a bad release cannot be taken back cleanly:

```bash
python -m crane_llm.nb_extension.smoke_test    # backend checks, no LLM call
```

Push to a branch first and let the build workflow run. It performs everything
the release does except publishing, so a green build is a strong signal the
release will work.

---

## If a release goes wrong

- **The workflow failed before publishing.** Nothing was released. Fix the
  problem, delete the tag (`git tag -d v0.2.0 && git push --delete origin
  v0.2.0`) and tag again.
- **It published something broken.** Do not try to reuse the version. Fix,
  bump to the next patch version, and release again. You can additionally
  *yank* the bad version on PyPI, which hides it from new installs without
  breaking anyone who already has it pinned.
- **The tag and versions disagree.** The workflow tells you which file to
  edit. Delete the tag, fix, commit, and tag again.

---

## Installing without PyPI

Each release also attaches the wheel to a GitHub Release, so people can install
an exact version without PyPI at all:

```bash
pip install https://github.com/yarinamomo/crane_llm/releases/download/v0.2.0/crane_llm-0.2.0-py3-none-any.whl
```

This is useful before the PyPI name is sorted out, and for pinning a specific
build.

---

## Building a wheel by hand

You should not need this, but for local inspection:

```bash
cd crane_llm/nb_extension && jlpm install && jlpm build:prod && cd ../..
rm -rf build                      # see below
python -m build
```

Delete `build/` first. `bdist_wheel` zips whatever is in `build/lib` and never
removes files that are no longer part of the package, so a stale directory can
put files into your wheel that the current configuration excludes. The CI
build never hits this because it starts from a fresh checkout.
