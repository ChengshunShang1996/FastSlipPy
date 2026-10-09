# Releasing FastSlipPy

FastSlipPy releases are created by `.github/workflows/release.yml`. A release
tag must point to `master`, and its version must exactly match `project.version`
in `pyproject.toml`.

The workflow runs the complete CI pipeline, builds the distributions once,
prepares a draft GitHub release, publishes the verified files to PyPI, and only
then makes the GitHub release public.

## One-time setup

### PyPI trusted publisher

In the PyPI settings for the `fastslippy` project, add a GitHub trusted
publisher with these values:

- Owner: `ChengshunShang1996`
- Repository: `FastSlipPy`
- Workflow: `release.yml`
- Environment: `pypi`

No PyPI API token or GitHub repository secret is required.

### GitHub environment

Create an environment named `pypi` in the repository settings. Add a required
reviewer when releases should require explicit approval before PyPI upload.
Restrict deployment tags to `v*`.

Protect `master` and release tags with repository rulesets. Require the CI jobs
before merging to `master`, and limit who can create tags matching `v*`.

## Release procedure

1. Update `project.version` in `pyproject.toml` and update the release notes.
2. Open a pull request to `master` and wait for every CI job to pass.
3. Merge the pull request.
4. In PowerShell, read the release version from `pyproject.toml` and create an
   annotated tag on the resulting `master` commit:

   ```powershell
   git switch master
   git pull --ff-only

   if (git status --porcelain) {
       throw "The working tree must be clean before a release"
   }

   $versionMatch = Select-String -Path pyproject.toml -Pattern '^version\s*=\s*"(\d+\.\d+\.\d+)"$'
   if (-not $versionMatch) {
       throw "Could not read an X.Y.Z project.version from pyproject.toml"
   }
   $releaseVersion = $versionMatch.Matches[0].Groups[1].Value
   Write-Host "Preparing FastSlipPy $releaseVersion"

   git tag -a "v$releaseVersion" -m "FastSlipPy $releaseVersion"
   git show "v$releaseVersion" --no-patch

   $confirmation = Read-Host "Push v$releaseVersion and start the release? Type yes"
   if ($confirmation -ne "yes") {
       throw "Release cancelled; the tag remains local"
   }
   git push origin "v$releaseVersion"
   ```

   Inspect the `git show` output before confirming the final push.

5. Review the workflow run and approve the `pypi` environment deployment.
6. Test the published package in a temporary environment from the same
   PowerShell session:

   ```powershell
   python -m venv .venv-release-check
   .\.venv-release-check\Scripts\python -m pip install --no-cache-dir "fastslippy==$releaseVersion"
   .\.venv-release-check\Scripts\python -c "from fastslippy import FastSlipPy"
   ```

PyPI versions are immutable. Never move or reuse a release tag after its files
have been published. If a PyPI upload succeeds but a later job fails, rerun only
the failed job rather than rerunning the entire workflow.
