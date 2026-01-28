# Documentation Build and Testing Guide

## Building the Documentation

### Prerequisites

Ensure Sphinx and extensions are installed:

```bash
cd cdiutils/
pip install -e ".[docs]"

# Or manually:
pip install sphinx sphinx-rtd-theme nbsphinx pydata-sphinx-theme
```

### Build HTML Documentation

```bash
cd docs/
make clean
make html
```

Output will be in `docs/_build/html/index.html`

### View Locally

```bash
# Option 1: Python HTTP server
cd docs/_build/html
python -m http.server 8000
# Open browser to http://localhost:8000

# Option 2: Direct file
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux
```

## Common Build Issues

### Missing Dependencies

**Error:** `sphinx-build: not found`
```bash
pip install sphinx
```

**Error:** `No module named 'sphinx_rtd_theme'`
```bash
pip install sphinx-rtd-theme pydata-sphinx-theme
```

**Error:** `No module named 'nbsphinx'`
```bash
pip install nbsphinx
```

### Import Errors During Build

Sphinx autodoc tries to import modules. If optional dependencies missing:

**Solution 1:** Mock imports (already in `conf.py`):
```python
autodoc_mock_imports = [
    "vtk",
    "pynx",
    "xrayutilities",
    "silx",
    "PyQt5",
    "fabio",
]
```

**Solution 2:** Install in doc build environment:
```bash
conda install -c conda-forge pynx xrayutilities
```

### Broken Cross-References

**Warning:** `undefined label: 'some_label'`

Check:
1. Target file exists in toctree
2. Label/reference syntax correct
3. No typos in class/function names

### Notebook Errors

**Error:** Jupyter kernel not found

```bash
conda install -c conda-forge ipykernel
python -m ipykernel install --user --name=python3
```

## Testing Documentation

### Test 1: Check for Warnings

Sphinx warnings indicate problems:

```bash
cd docs/
make html 2>&1 | grep -i warning
```

**Zero warnings = good**

### Test 2: Verify All Code Examples

Extract and test code blocks:

```bash
# Manual testing
cd docs/getting_started/
grep -A 20 ".. code-block:: python" quickstart.rst > /tmp/test_quickstart.py
python /tmp/test_quickstart.py
```

Automated (future work):
```python
# tests/test_docs.py
import pytest
import doctest

def test_user_guide_examples():
    doctest.testfile("../docs/user_guide/coordinate_systems.rst")
```

### Test 3: Link Checking

```bash
cd docs/
make linkcheck
```

Reviews all hyperlinks (external and internal)

### Test 4: Fresh Environment

Test in clean conda environment:

```bash
# Create fresh environment
conda create -n test_docs python=3.10
conda activate test_docs

# Install cdiutils
cd cdiutils/
pip install -e .

# Try quickstart example
python <<EOF
from cdiutils.pipeline import BcdiPipeline
print("Import successful!")
EOF
```

## ReadTheDocs Integration

### Configuration

Ensure `.readthedocs.yaml` exists:

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.10"

sphinx:
  configuration: docs/conf.py

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

### Trigger Build

ReadTheDocs builds automatically on:
- Push to `master` or `dev` branch
- Pull request creation

Manual trigger:
1. Visit https://readthedocs.org/projects/cdiutils/
2. Click "Build Version"

### Review Build Logs

Check https://readthedocs.org/projects/cdiutils/builds/ for errors

## Pre-Release Checklist

Before merging documentation changes:

- [ ] `make html` completes without errors
- [ ] No Sphinx warnings
- [ ] All cross-references resolve
- [ ] Code examples tested in fresh environment
- [ ] Broken links checked (`make linkcheck`)
- [ ] ReadTheDocs preview build successful
- [ ] Navigation structure makes sense
- [ ] Mobile view works (check ReadTheDocs preview)

## Continuous Improvement

### Metrics to Track

1. **User engagement:** Google Analytics on ReadTheDocs
2. **GitHub issues:** "Documentation" label count (should decrease)
3. **Support questions:** Repeated questions indicate doc gaps
4. **Time to first analysis:** Survey new users

### Feedback Loop

1. Monitor GitHub issues for doc requests
2. Track Stack Overflow questions about cdiutils
3. Survey workshop attendees
4. Review README.md vs docs consistency

### Maintenance Schedule

- **Weekly:** Review new issues for doc gaps
- **Monthly:** Update changelog.rst
- **Quarterly:** Review most-visited pages (Google Analytics)
- **Per release:** Update version in conf.py, verify all examples

## Known Limitations

### Current State (Post Phase 1 & 2)

✅ Structure in place
✅ Critical examples fixed
✅ BCDI-specific topics covered (coordinates, calibration, wavefront)
⏸️ 9 user guide placeholders remain
⏸️ Source docstrings need Phase 3 overhaul
⏸️ No automated example testing

### Next Steps

See `DOCUMENTATION_OVERHAUL_SUMMARY.md` for detailed roadmap.

Priority order:
1. Phase 3: Rewrite top 5 class docstrings
2. Fill user guide placeholders (phase_retrieval_tuning, pipeline, beamlines)
3. Implement automated doctest
4. Review and integrate existing tutorials/

## Questions?

**Build fails:** Check Sphinx version >= 4.0
**Import errors:** Add to autodoc_mock_imports in conf.py  
**Broken examples:** See DOCUMENTATION_OVERHAUL_SUMMARY.md for verified examples
**Structure unclear:** Review docs/getting_started/index.rst for navigation
