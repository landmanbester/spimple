# Ruff and Pre-commit Setup Guide

This document provides comprehensive instructions for setting up Ruff formatting and linting with pre-commit hooks in the Spimple project.

## Quick Setup for New Team Members

### 1. Install Dependencies
```bash
# Install development dependencies (including ruff, pre-commit, mypy)
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type pre-push
```

### 2. Test the Setup
```bash
# Run pre-commit on all files to see current issues
uv run pre-commit run --all-files

# Format code manually
uv run ruff format .

# Lint code manually
uv run ruff check . --fix

# Type check manually
uv run mypy spimple/
```

## Manual Commands

### Formatting and Linting
```bash
# Format all Python files
uv run ruff format .

# Lint and fix all fixable issues
uv run ruff check . --fix

# Check without fixing
uv run ruff check .

# Show unsafe fixes available
uv run ruff check . --unsafe-fixes

# Format specific file
uv run ruff format path/to/file.py

# Lint specific file
uv run ruff check path/to/file.py --fix

# Check for line length violations specifically
uv run ruff check . --select=E501

# Check line length in specific file
uv run ruff check path/to/file.py --select=E501
```

### Pre-commit Operations
```bash
# Run pre-commit on staged files
uv run pre-commit run

# Run pre-commit on all files
uv run pre-commit run --all-files

# Run specific hook
uv run pre-commit run ruff

# Update pre-commit hooks to latest versions
uv run pre-commit autoupdate

# Temporarily skip hooks for a commit
git commit -m "message" --no-verify
```

### Testing
```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=spimple

# Run specific test file
uv run pytest tests/test_convolve2gaussres.py

# Run specific test function
uv run pytest tests/test_convolve2gaussres.py::test_convolve2gaussres
```

## Configuration Details

### Ruff Configuration (pyproject.toml)

The Ruff configuration has been optimized for scientific computing with:

- **Line length**: 88 characters (Black-compatible)
- **Target Python version**: 3.10+
- **Import sorting**: Automatic with scientific library priorities
- **Comprehensive rule set**: Includes NumPy-specific rules, performance checks, and code quality rules
- **Scientific computing exceptions**: Allows single-letter variables, complex functions, and other patterns common in mathematical code

Key ignored rules for scientific computing:
- `E741`: Ambiguous variable names (allows `i`, `j`, `k`, etc.)
- `PLR2004`: Magic values (allows scientific constants)
- `PLR0913`: Too many arguments (common in scientific APIs)
- `T201`/`T203`: Print statements (useful for debugging)
- `S101`: Assert statements (common in validation)

### Pre-commit Hooks

The pre-commit configuration includes:

1. **Basic file checks**: Trailing whitespace, end-of-file fixes, YAML/TOML validation
2. **Ruff linting and formatting**: Automatic code formatting and issue fixing
3. **Documentation formatting**: Formats code examples in docstrings
4. **Type checking**: MyPy integration with scientific library stubs
5. **Test validation**: Pytest runs on pre-push (not every commit)

### VS Code Integration

The `.vscode/settings.json` configures:

- **Ruff as primary formatter**: Replaces Black, isort, flake8, etc.
- **Format on save**: Automatic formatting when saving files
- **Import organization**: Automatic import sorting on save
- **Jupyter support**: Works with notebook cells
- **Scientific file exclusions**: Ignores .fits, .npy, .npz files from search
- **PyLance type checking disabled**: Removes annoying type checking errors for scientific code
- **Pyright configuration**: Global type checking settings in `pyrightconfig.json`

## Workflow Integration

### Development Workflow
1. **Write code** - VS Code automatically formats on save
2. **Commit changes** - Pre-commit hooks run automatically
3. **Push changes** - Pre-push hooks run tests
4. **CI/CD** - Use same Ruff commands in your CI pipeline

### CI Integration Example
```yaml
# .github/workflows/ci.yml
- name: Lint with Ruff
  run: |
    uv run ruff check .
    uv run ruff format --check .

- name: Type check with MyPy
  run: uv run mypy spimple/
```

### Jupyter Notebooks

Ruff works with Jupyter notebooks:
```bash
# Format notebook
uv run ruff format notebook.ipynb

# Lint notebook
uv run ruff check notebook.ipynb --fix
```

## Line Length Handling

### Why Ruff Doesn't Auto-Fix Long Lines

Ruff's formatter (like Black) **intentionally doesn't automatically wrap long lines** because:
- Context matters for readability
- Automated wrapping can create awkward breaks
- Some long lines are better than poor wrapping

### Handling Line Length Violations

1. **Detection**: E501 errors are enabled and will show line-too-long violations
2. **Manual fixes**: Consider these approaches for long lines:

```python
# Long docstring - break into multiple lines
def function():
    """
    This is a very long docstring that exceeds the line limit and should be
    broken into multiple lines for better readability.
    """


# Long function call - use trailing commas and multiple lines
result = some_function(
    parameter_one,
    parameter_two,
    parameter_three,
)

# Long string - use implicit concatenation or f-strings
message = (
    "This is a very long string that needs to be broken up "
    "across multiple lines for readability"
)

# Long expression - break at logical points
condition = (
    some_long_variable_name > threshold and another_condition_check and final_validation
)
```

3. **VS Code helpers**:
   - Ruler at column 88 shows line limit
   - Ruff will highlight E501 violations
   - Use auto-wrap extensions if needed

## Troubleshooting

### Common Issues

1. **Pre-commit installation fails**:
   ```bash
   # Reinstall pre-commit hooks
   uv run pre-commit uninstall
   uv run pre-commit install
   uv run pre-commit install --hook-type pre-push
   ```

2. **Ruff conflicts with existing tools**:
   - Remove/disable Black, isort, flake8, autopep8 in your editor
   - Update CI/CD to use only Ruff

3. **VS Code not formatting**:
   - Ensure the Ruff extension is installed and enabled
   - Check that other formatters are disabled
   - Restart VS Code

4. **Too many linting errors**:
   ```bash
   # Fix issues gradually by category
   uv run ruff check . --select F  # Only critical errors
   uv run ruff check . --select E,W  # Style issues
   ```

### Performance Tips

- **Large files**: Ruff is extremely fast, but for very large files, consider per-file exclusions
- **Scientific data files**: Already excluded (.fits, .npy, .npz, .dat)
- **Generated code**: Add to `exclude` in pyproject.toml if needed

### Customization

To modify Ruff rules for your specific needs:
1. Edit `[tool.ruff.lint]` section in `pyproject.toml`
2. Add rules to `ignore` list to disable them
3. Add rules to `select` list to enable new ones
4. Use `per-file-ignores` for file-specific exceptions

## Migration from Other Tools

### From Black + isort + flake8
1. Remove old tool configurations
2. Uninstall old tools: `uv remove black isort flake8`
3. Update CI/CD scripts to use Ruff
4. Update editor settings to use Ruff

### Editor-specific migrations
- **PyCharm**: Install Ruff plugin, disable other formatters
- **Vim/Neovim**: Use Ruff server with your LSP client (nvim-lspconfig, etc.)
- **Emacs**: Use eglot or lsp-mode with Ruff server
- **VS Code**: Uses Ruff server natively with the official extension

## Ruff Server Configuration

Ruff server is the modern replacement for ruff-lsp. It provides better performance and integration:

### VS Code
The configuration is already set with `"ruff.nativeServer": true` in `.vscode/settings.json`.

### Neovim (with nvim-lspconfig)
```lua
require('lspconfig').ruff.setup {
  init_options = {
    settings = {
      args = {},
    }
  }
}
```

### Emacs (with eglot)
```elisp
(add-to-list 'eglot-server-programs
             '(python-mode . ("ruff" "server" "--preview")))
```

### Command Line
You can start Ruff server manually for testing:
```bash
uv run ruff server --preview
```

This setup provides a modern, fast, and comprehensive code quality solution optimized for scientific computing workflows.
