# Container Usage Guide

This document explains how to use and publish spimple containers.

## GitHub Container Registry (GHCR) Setup

### For Repository Maintainers

#### 1. Enable GitHub Container Registry (first time only)

The workflow is already configured with the correct permissions. No additional setup needed!

#### 2. Create a New Release

To trigger a container build and push:

```bash
# Option A: Create release via GitHub UI
# 1. Go to https://github.com/landmanbester/spimple/releases/new
# 2. Create a new tag (e.g., v0.0.6)
# 3. Fill in release notes
# 4. Click "Publish release"

# Option B: Create release via command line
git tag v0.0.6
git push origin v0.0.6
gh release create v0.0.6 --generate-notes
```

The workflow will automatically:
1. Build the container
2. Test all four commands (imconv, spifit, binterp, mosaic)
3. Tag with version numbers (e.g., `0.0.6`, `0.0`, `0`, `latest`)
4. Push to `ghcr.io/landmanbester/spimple`

#### 3. Verify the Container

After the workflow completes, check:
- GitHub Actions: https://github.com/landmanbester/spimple/actions
- Published packages: https://github.com/landmanbester/spimple/pkgs/container/spimple

#### 4. Make the Container Public (optional)

By default, GHCR containers are private. To make public:

1. Go to https://github.com/users/landmanbester/packages/container/spimple/settings
2. Scroll to "Danger Zone"
3. Click "Change visibility" → "Public"

## Using the Published Container

### Pull the Container

```bash
# Using Docker
docker pull ghcr.io/landmanbester/spimple:latest
docker pull ghcr.io/landmanbester/spimple:0.0.6

# Using Podman
podman pull ghcr.io/landmanbester/spimple:latest
podman pull ghcr.io/landmanbester/spimple:0.0.6
```

### Run Commands

```bash
# Show help
docker run --rm ghcr.io/landmanbester/spimple:latest

# Run specific command
docker run --rm ghcr.io/landmanbester/spimple:latest imconv --help

# Process files (mount current directory)
docker run --rm -v "$PWD:/data:z" -w /data \
  ghcr.io/landmanbester/spimple:latest imconv \
  --image input.fits \
  --output-filename output
```

### Available Tags

- `latest` - Latest stable release
- `0.0.6` - Specific version (full semver)
- `0.0` - Minor version (receives patches)
- `0` - Major version (receives minor updates)

## Local Development

### Build Locally

```bash
# Using Docker
docker build -t spimple:dev .

# Using Podman
podman build -t spimple:dev .
```

### Test Locally

```bash
# Run tests
docker run --rm spimple:dev
docker run --rm spimple:dev imconv --help
docker run --rm spimple:dev spifit --help
docker run --rm spimple:dev binterp --help
docker run --rm spimple:dev mosaic --help
```

## CI/CD Workflow Details

### Trigger Events

- **Automatic**: When a new release is published on GitHub
- **Manual**: Via "Run workflow" button in Actions tab (builds `dev` tag)

### Build Process

1. Checkout code
2. Set up Docker Buildx (for efficient caching)
3. Login to GHCR
4. Extract version from release tag
5. Build container
6. Run test suite (all 4 commands)
7. Push to GHCR with multiple tags
8. Generate build attestation (provenance)

### Tags Generated

For release `v0.0.6`:
- `ghcr.io/landmanbester/spimple:0.0.6`
- `ghcr.io/landmanbester/spimple:0.0`
- `ghcr.io/landmanbester/spimple:0`
- `ghcr.io/landmanbester/spimple:latest`

## Troubleshooting

### Container Build Fails

Check the Actions tab for logs: https://github.com/landmanbester/spimple/actions

### Cannot Pull Container (403 Forbidden)

If the container is private, authenticate first:

```bash
# Create a GitHub Personal Access Token (PAT) with 'read:packages' scope
# Then login:
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

### Need to Rebuild for Dependency Update

If `hip-cargo` or another dependency updates:

1. Create a new patch release (e.g., v0.0.7)
2. Or manually trigger workflow via Actions tab

## HPC/Singularity Usage

Convert to Singularity for HPC environments:

```bash
# Pull and convert
singularity pull spimple_0.0.6.sif docker://ghcr.io/landmanbester/spimple:0.0.6

# Run
singularity run spimple_0.0.6.sif imconv --help
```
