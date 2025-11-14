# Tutorial: Splitting and Recombining ROCm Artifacts

> **⚠️ DISCLAIMER**: This tutorial represents the QA plan for the artifact split/recombine workflow. Not all functionality is working properly yet. Future revisions will address known issues and improve the process.

## Known Issues to Address

1. **Generic vs Arch-Specific Artifact Separation**: Currently, the recombined artifacts place host code (e.g., `librocblas.so`) in arch-specific artifacts like `blas_lib_gfx110X`. The correct behavior should be:
   - **Generic artifact** (`blas_lib_generic`): Contains host code (shared libraries with `.rocm_kpack_ref` markers)
   - **Arch-specific artifacts** (`blas_lib_gfx110X`, etc.): Contains only `.kpack` files and architecture-specific database files

   This needs to be fixed in the recombination tooling to properly separate generic from arch-specific content.

2. **Generic-Only Artifacts**: Components that have no architecture-specific device code (e.g., `support_dev`, `support_doc`) currently produce empty arch-specific artifacts. The tooling needs to handle these cases by:
   - Detecting when a component has no arch-specific content
   - Either skipping arch-specific artifact creation or clearly marking them as generic-only
   - Ensuring downstream packaging understands which artifacts are truly generic

## Overview

The ROCm build system produces **fat binary** artifacts containing GPU kernels for multiple architectures embedded in shared libraries. For distribution, these need to be:

1. **Split** - Separate GPU kernels from host code, organize by architecture
2. **Recombined** - Merge splits from multiple build shards into package groups

This enables:
- Smaller package sizes (users download only their GPU architecture)
- Faster builds (parallel builds for different GPU families)
- Better artifact reuse across builds

## Prerequisites

### Tools Required

- Python 3.9+ with rocm-kpack installed
- `clang-offload-bundler` from ROCm LLVM toolchain
- Sufficient disk space (~3x the input artifact size)

### Environment Setup

```bash
# Set environment variables for easy editing
export THEROCK_DIR=/path/to/therock
export RUN_ID=your-github-run-id

# Activate Python virtual environment
source /path/to/your/venv/bin/activate

# Locate clang-offload-bundler
BUNDLER=$(find /path/to/rocm/install -name "clang-offload-bundler" | head -1)
echo "Using bundler: $BUNDLER"
```

## Step 1: Download Artifacts

Download build artifacts from CI using `fetch_artifacts.py`:

```bash
# Example: Download artifacts for three architecture groups
python $THEROCK_DIR/build_tools/fetch_artifacts.py \
  --run-id $RUN_ID \
  --artifact-group gfx110X-dgpu \
  --output-dir /develop/artifacts/gfx110X-build

python $THEROCK_DIR/build_tools/fetch_artifacts.py \
  --run-id $RUN_ID \
  --artifact-group gfx1151 \
  --output-dir /develop/artifacts/gfx1151-build

python $THEROCK_DIR/build_tools/fetch_artifacts.py \
  --run-id $RUN_ID \
  --artifact-group gfx120X-all \
  --output-dir /develop/artifacts/gfx120X-build
```

**Note**: The artifact group names (gfx110X-dgpu, gfx1151, gfx120X-all) are just build job identifiers. The actual GPU architectures inside the binaries may differ and will be discovered during splitting.

## Step 2: Discovery - Understand Actual Architectures

Before splitting all artifacts, split a representative component to discover what GPU architectures are actually present:

```bash
# Create discovery output directory
mkdir -p /develop/tmp/discovery/gfx110X-build

# Split blas_lib from first shard to discover architectures
python python/rocm_kpack/tools/split_artifacts.py \
  --input-dir /develop/artifacts/gfx110X-build/blas_lib_gfx110X-dgpu \
  --output-dir /develop/tmp/discovery/gfx110X-build \
  --component-name blas_lib \
  --tmp-dir /develop/tmp \
  --split-databases rocblas hipblaslt \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

# Check what architectures were discovered
ls /develop/tmp/discovery/gfx110X-build/
# Output: blas_lib_generic blas_lib_gfx1100 blas_lib_gfx1101 blas_lib_gfx1102 blas_lib_gfx906
```

Repeat for other shards:

```bash
# Discover architectures in gfx1151-build
mkdir -p /develop/tmp/discovery/gfx1151-build
python python/rocm_kpack/tools/split_artifacts.py \
  --input-dir /develop/artifacts/gfx1151-build/blas_lib_gfx1151 \
  --output-dir /develop/tmp/discovery/gfx1151-build \
  --component-name blas_lib \
  --tmp-dir /develop/tmp \
  --split-databases rocblas hipblaslt \
  --clang-offload-bundler "$BUNDLER"

ls /develop/tmp/discovery/gfx1151-build/
# Output: blas_lib_generic blas_lib_gfx1151 blas_lib_gfx906

# Discover architectures in gfx120X-build
mkdir -p /develop/tmp/discovery/gfx120X-build
python python/rocm_kpack/tools/split_artifacts.py \
  --input-dir /develop/artifacts/gfx120X-build/blas_lib_gfx120X-all \
  --output-dir /develop/tmp/discovery/gfx120X-build \
  --component-name blas_lib \
  --tmp-dir /develop/tmp \
  --split-databases rocblas hipblaslt \
  --clang-offload-bundler "$BUNDLER"

ls /develop/tmp/discovery/gfx120X-build/
# Output: blas_lib_generic blas_lib_gfx1200 blas_lib_gfx1201 blas_lib_gfx906
```

**Discovered Architectures:**
- **gfx110X-build**: gfx1100, gfx1101, gfx1102
- **gfx1151-build**: gfx1151
- **gfx120X-build**: gfx1200, gfx1201

**Known Issue - gfx906 Fallback Bug:**
All shards contain `gfx906` artifacts. This is a build system bug where clang defaults to gfx906 when libraries don't support the requested architectures. These should be ignored for now and will need to be fixed in the build system later.

## Step 3: Create Packaging Configuration

Based on discovered architectures, create a JSON configuration defining package groups:

```bash
cat > /develop/tmp/packaging-config.json << 'EOF'
{
  "primary_shard": "gfx110X-build",
  "architecture_groups": {
    "gfx110X": {
      "display_name": "ROCm gfx110X (RDNA3)",
      "architectures": [
        "gfx1100",
        "gfx1101",
        "gfx1102"
      ]
    },
    "gfx115X": {
      "display_name": "ROCm gfx115X (RDNA3.5)",
      "architectures": [
        "gfx1151"
      ]
    },
    "gfx120X": {
      "display_name": "ROCm gfx120X (RDNA4)",
      "architectures": [
        "gfx1200",
        "gfx1201"
      ]
    }
  },
  "validation": {
    "error_on_duplicate_device_code": true,
    "verify_generic_artifacts_match": false,
    "error_on_missing_architecture": false
  }
}
EOF
```

**Configuration Fields:**
- `primary_shard`: Which shard to use for generic (host-code) artifacts
- `architecture_groups`: Package groups and which GPU architectures go in each
- `validation`: Rules for handling edge cases during recombination

## Step 4: Split All Artifacts

Create a script to split all artifacts from each shard systematically:

```bash
cat > /develop/tmp/split-all.sh << 'EOF'
#!/bin/bash
set -e

BUNDLER="/home/stella/workspace/rocm/gfx1100/lib/llvm/bin/clang-offload-bundler"
TMP_DIR="/develop/tmp"

# Function to split all arch-specific artifacts from a shard
split_shard() {
    local shard_name=$1
    local shard_dir="/develop/artifacts/${shard_name}"
    local output_dir="/develop/tmp/split-artifacts/${shard_name}"

    echo "========================================="
    echo "Splitting artifacts from ${shard_name}"
    echo "========================================="

    source /develop/therock-venv/bin/activate
    mkdir -p "$output_dir"

    local total=0
    local success=0

    for artifact_dir in "$shard_dir"/*; do
        [ ! -d "$artifact_dir" ] && continue
        local artifact_name=$(basename "$artifact_dir")

        # Skip generic artifacts (host-* prefix or _generic suffix)
        if [[ "$artifact_name" == host-* ]] || [[ "$artifact_name" == *_generic ]]; then
            continue
        fi

        # Skip if doesn't contain architecture suffix
        if [[ ! "$artifact_name" =~ gfx[0-9] ]]; then
            continue
        fi

        # Extract component name by removing architecture suffix
        local component_name=$(echo "$artifact_name" | sed 's/_gfx[0-9].*$//')

        total=$((total + 1))
        echo "[$total] Processing: $artifact_name (component: $component_name)"

        # Determine database splitting requirements
        local db_args=""
        if [[ "$component_name" == blas_* ]]; then
            db_args="--split-databases rocblas hipblaslt"
        fi

        # Run split
        if python /develop/rocm-kpack/python/rocm_kpack/tools/split_artifacts.py \
            --input-dir "$artifact_dir" \
            --output-dir "$output_dir" \
            --component-name "$component_name" \
            --tmp-dir "$TMP_DIR" \
            $db_args \
            --clang-offload-bundler "$BUNDLER" \
            2>&1 | grep -q "Splitting complete"; then
            success=$((success + 1))
            echo "    ✓ Success"
        else
            echo "    ✗ Failed"
        fi
    done

    echo "Summary: $success/$total succeeded"
    echo ""
}

# Split all three shards
split_shard "gfx110X-build"
split_shard "gfx1151-build"
split_shard "gfx120X-build"
EOF

chmod +x /develop/tmp/split-all.sh
/develop/tmp/split-all.sh
```

This will process all architecture-specific artifacts from all three shards. Expected output:
- **gfx110X-build**: ~54 artifacts
- **gfx1151-build**: ~54 artifacts
- **gfx120X-build**: ~54 artifacts

**What Split Does:**

For each artifact:
1. Scans for fat binaries (shared libraries with embedded GPU code)
2. Unbundles GPU kernels using `clang-offload-bundler`
3. Creates `.kpack` archive files per architecture
4. Strips GPU code from shared libraries (PROGBITS → NOBITS)
5. Adds `.rocm_kpack_ref` marker pointing to `.kpack` files
6. For databases (rocBLAS, hipBLASLt): separates by architecture
7. Creates generic artifact (host code + markers)
8. Creates arch-specific artifacts (GPU kernels + databases)

**Output Structure:**
```
/develop/tmp/split-artifacts/
├── gfx110X-build/
│   ├── blas_lib_generic/           # Host code, no GPU kernels
│   ├── blas_lib_gfx1100/           # gfx1100 GPU kernels + databases
│   ├── blas_lib_gfx1101/
│   ├── blas_lib_gfx1102/
│   ├── fft_lib_generic/
│   ├── fft_lib_gfx1100/
│   └── ...
├── gfx1151-build/
│   ├── blas_lib_generic/
│   ├── blas_lib_gfx1151/
│   └── ...
└── gfx120X-build/
    ├── blas_lib_generic/
    ├── blas_lib_gfx1200/
    ├── blas_lib_gfx1201/
    └── ...
```

## Step 5: Verify Split Artifacts

Verify that splits meet all invariants:

```bash
# Verify each shard
python python/rocm_kpack/tools/verify_artifacts.py \
  --artifacts-dir /develop/tmp/split-artifacts/gfx110X-build \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

python python/rocm_kpack/tools/verify_artifacts.py \
  --artifacts-dir /develop/tmp/split-artifacts/gfx1151-build \
  --clang-offload-bundler "$BUNDLER" \
  --verbose

python python/rocm_kpack/tools/verify_artifacts.py \
  --artifacts-dir /develop/tmp/split-artifacts/gfx120X-build \
  --clang-offload-bundler "$BUNDLER" \
  --verbose
```

**Verification Checks:**
1. ✓ All artifact manifests present and valid
2. ✓ Fat binaries converted (PROGBITS → NOBITS)
3. ✓ Architecture separation (no cross-contamination)
4. ✓ Kpack archives valid (KPAK magic, MessagePack TOC)

**Note**: The "Fat Binary Conversion" check may show as failed with "0 binaries found" - this is expected and not a real failure. The conversion happens during splitting, not post-processing.

## Step 6: Recombine into Package Groups

Combine split artifacts from all shards according to the packaging configuration:

```bash
python python/rocm_kpack/tools/recombine_artifacts.py \
  --input-shards-dir /develop/tmp/split-artifacts \
  --config /develop/tmp/packaging-config.json \
  --output-dir /develop/tmp/recombined-artifacts \
  --verbose
```

**What Recombine Does:**

For each component and each architecture group:
1. Copies generic artifact from primary shard (gfx110X-build)
2. Collects arch-specific artifacts from all shards
3. Copies `.kpack` files for all architectures in the group
4. Copies architecture-specific database files
5. Merges `.kpm` manifest files
6. Creates final combined artifact

**Output:**
```
/develop/tmp/recombined-artifacts/
├── blas_lib_gfx110X/      # Contains gfx1100, gfx1101, gfx1102 kernels
├── blas_lib_gfx115X/      # Contains gfx1151 kernels
├── blas_lib_gfx120X/      # Contains gfx1200, gfx1201 kernels
├── fft_lib_gfx110X/
├── fft_lib_gfx115X/
├── fft_lib_gfx120X/
└── ...                     # 105 total (35 components × 3 groups)
```

**Current Behavior (INCORRECT - See Known Issues):**
Each recombined artifact currently contains:
- Generic (host) shared libraries with `.rocm_kpack_ref` markers
- `.kpack` files for all architectures in the group
- `.kpm` manifest listing all kpack files
- Architecture-specific database files

**Expected Behavior (TO BE IMPLEMENTED):**
- **Generic artifact** (`blas_lib_generic`): Host shared libraries only
- **Arch-specific artifacts** (`blas_lib_gfx110X`, etc.): `.kpack` files and arch databases only

## Step 7: Inspect Results

Examine a recombined artifact to understand the structure:

```bash
# Check blas_lib for RDNA3 (gfx110X) - Note: Structure shown is CURRENT, not CORRECT
tree -L 4 /develop/tmp/recombined-artifacts/blas_lib_gfx110X/

# Current Output (INCORRECT - host code should be in generic artifact):
# blas_lib_gfx110X/
# ├── artifact_manifest.txt
# └── math-libs/
#     └── BLAS/
#         ├── rocBLAS/
#         │   └── stage/
#         │       ├── .kpack/
#         │       │   ├── blas_lib_gfx1100.kpack
#         │       │   ├── blas_lib_gfx1101.kpack
#         │       │   ├── blas_lib_gfx1102.kpack
#         │       │   └── blas_lib.kpm
#         │       └── lib/
#         │           ├── librocblas.so.5.2  # ← WRONG: Should be in generic artifact
#         │           └── rocblas/library/
#         │               ├── TensileLibrary_*_gfx1100.dat
#         │               ├── TensileLibrary_*_gfx1101.dat
#         │               └── TensileLibrary_*_gfx1102.dat
#         └── hipBLASLt/
#             └── ...

# Check manifest
cat /develop/tmp/recombined-artifacts/blas_lib_gfx110X/artifact_manifest.txt

# Inspect .kpm manifest (MessagePack format)
python -c "
import msgpack
with open('/develop/tmp/recombined-artifacts/blas_lib_gfx110X/math-libs/BLAS/rocBLAS/stage/.kpack/blas_lib.kpm', 'rb') as f:
    manifest = msgpack.unpack(f, raw=False)
    print('Component:', manifest['component_name'])
    print('Architectures:', list(manifest['kpack_files'].keys()))
    for arch, entry in manifest['kpack_files'].items():
        print(f'  {arch}: {entry[\"filename\"]} ({entry[\"size\"]} bytes, {entry[\"kernel_count\"]} kernels)')
"
```

## Common Issues and Solutions

### Issue: clang-offload-bundler not found

**Error**: `Could not find tool 'clang-offload-bundler' on system path`

**Solution**: Specify path explicitly with `--clang-offload-bundler` flag:
```bash
find /path/to/rocm -name "clang-offload-bundler"
# Use the path in your split command
```

### Issue: Out of disk space

**Error**: `No space left on device`

**Solution**: The workflow requires ~3x input artifact size:
- Input artifacts: 38 GB
- Split artifacts: ~40 GB
- Recombined artifacts: ~30 GB
- Temporary files: ~10 GB

Use `/develop/tmp` for temporary files (not `/tmp`) per CLAUDE.md instructions.

### Issue: gfx906 artifacts everywhere

**Problem**: All shards produce gfx906 artifacts unexpectedly.

**Explanation**: This is a known build system bug. When a library doesn't support the requested architectures, clang defaults to gfx906. These artifacts should be excluded from package groups for now. The build system needs to be fixed to handle this properly.

### Issue: Binary not stripped or grew in size

**Warning**: `Binary was not stripped or grew in size: /path/to/binary`

**Explanation**: Some test binaries or specific libraries may not shrink after stripping device code. This is usually harmless and doesn't affect functionality. The warning appears when:
- Binary is already host-only (no device code to strip)
- Binary has very small or zero-size .hip_fatbin section
- Binary structure makes it difficult to remove sections efficiently

This doesn't indicate a failure - the process still completes successfully.

## Next Steps

After recombination, these artifacts are ready for:
1. **Package creation** - Create .deb/.rpm packages from recombined artifacts
2. **Distribution** - Upload to package repositories
3. **Installation testing** - Verify packages install and work on target GPUs

## Reference

- **Tool source**: `/develop/rocm-kpack/python/rocm_kpack/tools/`
- **Design document**: `/develop/rocm-kpack/docs/kpack-build-integration.md`

## Appendix: File Formats

### artifact_manifest.txt
Plain text file listing installation prefixes (one per line):
```
math-libs/BLAS/rocBLAS/stage
math-libs/BLAS/hipBLASLt/stage
```

### .kpack (Kernel Pack Archive)
Binary archive format containing GPU kernels:
- Magic: `KPAK` (0x4b50414b)
- TOC: MessagePack table of contents
- Kernels: Compressed (zstd) or uncompressed code objects

### .kpm (Kernel Pack Manifest)
MessagePack format listing available kpack files:
```python
{
  'format_version': 1,
  'component_name': 'blas_lib',
  'prefix': 'math-libs/BLAS/rocBLAS/stage',
  'kpack_files': {
    'gfx1100': {
      'architecture': 'gfx1100',
      'filename': 'blas_lib_gfx1100.kpack',
      'size': 3456789,
      'kernel_count': 3
    },
    ...
  }
}
```

### .rocm_kpack_ref (Kpack Reference Marker)
Plain text format pointing to kpack search paths:
```
../../../.kpack:../../.kpack:.kpack
```

This file is mapped into a PT_LOAD segment and referenced by the `__CudaFatBinaryWrapper` to locate kernels at runtime.
