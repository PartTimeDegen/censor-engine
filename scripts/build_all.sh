#!/usr/bin/env bash
set -e

# ------------------------
# Common configuration
# ------------------------

APP_NAME="censor-engine"
SRC="src/main.py"

# Hidden imports
HIDDEN_IMPORTS=(
    "onnxruntime"
    "onnxruntime.capi._pybind_state"
    "cv2"
    "nudenet"
    "ffmpeg"
    "PIL"
    "pyyaml"
    "yaml"
)

# Convert hidden imports into PyInstaller flags
PI_HIDDEN_IMPORTS=""
for mod in "${HIDDEN_IMPORTS[@]}"; do
    PI_HIDDEN_IMPORTS="$PI_HIDDEN_IMPORTS --hidden-import $mod"
done

# ------------------------
# Linux build (WSL or native Linux)
# ------------------------
LINUX_DIST="build/linux"
echo "🐧 Building Linux binary..."
uv run --dev pyinstaller --name "$APP_NAME" \
    --onefile \
    --console \
    $PI_HIDDEN_IMPORTS \
    $SRC \
    --distpath "$LINUX_DIST" \
    --workpath "$LINUX_DIST/temp" \
    --specpath "$LINUX_DIST/spec"

# ------------------------
# Windows build
# ------------------------
WINDOWS_DIST="build/windows"

# Detect if running in WSL or Linux
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    echo "🪟 Building Windows .exe from WSL..."
    
    # Convert WSL path to Windows path
    WIN_PWD=$(wslpath -w "$(pwd)")
    
    # Use native Windows Python (adjust path if needed)
    powershell.exe -NoProfile -Command "& {
        cd '$WIN_PWD';
        python -m PyInstaller --name '$APP_NAME' \
            --onefile \
            --console \
            $PI_HIDDEN_IMPORTS \
            'src\main.py' \
            --distpath 'build/windows' \
            --workpath 'build/windows/temp' \
            --specpath 'build/windows/spec'
    }"
else
    echo "🪟 Building Windows .exe (native Windows)..."
    python -m PyInstaller --name "$APP_NAME" \
        --onefile \
        --console \
        $PI_HIDDEN_IMPORTS \
        $SRC \
        --distpath "$WINDOWS_DIST" \
        --workpath "$WINDOWS_DIST/temp" \
        --specpath "$WINDOWS_DIST/spec"
fi

echo "✅ Build complete!"