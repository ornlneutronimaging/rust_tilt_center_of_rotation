#!/bin/bash
# Rebuild (if needed) and launch the tilt & center-of-rotation tool.
cd "$(dirname "$0")" || exit 1
cargo build --release || exit 1
exec ./target/release/tilt_center_of_rotation "$@"
