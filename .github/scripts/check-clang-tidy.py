"""Check changed translation units using their CMake compilation commands."""

import json
from pathlib import Path
import subprocess
import sys


base, build_directory, clang_tidy = sys.argv[1:]
commands = json.loads((Path(build_directory) / "compile_commands.json").read_text())
compiled_files = {
    (Path(command["directory"]) / command["file"]).resolve()
    for command in commands
}
changed = subprocess.check_output(
    [
        "git", "diff", "--name-only", "--diff-filter=ACMR", "-z",
        f"{base}...HEAD", "--", "*.c", "*.cc", "*.cpp", "*.cxx",
    ]
).decode().split("\0")

# Standalone headers can select unrelated flags or require an unconfigured
# CUDA/MPI toolchain. Check the configured source files instead.
result = 0
for name in filter(None, changed):
    if Path(name).resolve() not in compiled_files:
        print(f"Skipping {name}: no compilation command", flush=True)
        continue
    print(f"Checking {name}", flush=True)
    status = subprocess.run(
        [
            clang_tidy, name, "-p", build_directory, "--header-filter=^$",
            "--warnings-as-errors=*", "--quiet",
        ]
    ).returncode
    if status:
        result = 1

sys.exit(result)
