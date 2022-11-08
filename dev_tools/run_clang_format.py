import argparse
import shutil
import subprocess
import sys

git_exe = shutil.which("git")
if git_exe is None:
    raise RuntimeError("Cannot find git")

clang_format_exe = shutil.which("clang-format")
if clang_format_exe is None:
    raise RuntimeError("Cannot find clang-format")


def run_clang_format(argv) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", help="force changes", action="store_true")

    args = parser.parse_args(argv)

    result = subprocess.run(
        f"{git_exe} ls-files *.cpp *.h", text=True, capture_output=True
    )
    if result.returncode != 0:
        print(
            f"Error: Cannot get list of files from git: {result.stderr}",
            file=sys.stderr,
        )
        return 1

    filenames = result.stdout.split()
    format_args = "" if args.force else "--dry-run"
    any_failed = []
    for filename in filenames:
        result = subprocess.run(
            f"{clang_format_exe} {format_args} {filename} --verbose --Werror -i"
        )
        if result.returncode != 0:
            any_failed.append(filename)
    if any_failed:
        print(f"Error: Failed for {any_failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run_clang_format(sys.argv[1:]))
