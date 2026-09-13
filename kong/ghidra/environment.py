"""Ghidra environment discovery — find Ghidra install and JDK."""

from __future__ import annotations

import glob as _glob
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


_IS_WINDOWS = os.name == "nt"


def _java_executable(java_home: str | Path) -> Path:
    """Path to the java binary inside *java_home* (``java.exe`` on Windows)."""
    return Path(java_home) / "bin" / ("java.exe" if _IS_WINDOWS else "java")


def _expand_globs(patterns: list[str]) -> list[str]:
    """Expand *patterns*, newest-looking match first, skipping non-directories."""
    seen: list[str] = []
    for pattern in patterns:
        for candidate in sorted(_glob.glob(pattern), reverse=True):
            if Path(candidate).is_dir() and candidate not in seen:
                seen.append(candidate)
    return seen


def _windows_program_dirs() -> list[str]:
    """Root directories where Windows installers place JDKs and tools."""
    roots = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramW6432"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("USERPROFILE"),
        "C:\\",
    ]
    return [r for r in roots if r]


def _windows_jdk_candidates() -> list[str]:
    """Common Windows JDK install locations, newest first."""
    if not _IS_WINDOWS:
        return []
    patterns = []
    for root in _windows_program_dirs():
        patterns.extend([
            str(Path(root) / "Eclipse Adoptium" / "jdk-*"),
            str(Path(root) / "Java" / "jdk-*"),
            str(Path(root) / "Microsoft" / "jdk-*"),
            str(Path(root) / "Amazon Corretto" / "jdk*"),
            str(Path(root) / "Zulu" / "zulu-*"),
            str(Path(root) / "Programs" / "Eclipse Adoptium" / "jdk-*"),
        ])
    return _expand_globs(patterns)


def _windows_ghidra_candidates() -> list[str]:
    """Common Windows Ghidra install locations, newest first."""
    if not _IS_WINDOWS:
        return []
    patterns = []
    for root in _windows_program_dirs():
        patterns.extend([
            str(Path(root) / "ghidra*"),
            str(Path(root) / "Ghidra*"),
            str(Path(root) / "ghidra*" / "ghidra_*"),
        ])
    return _expand_globs(patterns)


def _is_ghidra_dir(path: Path) -> bool:
    """True if *path* looks like a Ghidra installation root."""
    support = path / "support"
    return (support / "analyzeHeadless").exists() or (
        support / "analyzeHeadless.bat"
    ).exists()


def _java_version(java_home: str) -> int | None:
    """Return the major version of the JDK at *java_home*, or None on failure."""
    java_bin = _java_executable(java_home)
    if not java_bin.exists():
        return None
    try:
        result = subprocess.run(
            [str(java_bin), "-version"],
            capture_output=True, text=True, timeout=10,
        )
        # `java -version` prints to stderr, e.g. 'openjdk version "21.0.2"'
        output = result.stderr or result.stdout
        match = re.search(r'"(\d+)', output)
        return int(match.group(1)) if match else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def find_java_home(min_version: int = 21) -> str | None:
    """Auto-detect a JDK 21+ installation for Ghidra.

    Checks in order:
      1. ``JAVA_HOME`` environment variable (if it points to JDK 21+)
      2. macOS: ``/usr/libexec/java_home -v 21+``
      3. Homebrew: ``brew --prefix openjdk@21``
      4. Homebrew: ``brew --prefix openjdk``
      5. Windows: Adoptium / Oracle / Microsoft / Corretto / Zulu install dirs
    """
    # 1. Existing JAVA_HOME — only if version is sufficient
    env_java = os.environ.get("JAVA_HOME")
    if env_java and Path(env_java).is_dir():
        ver = _java_version(env_java)
        if ver is not None and ver >= min_version:
            return env_java
        logger.debug("JAVA_HOME=%s is JDK %s, need %d+", env_java, ver, min_version)

    # 2. macOS java_home utility
    try:
        result = subprocess.run(
            ["/usr/libexec/java_home", "-v", "21+"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            java_home = result.stdout.strip()
            if java_home and Path(java_home).is_dir():
                ver = _java_version(java_home)
                if ver is not None and ver >= min_version:
                    return java_home
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3-4. Homebrew openjdk
    for pkg in ["openjdk@21", "openjdk"]:
        try:
            result = subprocess.run(
                ["brew", "--prefix", pkg],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                prefix = Path(result.stdout.strip())
                # Homebrew JDK layout: <prefix>/libexec/openjdk.jdk/Contents/Home
                jdk_home = prefix / "libexec" / "openjdk.jdk" / "Contents" / "Home"
                if jdk_home.is_dir():
                    return str(jdk_home)
                # Fallback: prefix itself might be JAVA_HOME
                if _java_executable(prefix).exists():
                    return str(prefix)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # 5. Windows install locations
    for candidate in _windows_jdk_candidates():
        ver = _java_version(candidate)
        if ver is not None and ver >= min_version:
            return candidate

    return None


def find_ghidra_install() -> str | None:
    r"""Auto-detect Ghidra installation directory.

    Checks in order:
      1. ``GHIDRA_INSTALL_DIR`` environment variable
      2. Homebrew: ``brew --prefix ghidra`` → ``<prefix>/libexec``
      3. Common POSIX paths: ``/opt/ghidra*``, ``/Applications/ghidra*``
      4. Windows: ``%ProgramFiles%\ghidra*``, ``C:\ghidra*``, ``%USERPROFILE%\ghidra*``

    Returns the path as a string, or ``None`` if not found.
    """
    # 1. Environment variable
    env_dir = os.environ.get("GHIDRA_INSTALL_DIR")
    if env_dir and Path(env_dir).is_dir():
        return env_dir

    # 2. Homebrew
    try:
        result = subprocess.run(
            ["brew", "--prefix", "ghidra"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            prefix = Path(result.stdout.strip())
            libexec = prefix / "libexec"
            if libexec.is_dir():
                return str(libexec)
            if prefix.is_dir():
                return str(prefix)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Common POSIX paths
    for pattern in ["/opt/ghidra*", "/Applications/ghidra*", "/Applications/Ghidra*"]:
        for candidate in sorted(_glob.glob(pattern), reverse=True):
            p = Path(candidate)
            if p.is_dir() and _is_ghidra_dir(p):
                return str(p)

    # 4. Windows install locations
    for candidate in _windows_ghidra_candidates():
        if _is_ghidra_dir(Path(candidate)):
            return candidate

    return None
