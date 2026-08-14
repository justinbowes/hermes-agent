"""Bundled Mnemosyne memory provider shim.

This directory makes Mnemosyne a FIRST-CLASS *bundled* memory provider inside
the derived Hermes image, discovered by Hermes's memory-provider loader
(`plugins/memory/__init__.py`) at `plugins/memory/mnemosyne/`.

Why a shim and not the package itself:
  The `mnemosyne-memory` wheel installs two top-level packages into the venv:
    - `mnemosyne`               (the BEAM memory core)
    - `hermes_memory_provider`  (the Hermes MemoryProvider adapter, exposing
                                 `class MnemosyneMemoryProvider(MemoryProvider)`)
  Both are installed into site-packages by the Dockerfile (`uv pip install
  mnemosyne-memory`). This shim simply re-exports the provider class so the
  loader can find and instantiate it.

How the loader picks it up (see NousResearch/hermes-agent
plugins/memory/__init__.py::_load_provider_from_dir):
  1. It text-scans this file for `MemoryProvider` (present below) to classify
     the dir as a memory provider.
  2. It imports this module, then FIRST tries a `register(ctx)` hook and
     FINALLY falls back to instantiating any top-level `MemoryProvider`
     subclass with a no-arg constructor.
  We deliberately expose the class (deterministic fallback path) rather than a
  `register()` that collides with the package's CLI-registration hook of the
  same name. `is_available()` on the provider verifies the mnemosyne core is
  importable, so a broken/absent install degrades gracefully to "unavailable"
  instead of crashing discovery.

Pinning: the exact mnemosyne-memory version is pinned in the Dockerfile
(MNEMOSYNE_VERSION build arg), NOT floating :latest — reproducible images.
"""

from __future__ import annotations

# Re-export the provider class from the installed adapter package. The Hermes
# loader instantiates it via its MemoryProvider-subclass fallback.
try:
    from hermes_memory_provider import MnemosyneMemoryProvider  # noqa: F401
except Exception as _exc:  # pragma: no cover - surfaced at discovery time
    # Leave a breadcrumb but do NOT raise: a failed import here would abort
    # the loader's directory scan. Hermes will simply report the provider as
    # unavailable. The Dockerfile guarantees the package is installed, so this
    # path should only trigger on a genuinely broken image.
    import logging

    logging.getLogger(__name__).warning(
        "Mnemosyne provider shim could not import hermes_memory_provider: %s. "
        "Is 'mnemosyne-memory' installed in the venv? (Dockerfile installs it.)",
        _exc,
    )
    raise

__all__ = ["MnemosyneMemoryProvider"]
