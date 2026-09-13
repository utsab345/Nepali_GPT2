"""Smoke tests for the CLI dispatcher.

Guard against a subtle regression: ``__init__`` re-exports a *function*
named ``generate``, which shadows the ``generate`` module. The dispatcher
must resolve each sub-command to the real module's entry point.
"""

import sys

import nepali_gpt2.__main__ as dispatcher


def test_commands_map_to_callables() -> None:
    for name, fn in dispatcher.COMMANDS.items():
        assert callable(fn), f"{name} is not callable"


def test_known_command_dispatches() -> None:
    # generate --help should print usage and exit 0 (argparse exits on help).
    try:
        code = dispatcher.main(["generate", "--help"])
    except SystemExit as exc:  # argparse raises SystemExit(0) for --help
        code = exc.code
    assert code == 0, f"generate --help failed with code {code}"


def test_unknown_command_returns_nonzero() -> None:
    assert dispatcher.main(["bogus_command"]) == 1


def test_no_args_uses_usage() -> None:
    assert dispatcher.main([]) == 1