"""Check command discovery and dispatch without running model workloads."""

import pytest

from endless_voices.cli import COMMANDS, main


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_reaches_its_parser(command, capsys):
    with pytest.raises(SystemExit) as stopped:
        main([command, "--help"])
    assert stopped.value.code == 0
    assert f"endless-voices {command}" in capsys.readouterr().out


def test_unknown_command_fails_before_dispatch(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["missing-command"])
    assert stopped.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
