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


def test_preparation_progress_keeps_stdout_json_and_counts_selected_missions(tmp_path, capsys,
                                                                          monkeypatch):
    import json

    from test_dataset_builder import config, corpus, spec

    source = corpus(tmp_path, '''mission Selected
	on offer
		conversation
			`Welcome.`
mission HeldOut
	on offer
		conversation
			`Later.`
''')
    data = tmp_path / 'data'
    (data / 'sources').mkdir(parents=True)
    (data / 'dataset').mkdir()
    (data / 'local').mkdir()
    source.root.rename(data / 'local' / f'endless-sky-{source.revision[:12]}')
    (data / 'sources/source-statistics.json').write_text(json.dumps({
        'revision': source.revision,
        'files': [{'path': p, 'sha256': h} for p, h in source.files.items()]}))
    (data / 'dataset/config.json').write_text(json.dumps(
        config({'Selected': spec(), 'HeldOut': spec('validation')})))
    monkeypatch.chdir(tmp_path)
    main(['prepare-dataset'])
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {'examples': 1, 'missions': 1}
    assert '[0/1 missions complete] Selected' in captured.err
    assert '[1/1 missions complete] Writing dataset' in captured.err
    assert 'Finished in' in captured.err
    assert (data / 'dataset-resolved/manifest.json').is_file()
