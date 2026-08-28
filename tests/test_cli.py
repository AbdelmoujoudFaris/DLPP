import pytest

from interfaceshapeai.cli import build_parser, main


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_interface_subcommand_runs(synthetic_pdb_path, capsys):
    exit_code = main(["interface", "--input", str(synthetic_pdb_path), "--chain-a", "A", "--chain-b", "B"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "residue_number" in captured.out


def test_interface_subcommand_reports_no_interface(synthetic_pdb_path, capsys):
    exit_code = main(
        ["interface", "--input", str(synthetic_pdb_path), "--chain-a", "A", "--chain-b", "B",
         "--distance-cutoff", "0.01"]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "No interface detected" in captured.err


def test_build_parser_has_subcommands():
    parser = build_parser()
    assert parser.prog == "interfaceshapeai"


def test_explain_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["explain", "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "gradcam" in captured.out


def test_explain_subcommand_runs_end_to_end(synthetic_pdb_path, capsys):
    exit_code = main(
        ["explain", "--input", str(synthetic_pdb_path), "--chain-a", "A", "--chain-b", "B", "--method", "saliency"]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "residue_name" in captured.out
