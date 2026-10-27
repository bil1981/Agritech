from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_script_exists():
    assert (Path(__file__).resolve().parents[1] / "app.py").is_file()


def test_streamlit_app_renders_agritech_tabs():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert len(app.tabs) == 3
