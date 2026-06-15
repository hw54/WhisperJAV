import shlex
from types import SimpleNamespace

from whisperjav.webview_gui import api as gui_api


def _mock_render_group(monkeypatch, *, current_groups):
    monkeypatch.setattr(gui_api.os, "name", "posix", raising=False)
    monkeypatch.setattr(gui_api.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(gui_api.os, "getgroups", lambda: current_groups)
    monkeypatch.setattr(gui_api.os, "getgid", lambda: 1000)
    monkeypatch.setattr(gui_api.os, "getegid", lambda: 1000)
    monkeypatch.setenv("USER", "hinswong")
    monkeypatch.setattr(
        gui_api.grp,
        "getgrnam",
        lambda name: SimpleNamespace(gr_gid=992, gr_mem=["hinswong"]),
    )
    monkeypatch.setattr(
        gui_api.shutil,
        "which",
        lambda name: "/usr/bin/sg" if name == "sg" else None,
    )


def test_wraps_gui_command_with_sg_render_when_session_lacks_render_group(monkeypatch):
    _mock_render_group(monkeypatch, current_groups=[1000, 27])

    cmd = [
        "/env/bin/python",
        "-X",
        "utf8",
        "-m",
        "whisperjav.main",
        "/path with spaces/video.mp4",
    ]

    wrapped = gui_api._wrap_command_for_render_group(cmd)

    assert wrapped[:3] == ["sg", "render", "-c"]
    assert "whisperjav.main" in wrapped[3]
    assert shlex.quote("/path with spaces/video.mp4") in wrapped[3]


def test_keeps_gui_command_direct_when_session_already_has_render_group(monkeypatch):
    _mock_render_group(monkeypatch, current_groups=[1000, 992])
    cmd = ["/env/bin/python", "-m", "whisperjav.main"]

    assert gui_api._wrap_command_for_render_group(cmd) == cmd
