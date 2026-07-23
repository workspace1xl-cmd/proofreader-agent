from __future__ import annotations

import re
import subprocess
from pathlib import Path

HTML = Path("static/index.html")


def test_required_controls_and_workflows_are_present() -> None:
    source = HTML.read_text()
    required_ids = {
        "submitBtn",
        "cancelBtn",
        "findingSearch",
        "historySearch",
        "acceptAllBtn",
        "rejectAllBtn",
        "expDocx",
        "expMd",
        "expCsv",
        "expJson",
        "expAudit",
        "expPrint",
    }
    for element_id in required_ids:
        assert f'id="{element_id}"' in source
    assert "c.verified === true" in source
    assert "if(/^[=+\\-@]/.test(v))" in source


def test_inline_javascript_parses_with_node() -> None:
    source = HTML.read_text()
    scripts = re.findall(r"<script>([\s\S]*?)</script>", source)
    assert len(scripts) == 1
    command = (
        "const vm=require('vm');"
        f"new vm.Script({scripts[0]!r},{{filename:'static/index.html'}});"
    )
    result = subprocess.run(
        ["node", "-e", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
