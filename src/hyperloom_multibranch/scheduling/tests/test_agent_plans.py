# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The session's directions split into one share per Arbor agent (:mod:`hyperloom_multibranch.scheduling.agent_plans`),
with a fake Arbor whose split deals the open directions round-robin."""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom_multibranch.scheduling.agent_plans import agent_plans
from hyperloom_multibranch.tests.test_specialist_arbor_adapter import CHECKOUT_FILES, MAP, _committed, _write


def _map(tmp_path: Path, document: object = MAP) -> str:
    path = tmp_path / "map.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _ids(plan: str) -> list[str]:
    return [entry["id"] for entry in json.loads(Path(plan).read_text(encoding="utf-8"))["directions"]]


def _agents(tmp_path: Path) -> tuple[str, str]:
    arbor = _committed(_write(tmp_path / "arbor", CHECKOUT_FILES))
    other = _committed(_write(tmp_path / "other", {"README.md": "another agent\n"}))
    return str(arbor), str(other)


async def test_each_arbor_agent_gets_its_own_share_in_order_and_another_agent_none(tmp_path: Path) -> None:
    arbor, other = _agents(tmp_path)

    plans = await agent_plans((arbor, other, arbor), _map(tmp_path), tmp_path / "work")

    assert plans[1] == ""
    assert [_ids(plans[0]), _ids(plans[2])] == [["m-attn", "m-allreduce"], ["m-moe"]]
    assert Path(plans[0]).is_absolute() and Path(plans[2]).is_absolute()
    assert len(list((tmp_path / "work" / "sources").iterdir())) == 2


async def test_no_branch_gets_a_share_without_a_map_an_arbor_enough_directions_or_a_source(tmp_path: Path) -> None:
    arbor, other = _agents(tmp_path)
    one = {"directions": [{"id": "m-attn", "kind": "increment", "title": "attention decode path"}]}

    assert await agent_plans((arbor, arbor), "", tmp_path / "none") == ("", "")
    assert await agent_plans((other,), _map(tmp_path), tmp_path / "other-only") == ("",)
    assert await agent_plans((str(tmp_path / "nowhere"), arbor), _map(tmp_path), tmp_path / "unfetched") == ("", "")
    assert await agent_plans((arbor, arbor), _map(tmp_path, one), tmp_path / "too-few") == ("", "")
