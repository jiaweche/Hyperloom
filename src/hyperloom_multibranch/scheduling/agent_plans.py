# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each Arbor agent's share of the session's directions, split before EXPLORE opens the agent branches.

The user names one directions map with ``--agent-directions``: the open directions of an Arbor dossier map, or a list
written by hand, in the form Arbor's ``component.plan`` reads. Two Arbor agents given the whole map would work the
same directions. Split, each works its own. Before the agent branches open, each agent source is fetched once, and the
map is split by Arbor's own command line (:func:`~hyperloom_multibranch.specialist.arbor_adapter.split_directions`)
into as many disjoint shares as there are Arbor agents, one each, in the order the user named them. Any other agent
gets none: only Arbor reads a plan.

A map that cannot be split leaves every branch without a share, as without a map, and the log says why: a source that
cannot be fetched, no Arbor among the agents, or fewer independent directions than Arbor agents.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom_multibranch.specialist.agent_onboarding import AttemptFailed, fetch_source
from hyperloom_multibranch.specialist.arbor_adapter import SplitFailed, is_arbor, split_directions

logger = logging.getLogger(__name__)

SOURCES_DIR = "sources"
SHARES_DIR = "shares"


async def agent_plans(agents: tuple[str, ...], map_file: str, work_dir: Path) -> tuple[str, ...]:
    """Each of ``agents``' share of the directions in ``map_file``, in order: the absolute path of an Arbor agent's
    share, ``""`` for any other agent, and every one ``""`` without a map or when it cannot be split. ``work_dir``
    keeps the fetched sources and the shares."""

    none = ("",) * len(agents)
    if not map_file or not agents:
        return none
    try:
        checkouts = await _checkouts(agents, work_dir / SOURCES_DIR)
    except (AttemptFailed, OSError) as exc:
        logger.error("the agents' directions are not split: an agent source cannot be fetched: %s", exc)
        return none
    arbors = [index for index, agent in enumerate(agents) if checkouts[agent] is not None]
    if not arbors:
        logger.error("the agents' directions are not split: no agent named is Arbor")
        return none
    shares_dir = work_dir / SHARES_DIR
    shares_dir.mkdir(parents=True, exist_ok=True)
    try:
        shares = await split_directions(checkouts[agents[arbors[0]]], Path(map_file), len(arbors), shares_dir)
    except SplitFailed as exc:
        logger.error("the agents' directions are not split: %s", exc)
        return none
    plans = list(none)
    for index, share in zip(arbors, shares):
        plans[index] = str(share.absolute())
    logger.info("the directions of %s split into %d shares, one per Arbor agent", map_file, len(shares))
    return tuple(plans)


async def _checkouts(agents: tuple[str, ...], root: Path) -> dict[str, Path | None]:
    """Each distinct source of ``agents`` fetched once under ``root``, ``None`` for one that is not Arbor."""

    checkouts: dict[str, Path | None] = {}
    for index, source in enumerate(dict.fromkeys(agents)):
        path = root / str(index)
        await fetch_source(source, path)
        checkouts[source] = path if is_arbor(path) else None
    return checkouts
