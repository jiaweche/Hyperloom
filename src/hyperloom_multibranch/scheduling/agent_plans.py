# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Each Arbor agent's share of the session's directions, split before EXPLORE opens the agent branches.

The user names one directions map with ``--agent-directions``: the open directions of an Arbor dossier map, or a list
written by hand, in the form Arbor's ``component.plan`` reads. Two Arbor agents given the whole map would work the
same directions. Split, each works its own. Before the agent branches open, each agent source is fetched once, and the
map is split by Arbor's own command line (:func:`~hyperloom_multibranch.specialist.arbor_adapter.split_directions`)
into as many disjoint shares as there are Arbor agents, one each, in the order the user named them. Any other agent
gets none: only Arbor reads a plan.

Without a map, two or more Arbor agents split one Arbor generates from the root's op shares, PRELUDE's profile, with no
GPU time (:func:`~hyperloom_multibranch.specialist.arbor_adapter.generate_directions`): one direction per op family,
ranked by its share. A single Arbor gets none and maps its own directions, which its dossier measures better.

A map that cannot be split, or generated, leaves every branch without a share, as without a map, and the log says why:
a source that cannot be fetched, no Arbor among the agents, fewer independent directions than Arbor agents, or a root
with no op shares Arbor can use.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist.agent_onboarding import AttemptFailed, fetch_source, root_profile
from hyperloom_multibranch.specialist.arbor_adapter import SplitFailed, generate_directions, is_arbor, split_directions

logger = logging.getLogger(__name__)

SOURCES_DIR = "sources"
SHARES_DIR = "shares"
#: The map generated from the root's op shares, beside the profile it was generated from.
GENERATED_MAP = "generated-directions.json"
#: The fewest Arbor agents a generated map is split for; one Arbor maps its own directions.
GENERATED_FOR = 2


async def agent_plans(
    agents: tuple[str, ...], map_file: str, work_dir: Path, root_shares: OpShares | None = None
) -> tuple[str, ...]:
    """Each of ``agents``' share of the directions in ``map_file``, or, without one, of a map generated from
    ``root_shares`` for two or more Arbors, in order: the absolute path of an Arbor agent's share, ``""`` for any other
    agent, and every one ``""`` when there is no map or it cannot be split. ``work_dir`` keeps the fetched sources, a
    generated map and the shares."""

    none = ("",) * len(agents)
    generating = not map_file and root_shares is not None and len(agents) >= GENERATED_FOR
    if not agents or not (map_file or generating):
        return none
    try:
        checkouts = await _checkouts(agents, work_dir / SOURCES_DIR)
    except (AttemptFailed, OSError) as exc:
        logger.error("the agents' directions are not split: an agent source cannot be fetched: %s", exc)
        return none
    arbors = {index: path for index, agent in enumerate(agents) if (path := checkouts[agent]) is not None}
    if generating and len(arbors) < GENERATED_FOR:
        logger.info("no directions map is generated: %d Arbor agents, each maps its own", len(arbors))
        return none
    if not arbors:
        logger.error("the agents' directions are not split: no agent named is Arbor")
        return none
    try:
        shares = await _split(next(iter(arbors.values())), map_file, root_shares, len(arbors), work_dir)
    except SplitFailed as exc:
        logger.error("the agents' directions are not split: %s", exc)
        return none
    plans = list(none)
    for index, share in zip(arbors, shares):
        plans[index] = str(share.absolute())
    logger.info("the directions of %s split into %d shares, one per Arbor agent", map_file or "the root", len(shares))
    return tuple(plans)


async def _split(
    arbor: Path, map_file: str, root_shares: OpShares | None, shares: int, work_dir: Path
) -> tuple[Path, ...]:
    """``map_file``, or a map ``arbor`` generates from ``root_shares`` without one, split into ``shares`` plans."""

    if not map_file:
        profile = root_profile(root_shares, work_dir)
        if profile is None:
            raise SplitFailed("the root has no op shares to generate a map from")
        map_file = str(await generate_directions(arbor, profile, work_dir / GENERATED_MAP))
    out = work_dir / SHARES_DIR
    out.mkdir(parents=True, exist_ok=True)
    return await split_directions(arbor, Path(map_file), shares, out)


async def _checkouts(agents: tuple[str, ...], root: Path) -> dict[str, Path | None]:
    """Each distinct source of ``agents`` fetched once under ``root``, ``None`` for one that is not Arbor."""

    checkouts: dict[str, Path | None] = {}
    for index, source in enumerate(dict.fromkeys(agents)):
        path = root / str(index)
        await fetch_source(source, path)
        checkouts[source] = path if is_arbor(path) else None
    return checkouts
