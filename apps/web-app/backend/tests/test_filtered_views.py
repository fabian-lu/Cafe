"""Filtered results views (``GET /studies/{id}/results?fkey=…&fval=…``) are stored in the database:
computed once, loaded after a restart, and never served for results they weren't computed from."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from conftest import ITEMS, judged_bundle
from sqlalchemy.dialects import postgresql

pytestmark = pytest.mark.asyncio


async def test_a_filtered_view_is_computed_once_and_loads_after_a_restart(backend):
    study = await backend.add_study()
    first = await backend.view(study)
    assert backend.computed == [("topic", "a")]
    assert await backend.view(study) == first      # from memory
    await backend.restart()
    assert await backend.view(study) == first      # from the database: not computed again
    assert backend.computed == [("topic", "a")]
    await backend.view(study, "b")                 # another value is a view of its own
    assert backend.computed == [("topic", "a"), ("topic", "b")]


async def test_a_real_filtered_view_loads_unchanged_after_a_restart(backend, tmp_path):
    backend.real_statistics = True
    response = await backend.client.post("/api/studies/import", json=judged_bundle(tmp_path))
    assert response.status_code == 201, response.text
    study = response.json()["id"]
    assert await backend.get(f"/api/studies/{study}/filters") == {"topic": ["a", "b"]}

    first = await backend.view(study)
    assert first["filtered"] == {"key": "topic", "value": "a", "n_questions": 4, "n_total": 8}
    assert len(first["records"]) == 4 * 4 and first["effects"]   # refitted on the subset
    await backend.restart()
    assert await backend.view(study) == first
    assert backend.computed == [("topic", "a")]


async def test_a_rerun_drops_the_views_of_the_results_it_replaces(backend):
    response = await backend.client.post("/api/datasets", json={"name": "q", "items": ITEMS})
    response = await backend.client.post("/api/studies", json={
        "name": "toy", "pipeline": "toy", "dataset_id": response.json()["id"],
        "factors": [{"name": "reply", "levels": ["short", "long"]}]})
    study = response.json()["id"]
    await backend.run(study)
    first = await backend.view(study)
    assert len(await backend.stored_views()) == 1

    await backend.run(study)                       # replaces the results...
    assert await backend.stored_views() == []      # ...and drops their views
    second = await backend.view(study)
    current = await backend.get(f"/api/studies/{study}/results")
    assert second["computed_from"] == current["computed_at"] != first["computed_from"]
    assert len(backend.computed) == 2


async def test_a_statistics_refresh_drops_the_views_of_the_old_payload(backend):
    study = await backend.add_study()
    old = await backend.view(study)
    backend.stats_version_bump = 1                 # an update brings newer statistics
    await backend.restart()

    results = await backend.get(f"/api/studies/{study}/results")  # refreshed and stored again
    assert results["stats_version"] == backend.runner.STATS_VERSION
    assert await backend.stored_views() == []
    new = await backend.view(study)
    assert new["computed_from"] == results["computed_at"] != old["computed_from"]
    assert len(backend.computed) == 2


async def test_a_newer_statistics_version_recomputes_views_of_results_it_keeps(backend):
    study = await backend.add_study(judged=False)  # no statistics to refresh: the payload stays
    await backend.view(study)
    backend.stats_version_bump = 1
    await backend.restart()
    await backend.view(study)                      # computed again, with the new statistics
    await backend.view(study)
    assert len(backend.computed) == 2
    assert len(await backend.stored_views()) == 1


@pytest.mark.parametrize("computed_at", [2000.0, None], ids=["new-computed_at", "no-computed_at"])
async def test_a_view_is_only_served_for_the_payload_it_was_computed_from(backend, computed_at):
    study = await backend.add_study(judged=False)
    [result] = await backend.result_ids(study)
    await backend.view(study)
    # the results are replaced without the app dropping their views (e.g. by hand)
    payload = await backend.get(f"/api/studies/{study}/results")
    payload = {k: v for k, v in payload.items() if k != "computed_at"}
    if computed_at is not None:
        payload["computed_at"] = computed_at
    await backend.set_payload(result, payload)

    view = await backend.view(study)               # neither stored copy is served
    assert view["computed_from"] == computed_at and len(backend.computed) == 2
    await backend.restart()
    assert await backend.view(study) == view       # the new view took the old one's place
    assert len(backend.computed) == 2 and len(await backend.stored_views()) == 1


async def test_deleting_a_study_deletes_only_its_views(backend):
    doomed, kept = await backend.add_study(), await backend.add_study()
    await backend.view(doomed)
    await backend.view(kept)
    [kept_result] = await backend.result_ids(kept)

    assert (await backend.client.post(f"/api/studies/{doomed}/archive")).status_code == 200
    assert (await backend.client.delete(f"/api/studies/{doomed}")).status_code == 204
    assert [row[0] for row in await backend.stored_views()] == [kept_result]
    assert {key[0] for key in backend.views._MEMORY} == {kept}
    await backend.view(kept)
    assert len(backend.computed) == 2


async def test_concurrent_requests_for_one_view_compute_it_once(backend):
    study = await backend.add_study()
    backend.compute_delay = 0.5                    # the second request comes while it computes
    first, second = await asyncio.gather(backend.view(study), backend.view(study))
    assert first == second
    assert backend.computed == [("topic", "a")]
    assert len(await backend.stored_views()) == 1


async def test_two_writers_storing_one_view_keep_one_row(backend):
    study = await backend.add_study()
    [result] = await backend.result_ids(study)

    async def store(version):
        async with backend.db.SessionLocal() as session:
            assert await backend.views.store(session, result, "topic", "a", version, {"v": version})
            await session.commit()

    await asyncio.gather(store("v1"), store("v2"))  # e.g. two backend processes at once
    [row] = await backend.stored_views()
    assert row[3] in {"v1", "v2"}
    await store("v3")                               # a view of newer results replaces it
    assert [row[3] for row in await backend.stored_views()] == ["v3"]


async def test_a_view_of_results_that_are_gone_is_not_stored(backend):
    async with backend.db.SessionLocal() as session:
        assert not await backend.views.store(session, 12345, "topic", "a", "v1", {})
        await session.commit()
    assert await backend.stored_views() == []


async def test_only_views_of_values_the_filters_offer_are_stored(backend):
    study = await backend.add_study()
    await backend.view(study, "c")                 # no question has this topic
    await backend.view(study, "q1", fkey="id")     # not a filter
    assert await backend.stored_views() == []
    await backend.view(study, "b")
    assert [row[1:3] for row in await backend.stored_views()] == [("topic", "b")]


async def test_the_upsert_renders_for_postgres(backend):
    statements = []

    class PostgresSession:                          # what store() would send to Postgres
        def get_bind(self):
            return SimpleNamespace(dialect=postgresql.dialect())

        async def execute(self, statement):
            statements.append(str(statement.compile(dialect=postgresql.dialect())))
            return SimpleNamespace(first=lambda: (1,))

    assert await backend.views.store(PostgresSession(), 1, "topic", "a", "v1", {"n": 1})
    assert ("ON CONFLICT (study_result_id, fkey, fval) DO UPDATE SET version = excluded.version, "
            "payload = excluded.payload, computed_at = excluded.computed_at") in statements[-1]
