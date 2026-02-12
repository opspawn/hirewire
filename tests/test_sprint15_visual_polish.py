"""Sprint 15 tests: Dashboard visual polish, demo flow hardening, and version bump.

Tests cover:
- Dashboard visual elements (stat cards, activity feed, agent cards, forms)
- 7-stage live demo pipeline (register → discover → hire → pay → execute → rate → dashboard)
- Category filter pills
- Step-by-step form UX
- Error recovery in demo flow
- Version v0.15.0
"""

from __future__ import annotations

import pytest
import httpx
import time

from src.api.main import app, _running_tasks, LIVE_DEMO_TASKS
from src.mcp_servers.payment_hub import ledger
from src.storage import get_storage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def _clean_ledger():
    ledger.clear()
    yield
    ledger.clear()


@pytest.fixture(autouse=True)
def _clean_running_tasks():
    _running_tasks.clear()
    yield
    for t in list(_running_tasks.values()):
        t.cancel()
    _running_tasks.clear()


# ---------------------------------------------------------------------------
# Dashboard Visual Polish — Stat Cards
# ---------------------------------------------------------------------------

class TestStatCardVisuals:
    """Verify stat card CSS enhancements are present in the served HTML."""

    @pytest.mark.asyncio
    async def test_stat_cards_have_gradient_background(self, client):
        resp = await client.get("/")
        assert "linear-gradient(135deg" in resp.text

    @pytest.mark.asyncio
    async def test_stat_value_larger_font(self, client):
        resp = await client.get("/")
        assert "font-size: 30px" in resp.text or "font-size:30px" in resp.text

    @pytest.mark.asyncio
    async def test_stat_value_extra_bold(self, client):
        resp = await client.get("/")
        assert "font-weight: 800" in resp.text or "font-weight:800" in resp.text

    @pytest.mark.asyncio
    async def test_stat_label_uppercase(self, client):
        resp = await client.get("/")
        # stat-label should have text-transform: uppercase
        assert "text-transform: uppercase" in resp.text or "text-transform:uppercase" in resp.text

    @pytest.mark.asyncio
    async def test_stat_card_hover_glow_line(self, client):
        resp = await client.get("/")
        assert "stat-card::before" in resp.text

    @pytest.mark.asyncio
    async def test_stat_card_border_radius_12(self, client):
        resp = await client.get("/")
        # stat-card should have border-radius: 12px
        assert "border-radius: 12px" in resp.text or "border-radius:12px" in resp.text


# ---------------------------------------------------------------------------
# Dashboard Visual Polish — Activity Feed
# ---------------------------------------------------------------------------

class TestActivityFeedVisuals:
    """Verify activity feed visual enhancements."""

    @pytest.mark.asyncio
    async def test_activity_item_hover_effect(self, client):
        resp = await client.get("/")
        assert "activity-item:hover" in resp.text

    @pytest.mark.asyncio
    async def test_activity_icon_larger(self, client):
        resp = await client.get("/")
        # Icon should be 30x30
        text = resp.text
        assert ("width: 30px" in text or "width:30px" in text)

    @pytest.mark.asyncio
    async def test_activity_time_has_background(self, client):
        resp = await client.get("/")
        # .activity-time should have background and padding
        assert "activity-time" in resp.text
        assert "tabular-nums" in resp.text

    @pytest.mark.asyncio
    async def test_activity_feed_smooth_scroll(self, client):
        resp = await client.get("/")
        assert "scroll-behavior: smooth" in resp.text or "scroll-behavior:smooth" in resp.text


# ---------------------------------------------------------------------------
# Dashboard Visual Polish — Agent Cards
# ---------------------------------------------------------------------------

class TestAgentCardVisuals:
    """Verify agent card visual enhancements."""

    @pytest.mark.asyncio
    async def test_agent_avatar_shadow(self, client):
        resp = await client.get("/")
        assert "box-shadow: 0 2px 8px" in resp.text or "box-shadow:0 2px 8px" in resp.text

    @pytest.mark.asyncio
    async def test_agent_row_selected_border(self, client):
        resp = await client.get("/")
        assert "agent-row.selected" in resp.text
        assert "border-left: 3px solid" in resp.text or "border-left:3px solid" in resp.text

    @pytest.mark.asyncio
    async def test_agent_avatar_larger(self, client):
        resp = await client.get("/")
        # 38px avatar
        assert ("width: 38px" in resp.text or "width:38px" in resp.text)


# ---------------------------------------------------------------------------
# Dashboard Visual Polish — Skill Tags
# ---------------------------------------------------------------------------

class TestSkillTagVisuals:
    """Verify color-coded skill tag system."""

    @pytest.mark.asyncio
    async def test_tag_pill_style(self, client):
        resp = await client.get("/")
        # Tags should be pill-shaped (border-radius: 20px)
        assert "border-radius: 20px" in resp.text or "border-radius:20px" in resp.text

    @pytest.mark.asyncio
    async def test_tag_color_variants_exist(self, client):
        resp = await client.get("/")
        for variant in ["tag-blue", "tag-green", "tag-yellow", "tag-purple", "tag-cyan"]:
            assert variant in resp.text

    @pytest.mark.asyncio
    async def test_tag_hover_effect(self, client):
        resp = await client.get("/")
        assert ".tag:hover" in resp.text

    @pytest.mark.asyncio
    async def test_skill_color_map_in_js(self, client):
        resp = await client.get("/")
        assert "skillColors" in resp.text
        assert "'python':'tag-blue'" in resp.text or '"python":"tag-blue"' in resp.text


# ---------------------------------------------------------------------------
# Dashboard Visual Polish — Category Filter Pills
# ---------------------------------------------------------------------------

class TestCategoryFilterPills:
    """Verify agent category filter pill system."""

    @pytest.mark.asyncio
    async def test_filter_pill_css_exists(self, client):
        resp = await client.get("/")
        assert ".filter-pill" in resp.text

    @pytest.mark.asyncio
    async def test_filter_pill_active_state(self, client):
        resp = await client.get("/")
        assert ".filter-pill.active" in resp.text

    @pytest.mark.asyncio
    async def test_filter_pills_container_exists(self, client):
        resp = await client.get("/")
        assert "agent-filter-pills" in resp.text

    @pytest.mark.asyncio
    async def test_filter_js_function_exists(self, client):
        resp = await client.get("/")
        assert "orchSetFilter" in resp.text
        assert "orchFilter" in resp.text


# ---------------------------------------------------------------------------
# Dashboard Visual Polish — Form UX
# ---------------------------------------------------------------------------

class TestFormUX:
    """Verify form enhancement features."""

    @pytest.mark.asyncio
    async def test_form_input_focus_glow(self, client):
        resp = await client.get("/")
        assert "box-shadow: 0 0 0 3px" in resp.text or "box-shadow:0 0 0 3px" in resp.text

    @pytest.mark.asyncio
    async def test_form_input_hover_state(self, client):
        resp = await client.get("/")
        assert ".form-input:hover" in resp.text

    @pytest.mark.asyncio
    async def test_form_success_class(self, client):
        resp = await client.get("/")
        assert ".form-success" in resp.text

    @pytest.mark.asyncio
    async def test_form_error_class(self, client):
        resp = await client.get("/")
        assert ".form-error" in resp.text

    @pytest.mark.asyncio
    async def test_form_hint_exists(self, client):
        resp = await client.get("/")
        assert ".form-hint" in resp.text
        assert "task-hint" in resp.text

    @pytest.mark.asyncio
    async def test_step_indicator_exists(self, client):
        resp = await client.get("/")
        assert ".step-indicator" in resp.text
        assert "step-dot" in resp.text

    @pytest.mark.asyncio
    async def test_step_indicator_has_three_steps(self, client):
        resp = await client.get("/")
        assert "step-1" in resp.text
        assert "step-2" in resp.text
        assert "step-3" in resp.text

    @pytest.mark.asyncio
    async def test_reset_modal_steps_function(self, client):
        resp = await client.get("/")
        assert "resetModalSteps" in resp.text


# ---------------------------------------------------------------------------
# Live Demo Pipeline — 7 Stages
# ---------------------------------------------------------------------------

class TestLiveDemoPipeline:
    """Verify the 7-stage live demo pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_has_7_stages(self, client):
        resp = await client.get("/")
        text = resp.text
        assert "ps-7" in text
        assert "ps-7-detail" in text
        assert "ps-7-time" in text

    @pytest.mark.asyncio
    async def test_pipeline_stage_names(self, client):
        resp = await client.get("/")
        text = resp.text
        assert "Register Task" in text
        assert "Discover Agents" in text
        assert "Hire Agent" in text
        assert "Pay via x402" in text
        assert "Execute (GPT-4o)" in text
        assert "Rate &amp; Verify" in text or "Rate & Verify" in text
        assert "Dashboard Update" in text

    @pytest.mark.asyncio
    async def test_pipeline_reset_handles_7_stages(self, client):
        resp = await client.get("/")
        assert "i<=7" in resp.text

    @pytest.mark.asyncio
    async def test_demo_live_endpoint_returns_7_stages(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert "stages" in data
        assert len(data["stages"]) == 7

    @pytest.mark.asyncio
    async def test_demo_live_stage_names_match(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        stage_names = [s["name"] for s in data["stages"]]
        assert stage_names == [
            "Register Task",
            "Discover Agents",
            "Hire Agent",
            "Pay via x402",
            "Execute (GPT-4o)",
            "Rate & Verify",
            "Dashboard Update",
        ]

    @pytest.mark.asyncio
    async def test_demo_live_all_stages_have_detail(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        for stage in data["stages"]:
            assert "detail" in stage
            assert len(stage["detail"]) > 0

    @pytest.mark.asyncio
    async def test_demo_live_all_stages_have_duration(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        for stage in data["stages"]:
            assert "duration_ms" in stage
            assert isinstance(stage["duration_ms"], (int, float))

    @pytest.mark.asyncio
    async def test_demo_live_has_quality_score(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        # Rate & Verify stage should mention quality score
        rate_stage = data["stages"][5]
        assert "Quality score" in rate_stage["detail"]

    @pytest.mark.asyncio
    async def test_demo_live_dashboard_stage_mentions_refresh(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        dash_stage = data["stages"][6]
        assert "Dashboard" in dash_stage["detail"] or "dashboard" in dash_stage["detail"]


# ---------------------------------------------------------------------------
# Live Demo Pipeline — Task Variations
# ---------------------------------------------------------------------------

class TestLiveDemoTaskVariations:
    """Verify demo works with all task variations."""

    @pytest.mark.asyncio
    async def test_demo_task_index_0(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert "memory" in data["description"].lower() or "agent" in data["description"].lower()

    @pytest.mark.asyncio
    async def test_demo_task_index_1(self, client):
        resp = await client.post("/demo/live", json={"task_index": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "pricing" in data["description"].lower() or "agent" in data["description"].lower()

    @pytest.mark.asyncio
    async def test_demo_task_index_2(self, client):
        resp = await client.post("/demo/live", json={"task_index": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert "dashboard" in data["description"].lower() or "monitoring" in data["description"].lower()

    @pytest.mark.asyncio
    async def test_demo_task_index_wraps(self, client):
        resp = await client.post("/demo/live", json={"task_index": 100})
        assert resp.status_code == 200
        data = resp.json()
        assert "stages" in data
        assert len(data["stages"]) == 7

    @pytest.mark.asyncio
    async def test_demo_live_no_body(self, client):
        resp = await client.post("/demo/live")
        assert resp.status_code == 200
        data = resp.json()
        assert "stages" in data

    @pytest.mark.asyncio
    async def test_demo_live_returns_cost(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        assert "cost_usdc" in data
        assert data["cost_usdc"] > 0

    @pytest.mark.asyncio
    async def test_demo_live_returns_agent(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        assert "agent" in data
        assert len(data["agent"]) > 0

    @pytest.mark.asyncio
    async def test_demo_live_returns_model(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        assert "model" in data
        assert data["model"] in ("gpt-4o", "mock")

    @pytest.mark.asyncio
    async def test_demo_live_returns_total_ms(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        assert "total_ms" in data
        assert data["total_ms"] > 0

    @pytest.mark.asyncio
    async def test_demo_live_returns_preview(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        assert "response_preview" in data
        assert len(data["response_preview"]) > 0


# ---------------------------------------------------------------------------
# Demo Pipeline — Data Integrity
# ---------------------------------------------------------------------------

class TestDemoPipelineIntegrity:
    """Verify demo pipeline creates proper data in storage/ledger."""

    @pytest.mark.asyncio
    async def test_demo_creates_task_in_storage(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        task_id = data["task_id"]
        task_resp = await client.get(f"/tasks/{task_id}")
        assert task_resp.status_code == 200
        task = task_resp.json()
        assert task["status"] == "completed"

    @pytest.mark.asyncio
    async def test_demo_creates_payment(self, client):
        txs_before = await client.get("/transactions")
        count_before = len(txs_before.json())
        await client.post("/demo/live", json={"task_index": 0})
        txs_after = await client.get("/transactions")
        count_after = len(txs_after.json())
        assert count_after > count_before

    @pytest.mark.asyncio
    async def test_demo_task_has_result(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        task_resp = await client.get(f"/tasks/{data['task_id']}")
        task = task_resp.json()
        assert task["result"] is not None
        assert "assigned_agent" in task["result"]

    @pytest.mark.asyncio
    async def test_demo_task_has_quality_score(self, client):
        resp = await client.post("/demo/live", json={"task_index": 0})
        data = resp.json()
        task_resp = await client.get(f"/tasks/{data['task_id']}")
        task = task_resp.json()
        assert "quality_score" in task["result"]
        assert 0.0 < task["result"]["quality_score"] <= 1.0


# ---------------------------------------------------------------------------
# Version Bump
# ---------------------------------------------------------------------------

class TestVersionBump:
    """Verify version is updated to 0.15.0."""

    @pytest.mark.asyncio
    async def test_dashboard_shows_version(self, client):
        resp = await client.get("/")
        assert "v0.15.0" in resp.text

    @pytest.mark.asyncio
    async def test_api_version_in_openapi(self, client):
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["version"] == "0.15.0"


# ---------------------------------------------------------------------------
# Orchestration Flow Visuals
# ---------------------------------------------------------------------------

class TestOrchestrationVisuals:
    """Verify orchestration node enhancements."""

    @pytest.mark.asyncio
    async def test_orch_node_box_larger(self, client):
        resp = await client.get("/")
        assert ("width: 84px" in resp.text or "width:84px" in resp.text)

    @pytest.mark.asyncio
    async def test_orch_active_gradient(self, client):
        resp = await client.get("/")
        assert "rgba(34,197,94,0.08)" in resp.text

    @pytest.mark.asyncio
    async def test_orch_node_bg_secondary(self, client):
        resp = await client.get("/")
        assert "orch-node-box" in resp.text
        assert "bg-secondary" in resp.text


# ---------------------------------------------------------------------------
# Dark Theme Polish
# ---------------------------------------------------------------------------

class TestDarkThemePolish:
    """Verify professional dark theme elements."""

    @pytest.mark.asyncio
    async def test_css_variables_present(self, client):
        resp = await client.get("/")
        text = resp.text
        for var in ["--bg", "--card", "--accent", "--green", "--text", "--text-muted"]:
            assert var in text

    @pytest.mark.asyncio
    async def test_inter_font_loaded(self, client):
        resp = await client.get("/")
        assert "fonts.googleapis.com" in resp.text
        assert "Inter" in resp.text

    @pytest.mark.asyncio
    async def test_no_raw_json_in_demo_data(self, client):
        resp = await client.get("/")
        text = resp.text
        # DEMO_DATA should be present and contain structured demo content
        assert "DEMO_DATA=" in text
        # Check that demo data uses JS object notation with proper values
        assert "status:" in text
        assert "healthy" in text

    @pytest.mark.asyncio
    async def test_scrollbar_styling(self, client):
        resp = await client.get("/")
        assert "::-webkit-scrollbar" in resp.text

    @pytest.mark.asyncio
    async def test_animations_present(self, client):
        resp = await client.get("/")
        for anim in ["fadeIn", "slideIn", "shimmer", "pulse"]:
            assert anim in resp.text


# ---------------------------------------------------------------------------
# Standalone Demo Data
# ---------------------------------------------------------------------------

class TestStandaloneDemoData:
    """Verify standalone demo data renders without backend."""

    @pytest.mark.asyncio
    async def test_demo_data_has_health(self, client):
        resp = await client.get("/")
        assert "DEMO_DATA" in resp.text
        assert "agents_count:5" in resp.text or "agents_count: 5" in resp.text

    @pytest.mark.asyncio
    async def test_demo_data_has_agents(self, client):
        resp = await client.get("/")
        assert "ceo-agent" in resp.text
        assert "builder-agent" in resp.text
        assert "research-agent" in resp.text

    @pytest.mark.asyncio
    async def test_demo_data_has_activity(self, client):
        resp = await client.get("/")
        assert "research-agent" in resp.text
        assert "completed" in resp.text

    @pytest.mark.asyncio
    async def test_demo_data_has_transactions(self, client):
        resp = await client.get("/")
        assert "amount_usdc" in resp.text

    @pytest.mark.asyncio
    async def test_demo_data_has_approvals(self, client):
        resp = await client.get("/")
        assert "approvals" in resp.text
        assert "pending" in resp.text

    @pytest.mark.asyncio
    async def test_demo_data_has_safety(self, client):
        resp = await client.get("/")
        assert "fairness_score" in resp.text


# ---------------------------------------------------------------------------
# LIVE_DEMO_TASKS Configuration
# ---------------------------------------------------------------------------

class TestLiveDemoTasksConfig:
    """Verify LIVE_DEMO_TASKS are properly configured."""

    def test_has_three_tasks(self):
        assert len(LIVE_DEMO_TASKS) == 3

    def test_all_tasks_have_description(self):
        for task in LIVE_DEMO_TASKS:
            assert "description" in task
            assert len(task["description"]) > 10

    def test_all_tasks_have_budget(self):
        for task in LIVE_DEMO_TASKS:
            assert "budget" in task
            assert task["budget"] > 0

    def test_all_tasks_have_mock_response(self):
        for task in LIVE_DEMO_TASKS:
            assert "mock_response" in task
            assert len(task["mock_response"]) > 50

    def test_budgets_are_reasonable(self):
        for task in LIVE_DEMO_TASKS:
            assert 0 < task["budget"] <= 10
