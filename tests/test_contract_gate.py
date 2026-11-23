"""The contract gate (#192): a workflow that runs the career-ops contract tests
against a real checkout, the switch that makes a skip there a failure, and the
two triggers tests.yml gained so the release workflow can call both.

The switch is the part that has to be tested by running pytest. Both contract
modules skip wherever no checkout is installed, and they do it at several
layers (a module-level `pytestmark`, class and per-test marks). Contract tests
added later may skip at others. So the hook in tests/conftest.py is driven here
through `pytester` against every layer it claims to catch, rather than checked
by reading its source.

The rest are mirrors with no other guard. The gate sets the switch by the name
conftest reads and names the modules conftest scopes it to. It runs only in the
template, as does tests.yml's push/call path, because both files ship in every
private copy, where a run is billed.
"""

from pathlib import Path

import pytest

from tests.conftest import CONTRACT_MODULES, REQUIRE_CONTRACT_ENV, contract_required
from tests.test_workflow_artifacts import DAILY, _doc, _step_index, _steps, _triggers

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parent.parent
GATE = "career-ops-contract.yml"
GATE_JOB = "contract"
TESTS = "tests.yml"
TEMPLATE_ONLY = "github.repository == 'FrameAutomata/Millhand'"


def _cond(job: dict) -> str:
    return " ".join(str(job.get("if") or "").split())


def _gate_steps() -> list[dict]:
    return _steps(GATE, GATE_JOB)


def _pytest_step() -> dict:
    steps = _gate_steps()
    return steps[_step_index(steps, lambda s: "pytest" in str(s.get("run") or ""))]


# ---------------------------------------------------------------------------
# The switch, driven through a real pytest run
# ---------------------------------------------------------------------------

# Every layer a contract module can skip at. The two contract basenames are the
# real ones, because the hook is scoped by basename. A third module checks that
# the switch reaches no further than those two.
_LAYERS = {
    "test_merge_tracker_contract": """
        import pytest
        pytestmark = pytest.mark.skipif(True, reason="no checkout")
        def test_module_mark(): pass
    """,
    "test_batch_runner_contract": """
        import pytest
        needs_newer = pytest.mark.skipif(True, reason="checkout too old")

        @needs_newer
        def test_per_test_mark(): pass

        @pytest.mark.skipif(True, reason="no bash")
        class TestClassMark:
            def test_in_class(self): pass

        @pytest.fixture
        def unavailable():
            pytest.skip("fixture skipped")

        def test_fixture_skip(unavailable): pass

        def test_body_skip():
            pytest.skip("body skipped")

        @pytest.mark.xfail(reason="known break")
        def test_xfail():
            assert False

        def test_runs(): pass
    """,
    "test_elsewhere": """
        import pytest
        @pytest.mark.skipif(True, reason="not a contract module")
        def test_other(): pass
    """,
}


@pytest.fixture
def run_inner(pytester, monkeypatch):
    """Run pytest over `files` with the switch at `value` (None: unset), with
    the real hooks from tests/conftest.py and nothing else from this suite."""
    pytester.makeconftest(
        "from tests.conftest import pytest_make_collect_report, pytest_runtest_makereport\n")

    def run(files: dict, value):
        if value is None:
            monkeypatch.delenv(REQUIRE_CONTRACT_ENV, raising=False)
        else:
            monkeypatch.setenv(REQUIRE_CONTRACT_ENV, value)
        pytester.makepyfile(**files)
        return pytester.runpytest("-ra")
    return run


def test_every_skip_layer_fails_under_the_switch(run_inner):
    result = run_inner(_LAYERS, "1")
    # Setup-phase skips (the marks and the fixture) become errors; call-phase
    # ones (a body skip, an xfail) become failures. The module outside the
    # contract keeps its skip.
    result.assert_outcomes(passed=1, failed=2, errors=4, skipped=1)
    # The skip's own reason survives, since it says what the run could not find.
    result.stdout.fnmatch_lines_random([
        f"*{REQUIRE_CONTRACT_ENV} is set*Skipped: no checkout*",
        f"*{REQUIRE_CONTRACT_ENV} is set*Skipped: checkout too old*",
        f"*{REQUIRE_CONTRACT_ENV} is set*xfail: known break*",
    ])


def test_without_the_switch_nothing_changes(run_inner):
    run_inner(_LAYERS, None).assert_outcomes(passed=1, skipped=6, xfailed=1)


@pytest.mark.parametrize("value, on", [(None, False), ("", False), ("0", False),
                                       ("false", False), ("No", False), (" off ", False),
                                       ("1", True), ("true", True), ("yes", True)])
def test_what_turns_the_switch_on(monkeypatch, value, on):
    """Anything but unset or a spelling of off, so a typo still turns it on
    while `=false` does not."""
    if value is None:
        monkeypatch.delenv(REQUIRE_CONTRACT_ENV, raising=False)
    else:
        monkeypatch.setenv(REQUIRE_CONTRACT_ENV, value)
    assert contract_required() is on


@pytest.mark.parametrize("value, outcome", [("1", {"errors": 1}), (None, {"skipped": 1})])
def test_a_module_skipped_at_import(run_inner, value, outcome):
    """`importorskip` skips the module during collection, before any test
    report exists, which is why the hook wraps the collect report as well."""
    result = run_inner({"test_merge_tracker_contract": """
        import pytest
        pytest.importorskip("millhand_no_such_module")
        def test_never_collected(): pass
    """}, value)
    result.assert_outcomes(**outcome)
    if value:
        result.stdout.fnmatch_lines([f"*{REQUIRE_CONTRACT_ENV} is set*millhand_no_such_module*"])


def test_the_contract_modules_exist():
    """The hook is scoped by basename, so a renamed module would quietly fall
    outside it and skip in the gate again."""
    missing = [m for m in CONTRACT_MODULES if not (ROOT / "tests" / m).is_file()]
    assert not missing, f"CONTRACT_MODULES names files tests/ does not have: {missing}"


# ---------------------------------------------------------------------------
# The gate workflow
# ---------------------------------------------------------------------------

class TestTheGateWorkflow:
    def test_sets_the_switch_by_name_to_a_value_conftest_reads_as_on(self, monkeypatch):
        env = _pytest_step().get("env") or {}
        assert REQUIRE_CONTRACT_ENV in env, (
            f"{GATE} runs the contract tests without {REQUIRE_CONTRACT_ENV}, so a "
            f"run that finds no checkout skips them all and passes: {sorted(env)}")
        monkeypatch.setenv(REQUIRE_CONTRACT_ENV, str(env[REQUIRE_CONTRACT_ENV]))
        assert contract_required(), env[REQUIRE_CONTRACT_ENV]

    def test_runs_exactly_the_contract_modules_against_its_checkout(self):
        step = _pytest_step()
        named = sorted(tok for tok in str(step["run"]).split() if tok.endswith(".py"))
        assert named == sorted(f"tests/{m}" for m in CONTRACT_MODULES), named
        # The directory the fetch step creates, read relative to the repo root.
        assert (step.get("env") or {}).get("CAREER_OPS_PATH") == "career-ops"

    def test_every_job_is_template_only(self):
        jobs = _doc(GATE)["jobs"]
        wrong = {name: _cond(job) for name, job in jobs.items()
                 if _cond(job) != TEMPLATE_ONLY}
        assert jobs and not wrong, (
            f"{GATE} ships in every private copy, where its weekly run would be "
            f"billed; every job needs `if: {TEMPLATE_ONLY}`: {wrong}")

    def test_triggers_and_targets(self):
        """A dispatch and a call default to the pin, which is what copies run,
        and an input always wins; a PR or push tests the pin; only the weekly
        schedule, which has no inputs, tests the branch."""
        trig = _triggers(_doc(GATE))
        assert trig.get("schedule"), trig
        for event in ("workflow_dispatch", "workflow_call"):
            ref = trig[event]["inputs"]["career_ops_ref"]
            assert ref["default"] == "pin", (event, ref)
        assert trig["workflow_dispatch"]["inputs"]["career_ops_ref"]["options"] == ["pin", "branch"]
        target = " ".join(str(_doc(GATE)["jobs"][GATE_JOB]["env"]["CONTRACT_REF"]).split())
        # An input decides when there is one; only a schedule tests the branch.
        # An EMPTY input is no input (it is falsy here), so the event decides —
        # and a called workflow sees its caller's event, which is why a caller
        # passes `pin` or `branch` literally rather than forwarding an optional.
        assert target == ("${{ inputs.career_ops_ref || "
                          "(github.event_name == 'schedule' && 'branch' || 'pin') }}"), target
        # A PR and a push to main are when the contract changes, so they run it.
        assert "pull_request" in trig and trig["push"]["branches"] == ["main"], trig

    @pytest.mark.parametrize("action, key", [("actions/setup-python", "python-version"),
                                             ("actions/setup-node", "node-version")])
    def test_runs_the_daily_toolchain(self, action, key):
        """merge-tracker.mjs runs under the daily's Node, so that is the Node
        the contract has to hold under."""
        def version(steps):
            step = steps[_step_index(steps, lambda s: str(s.get("uses") or "").startswith(action))]
            return str(step["with"][key])
        assert version(_gate_steps()) == version(_steps(DAILY, "scrape-and-evaluate"))


def test_tests_yml_pushes_and_calls_only_in_the_template():
    """tests.yml ships in every copy too. A pull request runs it anywhere, as it
    always has; a push to main or a call runs it only in the template, since a
    copy's daily is to commit to that copy's main on every run (#211)."""
    doc = _doc(TESTS)
    trig = _triggers(doc)
    assert "pull_request" in trig and "workflow_call" in trig, trig
    assert trig["push"]["branches"] == ["main"], trig["push"]
    want = f"github.event_name == 'pull_request' || {TEMPLATE_ONLY}"
    wrong = {name: _cond(job) for name, job in doc["jobs"].items() if _cond(job) != want}
    assert not wrong, f"{TESTS} jobs need `if: {want}`: {wrong}"
