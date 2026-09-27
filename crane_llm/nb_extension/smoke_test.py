"""Offline checks for the notebook extension backend.

Run with ``python -m crane_llm.nb_extension.smoke_test``. No LLM call is made.

Each check corresponds to a defect that previously reached a live kernel, so
they are worth keeping even though they are quick.
"""

from __future__ import annotations

import sys

from crane_llm.nb_extension.cell_filter import is_internal_helper_cell
from crane_llm.nb_extension.extension import CraneNotebookExtension
from crane_llm.nb_extension.ipython_hooks import IPythonSessionTracker, _source_key
from crane_llm.nb_extension.prompt_builder import build_crane_prompt
from crane_llm.nb_extension.runinfo import collect_live_runinfo
from crane_llm.nb_extension.session_state import NotebookSessionState


class FakeShell:
    """Minimal stand-in for an IPython shell."""

    def __init__(self, user_ns):
        self.user_ns = user_ns


class FakeResult:
    """Minimal stand-in for an IPython ExecutionResult."""

    class _Info:
        def __init__(self, raw_cell, cell_id):
            self.raw_cell = raw_cell
            self.cell_id = cell_id

    def __init__(self, raw_cell, cell_id=None, success=True, execution_count=None):
        self.info = self._Info(raw_cell, cell_id)
        self.success = success
        self.error_before_exec = None
        self.error_in_exec = None if success else RuntimeError("boom")
        self.execution_count = execution_count


def check_prompt_shape():
    state = NotebookSessionState()
    state.record_executed_cell("cell-1", "a = 1", execution_count=1)
    state.set_target_cell("cell-2", "print(a)")

    prompt = build_crane_prompt(state, include_runinfo=False, shell=None)
    assert "# Executed Cells:" in prompt
    assert "# Current relevent runtime information:" not in prompt
    assert "# Target Cell:" in prompt
    assert "print(a)" in prompt


def check_runinfo_switch_changes_the_prompt():
    """The runtime-information switch must add or remove exactly that section."""

    state = NotebookSessionState()
    state.record_executed_cell("cell-1", "values = [1, 2, 3]", execution_count=1)
    state.set_target_cell("cell-2", "values.append(4)")
    shell = FakeShell({"values": [1, 2, 3]})

    with_runinfo = build_crane_prompt(state, include_runinfo=True, shell=shell)
    without_runinfo = build_crane_prompt(state, include_runinfo=False, shell=shell)

    assert "# Current relevent runtime information:" in with_runinfo
    assert "# Current relevent runtime information:" not in without_runinfo

    # Everything else is identical: same executed cells, same target cell.
    for prompt in (with_runinfo, without_runinfo):
        assert "values = [1, 2, 3]" in prompt
        assert prompt.rstrip().endswith("values.append(4)")


def check_runinfo_switch_selects_the_matching_system_prompt():
    """A prompt with no runtime section must not claim to have one."""

    from crane_llm.nb_extension.llm_client import default_client

    with_runinfo = default_client(model="gpt-5", include_runinfo=True).system_prompt
    without_runinfo = default_client(model="gpt-5", include_runinfo=False).system_prompt

    assert "runtime information" in with_runinfo
    assert "runtime information" not in without_runinfo
    # Both must still demand the same output contract.
    for prompt in (with_runinfo, without_runinfo):
        assert '"prediction": boolean' in prompt


class _IsolatedSettings:
    """Run a check against a throwaway configuration file.

    Without this the checks would read, and ``set_api_key`` would overwrite,
    the developer's real ``~/.crane_llm/config.json``. Stray environment
    variables are cleared for the same reason: a check that passes only on a
    machine with ``OPENAI_API_KEY`` set is not a check.
    """

    _CLEARED = (
        "CRANE_LLM_API_KEY",
        "CRANE_LLM_BASE_URL",
        "CRANE_LLM_MODEL",
        "CRANE_LLM_API_STYLE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    )

    def __enter__(self):
        import os
        import tempfile

        from crane_llm.nb_extension import settings

        self._saved = {name: os.environ.pop(name, None) for name in self._CLEARED}
        self._saved["CRANE_LLM_CONFIG"] = os.environ.get("CRANE_LLM_CONFIG")

        self._dir = tempfile.mkdtemp()
        os.environ["CRANE_LLM_CONFIG"] = os.path.join(self._dir, "config.json")
        return settings

    def __exit__(self, *exc_info):
        import os
        import shutil

        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        shutil.rmtree(self._dir, ignore_errors=True)
        return False


def check_missing_api_key_is_explained():
    """No key must produce the setup instructions, not an SDK stack trace."""

    from crane_llm.nb_extension.llm_client import default_client

    with _IsolatedSettings():
        try:
            default_client().run("hello")
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected a RuntimeError when no key is configured")

    assert "No API key found" in message
    assert "set_api_key" in message
    assert "CRANE_LLM_API_KEY" in message


def check_api_key_roundtrips_through_the_config_file():
    """``set_api_key`` is the documented setup path, so it has to be read back."""

    import crane_llm

    with _IsolatedSettings() as settings:
        assert settings.resolve().api_key is None

        path = crane_llm.set_api_key("sk-test-123")
        assert path.exists()
        assert settings.resolve().api_key == "sk-test-123"

        # An environment variable outranks the stored value.
        import os

        os.environ["CRANE_LLM_API_KEY"] = "sk-from-env"
        try:
            assert settings.resolve().api_key == "sk-from-env"
        finally:
            del os.environ["CRANE_LLM_API_KEY"]


def check_base_url_selects_the_chat_completions_client():
    """Anything but a bare OpenAI account must go through Chat Completions."""

    import crane_llm
    from crane_llm.nb_extension import llm_client

    with _IsolatedSettings() as settings:
        assert settings.resolve().api_style == "responses"
        assert isinstance(llm_client.default_client(), llm_client.OpenAILLMClient)

        crane_llm.set_api_key(
            "sk-or-test",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-sonnet-4.5",
        )

        client = llm_client.default_client()
        assert isinstance(client, llm_client.ChatCompletionsLLMClient)
        assert client.base_url == "https://openrouter.ai/api/v1"
        assert client.model == "anthropic/claude-sonnet-4.5"
        # An explicit argument still wins over the stored model.
        assert llm_client.default_client(model="gpt-5-mini").model == "gpt-5-mini"


def check_local_endpoint_needs_no_key():
    """A local inference server has no account, so a key must not be demanded."""

    import crane_llm
    from crane_llm.nb_extension import llm_client

    with _IsolatedSettings():
        crane_llm.set_api_key(
            base_url="http://localhost:11434/v1", model="qwen2.5-coder:32b"
        )
        client = llm_client.default_client()
        # Builds the SDK object; a missing key would raise here rather than on
        # the request, which is the failure this guards against.
        client._get_client()


def check_target_cell_excluded_from_executed_list():
    """A previously run cell must not be listed as executed while it is the target."""

    state = NotebookSessionState()
    state.record_executed_cell("cell-1", "a = 1", execution_count=1)
    state.record_executed_cell("cell-2", "risky(a)", execution_count=2)
    state.set_target_cell("cell-2", "risky(a)")

    prompt = build_crane_prompt(state, include_runinfo=False, shell=None)
    executed_section = prompt.split("# Target Cell:")[0]
    assert "a = 1" in executed_section
    assert "risky(a)" not in executed_section


def check_reexecution_is_deduplicated():
    state = NotebookSessionState()
    state.record_executed_cell("cell-1", "a = 1", execution_count=1)
    state.record_executed_cell("cell-1", "a = 2", execution_count=2)

    assert len(state.executed_cells) == 1
    assert state.executed_cells[0].source == "a = 2"


def check_failed_cells_are_not_recorded():
    state = NotebookSessionState()
    tracker = IPythonSessionTracker(state)

    tracker._record_result(FakeResult("a = 1", cell_id="c1", success=True))
    tracker._record_result(FakeResult("raise ValueError('x')", cell_id="c2", success=False))
    assert [cell.cell_id for cell in state.executed_cells] == ["c1"]

    # A cell that succeeded and then failed must lose its stale success record.
    tracker._record_result(FakeResult("a = boom", cell_id="c1", success=False))
    assert state.executed_cells == []


def check_history_seeded_cell_merges_with_real_execution():
    state = NotebookSessionState()
    tracker = IPythonSessionTracker(state)

    source = "a = 1"
    state.record_executed_cell(_source_key(source), source, session_sequence=1)
    tracker._record_result(FakeResult(source, cell_id="real-id", success=True))

    assert len(state.executed_cells) == 1
    assert state.executed_cells[0].cell_id == "real-id"


def check_internal_helper_cells_are_filtered():
    assert is_internal_helper_cell("")
    assert is_internal_helper_cell("from crane_llm.nb_extension.api import run_crane_llm_payload\nprint(1)")
    assert is_internal_helper_cell("from crane_llm.nb_extension.api import load_crane_llm\nload_crane_llm()")
    # Ordinary user code that merely mentions the extension is notebook content.
    assert not is_internal_helper_cell("# from crane_llm.nb_extension.api import get_prompt\nmodel.fit(x)")


def check_extension_driving_cells_are_filtered():
    """Cells that only drive CRANE-LLM must not be listed as executed code.

    A ``%%crane_llm`` cell was recorded as having run successfully, so every
    later check told the model that the previously checked cell had executed.
    """

    for source in (
        "%%crane_llm\nmodel.fit(x, y)",
        "%%crane_llm gpt-5-mini --no-runinfo\nmodel.fit(x, y)",
        "%load_ext crane_llm",
        "%reload_ext crane_llm",
        'crane_llm.set_api_key("sk-test")',
    ):
        assert is_internal_helper_cell(source), source

    state = NotebookSessionState()
    tracker = IPythonSessionTracker(state)
    tracker._record_result(FakeResult("a = 1", cell_id="c1"))
    tracker._record_result(FakeResult("%%crane_llm\nrisky(a)", cell_id="c2"))
    assert [cell.cell_id for cell in state.executed_cells] == ["c1"]


def check_api_keys_never_reach_the_prompt():
    """A key set in an executed cell must not be sent to the model."""

    state = NotebookSessionState()
    state.record_executed_cell(
        "c1",
        "import os\nos.environ['CRANE_LLM_API_KEY'] = 'sk-proj-abcdefghijklmnop1234'",
    )
    state.record_executed_cell(
        "c2",
        # Not the cell's first statement, so the cell is kept and only the
        # call, which continues over several lines, is removed.
        "x = 1\ncrane_llm.set_api_key(\n    'my-custom-key',\n    base_url='https://example.com/v1',\n)\ny = 2",
    )
    state.set_target_cell("c3", "print(x)")

    prompt = build_crane_prompt(state, include_runinfo=False, shell=None)
    assert "sk-proj-abcdefghijklmnop1234" not in prompt
    assert "my-custom-key" not in prompt
    assert "example.com" not in prompt
    # The rest of those cells is still there.
    assert "import os" in prompt
    assert "x = 1" in prompt
    assert "y = 2" in prompt
    assert "print(x)" in prompt


def check_key_set_after_first_use_is_picked_up():
    """A check run before the key was set must not stick with "no key"."""

    import os

    from crane_llm.nb_extension import assistant as assistant_module

    built = []

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def run(self, prompt):
            if not self.api_key:
                raise RuntimeError("No API key found.")
            return f"answered with {self.api_key}"

    def fake_default_client(model=None, include_runinfo=True, resolved=None):
        built.append(resolved.api_key)
        return FakeClient(resolved.api_key)

    original = assistant_module.default_client
    assistant_module.default_client = fake_default_client
    try:
        with _IsolatedSettings():
            assistant = assistant_module.CraneNotebookAssistant()
            try:
                assistant.call_llm("hello")
            except RuntimeError as exc:
                assert "No API key found" in str(exc)
            else:
                raise AssertionError("expected a RuntimeError when no key is configured")

            os.environ["CRANE_LLM_API_KEY"] = "sk-set-afterwards"
            assert assistant.call_llm("hello") == "answered with sk-set-afterwards"
            # Unchanged settings reuse the client rather than rebuilding it.
            assistant.call_llm("hello")
    finally:
        assistant_module.default_client = original

    assert built == [None, "sk-set-afterwards"], built


def check_hosted_secret_supplies_the_key():
    """On Kaggle, a shared secret is found with no setup cell at all."""

    import os
    import types

    from crane_llm.nb_extension import settings

    fake = types.ModuleType("kaggle_secrets")
    reads = []

    class UserSecretsClient:
        def get_secret(self, name):
            reads.append(name)
            if name == "CRANE_LLM_API_KEY":
                return "sk-from-kaggle-secret"
            raise KeyError(name)

    fake.UserSecretsClient = UserSecretsClient

    with _IsolatedSettings():
        saved_module = sys.modules.get("kaggle_secrets")
        sys.modules["kaggle_secrets"] = fake
        os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
        settings._HOSTED_SECRET_CACHE.clear()
        try:
            assert settings.resolve().api_key == "sk-from-kaggle-secret"
            # Read once, then cached.
            settings.resolve()
            assert reads == ["CRANE_LLM_API_KEY"], reads

            # An explicit environment variable still wins.
            os.environ["CRANE_LLM_API_KEY"] = "sk-from-env"
            assert settings.resolve().api_key == "sk-from-env"
            del os.environ["CRANE_LLM_API_KEY"]

            # With no secret shared, the message says how to share one.
            settings._HOSTED_SECRET_CACHE.clear()
            UserSecretsClient.get_secret = lambda self, name: (_ for _ in ()).throw(KeyError(name))
            assert settings.resolve().api_key is None
            assert "Add-ons > Secrets" in settings.missing_key_message()
        finally:
            os.environ.pop("KAGGLE_KERNEL_RUN_TYPE", None)
            settings._HOSTED_SECRET_CACHE.clear()
            if saved_module is None:
                sys.modules.pop("kaggle_secrets", None)
            else:
                sys.modules["kaggle_secrets"] = saved_module

    # Off Kaggle nothing is looked up and the usual message is shown.
    with _IsolatedSettings():
        assert settings.hosted_secret_api_key() is None
        assert "set_api_key" in settings.missing_key_message()


def check_requests_do_not_offer_brotli():
    """The reply must not come back brotli-encoded.

    Kaggle ships brotlipy under the name ``brotli``, and a new HTTP client then
    fails to decode brotli replies. Checked against a real local server, so it
    covers whatever header merging the SDK does.
    """

    import http.server
    import json
    import threading

    from crane_llm.nb_extension import llm_client

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["accept-encoding"] = self.headers.get("Accept-Encoding")
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"reasoning": "ok", "prediction": false}',
                            },
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = llm_client.ChatCompletionsLLMClient(
            model="m",
            system_prompt="s",
            api_key="k",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        assert '"prediction": false' in client.run("hello")
    finally:
        server.shutdown()

    offered = (seen.get("accept-encoding") or "").lower()
    assert "br" not in [part.strip() for part in offered.split(",")], offered


def check_verdict_is_read_like_the_frontend():
    from crane_llm.nb_extension.ui import read_verdict

    assert read_verdict('{"reasoning": "r", "prediction": true}') == ("crash", "crash predicted", "r")
    assert read_verdict('```json\n{"prediction": false}\n```')[0] == "safe"
    assert read_verdict('Sure: {"detection": "true", "reasoning": "x"}')[0] == "crash"
    assert read_verdict("no idea")[0] == "unknown"


def check_magic_output_goes_stale_when_a_cell_runs():
    """The magic's verdict greys out after a user cell, not after another check."""

    from crane_llm.nb_extension import ui

    class FakeHandle:
        def __init__(self):
            self.html = None

        def update(self, obj):
            self.html = obj.data

    extension = CraneNotebookExtension()
    extension.assistant.call_llm = (
        lambda prompt, include_runinfo=True: '{"reasoning": "fine", "prediction": false}'
    )

    handles = []

    def fake_show(self, status):
        self._label = status
        self._handle = FakeHandle()
        handles.append(self._handle)

    original_show = ui.VerdictView.show
    ui.VerdictView.show = fake_show
    try:
        extension.run_target_cell(source="print(1)", render=True, include_runinfo=False)
    finally:
        ui.VerdictView.show = original_show

    (handle,) = handles
    assert "no crash predicted" in handle.html
    assert "fine" in handle.html
    assert "<details" in handle.html and "print(1)" in handle.html

    extension._mark_views_stale(FakeResult("%%crane_llm\nother()"))
    assert "(stale)" not in handle.html, "another check must not retire this one"

    extension._mark_views_stale(FakeResult("a = 2"))
    assert "(stale)" in handle.html
    assert not extension._live_views


def check_magic_shows_errors_instead_of_raising():
    from crane_llm.nb_extension import ui

    class FakeHandle:
        html = ""

        def update(self, obj):
            FakeHandle.html = obj.data

    def fail(prompt, include_runinfo=True):
        raise RuntimeError("No API key found. Add-ons > Secrets")

    extension = CraneNotebookExtension()
    extension.assistant.call_llm = fail

    original_show = ui.VerdictView.show
    ui.VerdictView.show = lambda self, status: setattr(self, "_handle", FakeHandle())
    try:
        try:
            extension.run_target_cell(source="x", render=True, include_runinfo=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("run() must still raise for programmatic callers")
    finally:
        ui.VerdictView.show = original_show

    assert "could not get a prediction" in FakeHandle.html
    assert "Add-ons &gt; Secrets" in FakeHandle.html


def check_unparseable_target_cells_do_not_raise():
    """Magics, shell escapes and half-typed cells must not break prompt building."""

    for source in ("%%time\nlen(values)", "%matplotlib inline", "!pip install torch", "df.head("):
        state = NotebookSessionState()
        state.set_target_cell("cell-1", source)
        build_crane_prompt(state, include_runinfo=True, shell=FakeShell({"values": [1, 2]}))


def check_hostile_namespace_does_not_break_collection():
    """One object that raises on access must not take down the whole prompt."""

    class Exploding:
        @property
        def fit(self):
            raise RuntimeError("property exploded")

        def __len__(self):
            raise RuntimeError("len exploded")

    namespace = {"bad": Exploding(), "good": [1, 2, 3]}
    runinfo = collect_live_runinfo(shell=FakeShell(namespace), target_code="bad.fit(good)")
    assert "good" in runinfo


def check_summarisation_survives_missing_libraries():
    """Rules whose library is absent must be skipped, not fatal.

    Several rules import numpy, pandas, torch or sklearn at the top of the
    function body, so they raise for every value when that package is missing.
    Before this was handled, an environment without the ML stack produced an
    empty runtime section with no explanation.
    """

    import builtins

    from crane_llm.runinfo_parser.runtime_summary import summarize_variable

    real_import = builtins.__import__
    blocked = ("pandas", "numpy", "torch", "sklearn", "tensorflow")

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        summary = summarize_variable([1, 2, 3], {}, "values")
    finally:
        builtins.__import__ = real_import

    assert "type" in summary, summary


def check_namespace_is_not_mutated():
    namespace = {"model": [1, 2, 3]}
    before = set(namespace)
    collect_live_runinfo(shell=FakeShell(namespace), target_code="model.append(4)")
    assert set(namespace) == before, f"leaked names: {set(namespace) - before}"


CHECKS = (
    check_prompt_shape,
    check_runinfo_switch_changes_the_prompt,
    check_runinfo_switch_selects_the_matching_system_prompt,
    check_missing_api_key_is_explained,
    check_api_key_roundtrips_through_the_config_file,
    check_base_url_selects_the_chat_completions_client,
    check_local_endpoint_needs_no_key,
    check_target_cell_excluded_from_executed_list,
    check_reexecution_is_deduplicated,
    check_failed_cells_are_not_recorded,
    check_history_seeded_cell_merges_with_real_execution,
    check_internal_helper_cells_are_filtered,
    check_extension_driving_cells_are_filtered,
    check_api_keys_never_reach_the_prompt,
    check_key_set_after_first_use_is_picked_up,
    check_hosted_secret_supplies_the_key,
    check_requests_do_not_offer_brotli,
    check_verdict_is_read_like_the_frontend,
    check_magic_output_goes_stale_when_a_cell_runs,
    check_magic_shows_errors_instead_of_raising,
    check_unparseable_target_cells_do_not_raise,
    check_hostile_namespace_does_not_break_collection,
    check_summarisation_survives_missing_libraries,
    check_namespace_is_not_mutated,
)


def main():
    failures = []
    for check in CHECKS:
        try:
            check()
        except Exception as exc:
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
            print(f"FAIL {check.__name__}")
        else:
            print(f"ok   {check.__name__}")

    if failures:
        print("\n" + "\n".join(failures))
        raise SystemExit(f"{len(failures)} of {len(CHECKS)} checks failed.")

    print(f"\nAll {len(CHECKS)} CRANE-LLM notebook extension checks passed.")


if __name__ == "__main__":
    main()
