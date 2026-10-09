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
        "CRANE_LLM_SCAN_LIMIT",
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
    # What the JupyterLab frontend runs.
    assert is_internal_helper_cell(
        "from crane_llm.nb_extension.api import run_crane_llm_payload\n"
        "print(run_crane_llm_payload(source='x', cell_id='c', include_runinfo=True))"
    )
    assert is_internal_helper_cell(
        "from crane_llm.nb_extension.api import load_crane_llm, set_guard\nload_crane_llm();\nset_guard(True);"
    )
    assert is_internal_helper_cell('import crane_llm\ncrane_llm.set_api_key(\n    "sk-test",\n)')
    # Ordinary user code that merely mentions the extension is notebook content.
    assert not is_internal_helper_cell("# from crane_llm.nb_extension.api import get_prompt\nmodel.fit(x)")
    # A cell that drives the extension and also does work of its own is the
    # user's code: it must reach the prompt and the provenance record.
    for source in (
        "%load_ext crane_llm\nimport pandas as pd\ndf = pd.read_csv('train.csv')",
        "import crane_llm\nX = load()",
        "from crane_llm.nb_extension.api import run_crane_llm_payload\nprint(1)",
        "%load_ext crane_llm\n%matplotlib inline",
        "%load_ext crane_llm\n!pip install lightgbm",
    ):
        assert not is_internal_helper_cell(source), source


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
        "%crane_llm guard on",
        'crane_llm.set_api_key("sk-test")',
        "%load_ext crane_llm\n%crane_llm guard on",
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
        # The cell does other work too, so it is kept and only the call,
        # which continues over several lines, is removed.
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


def check_model_responses_are_read():
    from crane_llm.nb_extension.verdict import verdict_from_response

    verdict = verdict_from_response('{"reasoning": "r", "prediction": true}', model="m")
    assert (verdict.tone, verdict.label, verdict.reasoning) == ("crash", "crash predicted", "r")
    assert verdict.source == "model" and not verdict.certain and verdict.model == "m"
    assert verdict_from_response('```json\n{"prediction": false}\n```').tone == "safe"
    assert verdict_from_response('Sure: {"detection": "true", "reasoning": "x"}').tone == "crash"
    assert verdict_from_response("no idea").tone == "unknown"

    # Blamed variables are reduced to the notebook names they start from.
    blamed = verdict_from_response(
        '{"reasoning": "r", "prediction": true, "variables": ["model.coef_", "X_test", 3]}'
    )
    assert blamed.variables == ["model", "X_test"], blamed.variables


def _run_target(code: str, namespace: dict):
    """Run a target cell for real, returning the exception it raised, if any."""

    try:
        exec(compile(code, "<cell>", "exec"), namespace)
    except Exception as exc:
        return exc
    return None


def check_builtin_checks_are_certain():
    """Every crash a check reports must really happen, with that exception.

    Each case is checked against the namespace first, then actually run.
    """

    from crane_llm.nb_extension.checks import run_checks

    def namespace():
        ns = {"nothing": None, "d": {"a": 1}, "lst": [1, 2, 3], "n": 0, "s": "abc"}
        # Notebook functions using names that are also namespace writers, as
        # attributes only; they must not stop the checks after unknown code.
        exec("def train(model):\n    model.compile()\n    model.eval()\n    setattr(model, 'x', 1)", ns)
        return ns

    cases = [
        ("undefined_thing + 1", "undefined-name"),
        ("nothing.head()", "missing-attribute"),
        ("nothing['a']", "not-subscriptable"),
        ("len(nothing)", "len-unsized"),
        ("nothing + 1", "operand-types"),
        ("d['b']", "missing-key"),
        ("lst[3]", "index-range"),
        ("1 / n", "division-by-zero"),
        ("int('3.5')", "conversion"),
        ("range(0, 10, n)", "range-step"),
        ("a, b = lst", "unpack-count"),
        ("import os\nos.nope", "missing-attribute"),
        ("print(len(lst))\nd['b']", "missing-key"),
        ("x = 1\nx.nope", "missing-attribute"),
        ("for x in nothing:\n    pass", "not-iterable"),
        ("a, b = n", "not-iterable"),
        ("[v for v in nothing]", "not-iterable"),
        ("v = 1 in 5", "unsupported-comparison"),
        ("1 < 'a'", "unsupported-comparison"),
        # The first comparison of a chain always runs.
        ("1 < 'a' < 3", "unsupported-comparison"),
        ("d[[1]]", "unhashable-key"),
        ("{[1]: 2}", "unhashable-key"),
        ("t = (1, 2)\nt[0] = 5", "immutable-assignment"),
        ("'{} {}'.format(1)", "bad-format"),
        ("f'{1.5:d}'", "bad-format"),
        ("nothing()", "not-callable"),
        ("def f(a, b):\n    return a\nf(1)", "bad-argument"),
        ("import definitely_not_installed_pkg", "missing-module"),
        ("from os import nope_name", "missing-module"),
        ("open('definitely_missing_file.txt')", "missing-file"),
        ("'%d %d' % (1,)", "bad-format"),
        ("-s", "bad-operand"),
        ("del d['b']", "missing-key"),
        ("t = (1, 2)\ndel t[0]", "immutable-assignment"),
        ("print(*n)", "not-iterable"),
        ("dict(**lst)", "not-a-mapping"),
        ("for k, v in {'abc': 1}:\n    pass", "unpack-count"),
        ("lst['a':]", "index-type"),
        ("raise 5", "explicit-raise"),
        ("raise ValueError('bad input')", "explicit-raise"),
        ("import os\nos.environ['CRANE_LLM_SURELY_UNSET']", "missing-key"),
        # Past code the walk cannot follow, every rule still applies to what
        # that code cannot change: which names exist, and immutable values.
        ("d.update(b=2)\nundefined_later.head()", "undefined-name"),
        ("d.update(b=2)\nif lst:\n    pass\nprint(f'{undefined_later}')", "undefined-name"),
        ("d.update(b=2)\nnothing.head()", "missing-attribute"),
        ("d.update(b=2)\n1 / n", "division-by-zero"),
        ("d.update(b=2)\ns + 1", "operand-types"),
    ]
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        pass
    else:
        def namespace(base=namespace):
            ns = base()
            ns.update(
                np=np,
                pd=pd,
                df=pd.DataFrame({"age": [1, 2, 3], "fare": [4.0, 5.0, 6.0]}),
                X=np.ones((5, 3)),
                X4=np.ones((2, 4)),
            )
            return ns

        cases += [
            ("df['agee']", "missing-column"),
            ("df[['age', 'fair']]", "missing-column"),
            ("df.head()\ndf.drop(columns=['nope'])", "missing-column"),
            ("df.groupby('nope')", "missing-column"),
            ("df.nope", "missing-attribute"),
            ("df['new'] = [1, 2]", "column-length"),
            ("X @ X", "matmul-shape"),
            ("X + X4", "broadcast"),
            ("np.concatenate([X, X4])", "concatenate-shape"),
            ("X.reshape(4, 4)", "reshape-size"),
            ("X[5]", "index-range"),
            ("if df:\n    pass", "ambiguous-truth"),
            # ``nothing or df`` is df, which the ``if`` then tests.
            ("if nothing or df:\n    pass", "ambiguous-truth"),
            ("df.shape()", "not-callable"),
            ("pd.Series(['n/a', '1']).astype(int)", None),
            ("df.iloc[10]", "index-range"),
            ("df.loc[7]", "missing-label"),
            ("df.loc[0, 'nope']", "missing-label"),
            ("pd.read_csv('missing_dir/nope.csv')", "missing-file"),
            ("X.sum(axis=3)", "axis-range"),
            ("X['id']", "index-type"),
            ("df['age'].str", "missing-attribute"),
            ("df['age'] + [1, 2]", "length-mismatch"),
            ("pd.concat([])", "empty-concat"),
            ("df = df.reset_index().drop(columns=['index'])\ndata1.head(25)", "undefined-name"),
        ]

        # A Series already in the namespace, so that astype can see its values.
        def namespace(base=namespace):
            ns = base()
            ns["text_ids"] = pd.Series(["n/a", "1"])
            return ns

        cases += [("text_ids.astype(int)", "astype-int")]

    try:
        import pandas as pd
        from sklearn.linear_model import LinearRegression
    except ImportError:
        pass
    else:
        # A plain estimator does check the columns, unlike a ColumnTransformer
        # (see check_builtin_checks_pass_working_code).
        def namespace(base=namespace):
            ns = base()
            train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
            ns.update(
                lr=LinearRegression().fit(train, [1.0, 2.0, 3.0]),
                swapped=train[["b", "a"]],
                one_column=train[["a"]],
            )
            return ns

        cases += [
            ("lr.predict(swapped)", "feature-names"),
            ("lr.predict(one_column)", "feature-count"),
        ]

    try:
        import torch
    except ImportError:
        pass
    else:
        def namespace(base=namespace):
            ns = base()
            ns.update(torch=torch, layer=torch.nn.Linear(4, 2), x5=torch.ones(3, 5),
                      grad=torch.ones(2, requires_grad=True), t10=torch.ones(10))
            return ns

        cases += [
            ("layer(x5)", "matmul-shape"),
            ("grad.numpy()", "tensor-conversion"),
            ("t10.view(3, 4)", "reshape-size"),
        ]

    for code, rule in cases:
        if rule is None:
            # Built inside the cell, so not a known value: nothing is claimed.
            assert run_checks(code, namespace()) is None, code
            continue
        finding = run_checks(code, namespace())
        assert finding is not None and finding.rule == rule, (code, finding)
        raised = _run_target(code, namespace())
        assert raised is not None, f"{code!r} was reported to crash but ran"
        assert type(raised).__name__ == finding.exception, (code, raised, finding.exception)


def check_builtin_checks_stop_at_unknown_code():
    """Past code whose effect is unknown, nothing is claimed.

    Each of these would crash on the namespace as it is now, but what runs
    first may change that, so the model must be asked instead.
    """

    from crane_llm.nb_extension.checks import run_checks

    def namespace():
        ns = {"d": {"a": 1}, "lst": [1, 2, 3], "helper": lambda: None, "n": 0}
        # Notebook functions that assign variables as globals.
        exec("def define_later():\n    global later\n    later = 1", ns)
        exec("def bump():\n    global n\n    n = 1", ns)
        return ns

    for code in (
        # Past unknown code: values it can change, and names it can define.
        "bump()\n1 / n",
        "d.update(b=2)\nn = 5\n1 / n",
        "d.update(b=2)\nopen('definitely_missing_file.txt')",
        "d.update(b=2)\nimport os\nos.nope",
        "d.update(b=2)\nwhile lst:\n    break\n1 / n",
        "define_later()\nlater",
        "d.update(b=2)\nexec('later = 1')\nlater",
        "d.update(b=2)\nglobals()['later'] = 1\nlater",
        "d.update(b=2)\nlater = 1\nlater",
        "d.update(b=2)\nfor i in lst:\n    pass\nlater",
        "d.update(b=2)\nx = later if lst else 0",
        "d.update(b=2)\ntry:\n    later\nexcept NameError:\n    pass",
        "try:\n    d['b']\nexcept KeyError:\n    pass",
        "for i in range(3):\n    d['b']",
        "if lst:\n    d['b']",
        "helper()\nd['b']",
        "d.update(b=2)\nd['b']",
        "lst.append(4)\nlst[3]",
        "del d\nd['b']",
        "def f():\n    return d['b']",
        "if lst:\n    d['b']",
        "d['b'] = 2\nd['b']",
        "%matplotlib inline",
    ):
        assert run_checks(code, namespace()) is None, code


def check_builtin_checks_pass_working_code():
    """Cells that run fine must not be reported to crash.

    With the guard on, such a report stops a working cell from running. Each
    case is run for real first, to confirm it works.
    """

    from crane_llm.nb_extension.checks import run_checks

    def namespace():
        return {"override": None, "limit": 3, "cfg": {}}

    cases = [
        # The last operand of ``or``/``and`` is returned, never tested.
        "value = override or 5",
        # A chain stops at its first false comparison.
        "ok = 0 < -1 < undefined_name",
        "ok = 5 < limit < cfg['max']",
    ]
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        pass
    else:
        def namespace(base=namespace):
            ns = base()
            ns.update(df=pd.DataFrame({"a": [1, 2, 3]}), arr=np.arange(5), flag=True)
            return ns

        cases += ["data = override or df", "y = flag and arr"]

    try:
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pass
    else:
        def namespace(base=namespace):
            ns = base()
            train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
            ns.update(
                ct=ColumnTransformer([("s", StandardScaler(), ["a", "b"])]).fit(train),
                extra=train.assign(other=[0.0, 0.0, 0.0]),
                reordered=train[["b", "a"]],
            )
            return ns

        # A ColumnTransformer picks its columns by name, in any order.
        cases += ["out = ct.transform(extra)", "out = ct.transform(reordered)"]

    for code in cases:
        raised = _run_target(code, namespace())
        assert raised is None, f"{code!r} is meant to run, but raised {raised!r}"
        finding = run_checks(code, namespace())
        assert finding is None, (code, finding)


def check_scan_limit_skips_data_reading_checks():
    """Above the scan limit a check that reads every value is skipped, not guessed."""

    import os

    from crane_llm.nb_extension.checks import run_checks

    try:
        import pandas as pd
    except ImportError:
        return

    def namespace():
        return {"pd": pd, "ids": pd.Series(["n/a", "1"])}

    with _IsolatedSettings() as settings:
        assert settings.scan_limit() == settings.DEFAULT_SCAN_LIMIT
        assert run_checks("ids.astype(int)", namespace()).rule == "astype-int"

        os.environ["CRANE_LLM_SCAN_LIMIT"] = "1"
        try:
            assert settings.scan_limit() == 1
            assert run_checks("ids.astype(int)", namespace()) is None
        finally:
            del os.environ["CRANE_LLM_SCAN_LIMIT"]

        # The configuration file, through the documented setter.
        import crane_llm

        crane_llm.set_scan_limit(0)
        assert settings.scan_limit() == 0
        assert run_checks("ids.astype(int)", namespace()) is None
        crane_llm.set_scan_limit(None)
        assert settings.scan_limit() == settings.DEFAULT_SCAN_LIMIT


def check_check_answers_without_calling_the_model():
    """A certain crash is reported as a check, and the model is not called."""

    extension = CraneNotebookExtension()

    def no_model(prompt, **_):
        raise AssertionError("the model must not be called when a check answers")

    extension.assistant.call_llm = no_model
    extension.set_target_cell("t", "df.head()")
    result = extension.assistant.run(shell=FakeShell({"df": None}), include_runinfo=True)

    assert result.verdict.certain and result.verdict.source == "check"
    assert result.verdict.tone == "crash" and result.verdict.label == "will crash"
    assert result.verdict.variables == ["df"]
    assert result.prompt == "" and result.response == ""

    # With runtime information switched off, the checks, which read the live
    # kernel state, must not run either.
    extension.assistant.call_llm = lambda prompt, **_: '{"prediction": false}'
    result = extension.assistant.run(shell=FakeShell({"df": None}), include_runinfo=False)
    assert result.verdict.source == "model"


def check_progress_says_the_checker_found_nothing():
    """When the checker finds nothing, the steps say so before the model is asked."""

    extension = CraneNotebookExtension()
    extension.assistant.call_llm = lambda prompt, **_: '{"prediction": false}'

    def stages(source, namespace, include_runinfo=True):
        seen = []
        extension.set_target_cell("t", source)
        result = extension.assistant.run(
            shell=FakeShell(namespace),
            include_runinfo=include_runinfo,
            progress=lambda stage, prompt, model: seen.append(stage),
        )
        return seen, result.verdict

    seen, verdict = stages("df.head()", {"df": None})
    assert seen == ["checking"], seen
    assert verdict.certain

    seen, verdict = stages("x = 1", {})
    assert seen == ["checking", "no-finding", "building", "waiting"], seen
    assert verdict.checks_ran and not verdict.certain

    seen, verdict = stages("x = 1", {}, include_runinfo=False)
    assert seen == ["building", "waiting"], seen
    assert not verdict.checks_ran


def check_llm_switched_off_runs_only_the_checker():
    """With the LLM off nothing is sent, and finding nothing is not "safe"."""

    extension = CraneNotebookExtension()

    def no_model(prompt, **_):
        raise AssertionError("the model must not be called with the LLM switched off")

    extension.assistant.call_llm = no_model

    def judge(source, namespace, include_runinfo=True):
        seen = []
        extension.set_target_cell("t", source)
        result = extension.assistant.run(
            shell=FakeShell(namespace),
            include_runinfo=include_runinfo,
            use_llm=False,
            progress=lambda stage, prompt, model: seen.append(stage),
        )
        return seen, result

    seen, result = judge("df.head()", {"df": None})
    assert result.verdict.certain and seen == ["checking"]

    seen, result = judge("x = 1", {})
    verdict = result.verdict
    assert verdict.tone == "none" and verdict.source == "check" and not verdict.certain, verdict
    assert seen == ["checking"] and result.prompt == "" and result.origins == []

    # The checker still runs with runtime information off, since nothing is
    # sent anywhere and there is no code-only comparison to protect.
    seen, result = judge("df.head()", {"df": None}, include_runinfo=False)
    assert result.verdict.certain

    # Nothing about the LLM is read: a broken configuration file, which would
    # stop an LLM call, must not stop the checker.
    with _IsolatedSettings() as settings:
        settings.config_path().write_text("{not json", encoding="utf-8")
        seen, result = judge("df.head()", {"df": None})
        assert result.verdict.certain
        seen, result = judge("x = 1", {})
        assert result.verdict.tone == "none"

    from crane_llm.nb_extension.ui import source_badge, source_note
    from crane_llm.nb_extension.texts import text

    assert source_badge(verdict) == text("verdict.badge_check_only")
    assert source_note(verdict) == text("verdict.note_check_only")


def check_model_is_chosen_per_check():
    """A model named for one check applies to that check only.

    It used to be stored on the session-wide extension, so naming a different
    one rebuilt the extension and lost the record of every cell run so far.
    """

    with _IsolatedSettings() as settings:
        settings.write_config(model="configured-model")
        extension = CraneNotebookExtension()
        extension.assistant.call_llm = lambda prompt, **_: '{"prediction": false}'
        waiting = []

        def judge(model):
            waiting.clear()
            extension.set_target_cell("t", "x = 1")
            result = extension.assistant.run(
                shell=FakeShell({}),
                include_runinfo=False,
                model=model,
                progress=lambda stage, prompt, name: waiting.append(name) if stage == "waiting" else None,
            )
            return result.verdict.model

        assert judge("gpt-5-mini") == "gpt-5-mini" and waiting == ["gpt-5-mini"]
        assert judge(None) == "configured-model" and waiting == ["configured-model"]


def check_every_text_key_exists():
    """Every text the code asks for must be in ui_texts.json, in both halves."""

    import json
    import re
    from pathlib import Path

    from crane_llm.nb_extension.texts import TEXTS_PATH

    texts = json.loads(TEXTS_PATH.read_text(encoding="utf-8"))
    here = Path(__file__).parent

    used = set()
    for path in here.glob("*.py"):
        used |= set(re.findall(r"""\btext\(\s*f?["']([\w.]+)["']""", path.read_text(encoding="utf-8")))
    source = (here / "src" / "index.ts").read_text(encoding="utf-8")
    used |= set(re.findall(r"""\bt\(\s*'([\w.]+)'""", source))
    # Keys built at runtime: one per origin role.
    used |= {f"origins.roles.{role}" for role in
             ("assigned", "modified", "possibly_modified", "deleted", "defines")}

    missing = []
    for key in sorted(used):
        node = texts
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if not isinstance(node, str):
            missing.append(key)
    assert not missing, f"missing from ui_texts.json: {missing}"
    assert len(used) > 50, len(used)


def check_cell_writes_are_recorded():
    """Assignments, in-place changes and deletions are told apart."""

    from crane_llm.nb_extension.provenance import ProvenanceLog

    log = ProvenanceLog()
    ns = {"lst": [1, 2], "d": {"a": 1}, "gone": 1}

    def run(cell_id, code, succeeded=True):
        log.before_cell(code, ns)
        error = _run_target(code, ns)
        log.after_cell(code, cell_id, execution_count=len(log.events) + 1,
                       succeeded=succeeded and error is None, namespace=ns)

    run("c1", "x = 1\nlst.append(3)")
    run("c2", "del gone")
    run("c3", "d['b'] = 2")
    run("c4", "y = 5\nraise ValueError('halfway')")

    events = {event.cell_id: event for event in log.events}
    assert events["c1"].assigned == {"x": 1}
    assert events["c1"].modified == {"lst": 2}
    assert events["c2"].deleted == {"gone": 1}
    assert "d" in events["c3"].modified
    # A cell that raised halfway still changed the state before it raised.
    assert events["c4"].assigned == {"y": 1} and not events["c4"].succeeded


def check_origins_lead_to_the_responsible_cells():
    from crane_llm.nb_extension.provenance import ProvenanceLog, locate_origins

    log = ProvenanceLog()
    ns = {}

    def run(cell_id, code):
        log.before_cell(code, ns)
        error = _run_target(code, ns)
        log.after_cell(code, cell_id, execution_count=None, succeeded=error is None, namespace=ns)

    run("c1", "data = [1, 2, 3]")
    run("c2", "data.append(4)")
    run("c3", "data = data.sort()")  # list.sort returns None
    run("c4", "other = 1")

    (origin,) = locate_origins(["data"], log, ns)
    # Only the last assignment matters; the append before it is history.
    assert [(s.cell_id, s.role) for s in origin.steps] == [("c3", "assigned")]
    assert origin.steps[0].line_text == "data = data.sort()"

    run("c5", "values = [1]")
    run("c6", "values.append(2)")
    (origin,) = locate_origins(["values"], log, ns)
    assert [(s.cell_id, s.role) for s in origin.steps] == [("c5", "assigned"), ("c6", "modified")]

    # Running the same cell again is one cell, not one entry per run, also
    # when a later run changed the variable without a visible difference.
    run("c6", "values.append(2)")
    run("c6", "values.sort()")
    (origin,) = locate_origins(["values"], log, ns)
    assert [(s.cell_id, s.role) for s in origin.steps] == [("c5", "assigned"), ("c6", "modified")]
    assert origin.steps[1].note == "3 times", origin.steps[1].note

    # A cell is named by its last run, the number the notebook shows next to
    # it, even when that run did not change the variable.
    numbered = ProvenanceLog()
    for count, (cell_id, code) in enumerate(
        [("n1", "items = [1]"), ("n2", "items.append(2)"), ("n2", "other = 1")], start=1
    ):
        numbered.before_cell(code, ns)
        _run_target(code, ns)
        numbered.after_cell(code, cell_id, execution_count=count, succeeded=True, namespace=ns)
    (origin,) = locate_origins(["items"], numbered, ns)
    assert [(s.cell_id, s.execution_count) for s in origin.steps] == [("n1", 1), ("n2", 3)]
    assert "cell [3]" in origin.summary, origin.summary

    # A name that does not exist leads to the notebook cells that would define it.
    run("c7", "result = missing_function()")
    cells = [
        {"id": "c7", "source": "result = missing_function()"},
        {"id": "c8", "source": "result = 2"},
        {"id": "t", "source": "print(result)"},
    ]
    (origin,) = locate_origins(["result"], log, ns, notebook_cells=cells, target_cell_id="t")
    notes = {s.cell_id: s.note for s in origin.steps}
    assert notes == {"c7": "raised before defining it", "c8": "has not run in this kernel session"}, notes


def check_origins_are_traced_from_a_live_kernel():
    """The hooks feed provenance, and a verdict comes back with its origins."""

    from IPython.core.interactiveshell import InteractiveShell

    from crane_llm.nb_extension import api

    shell = InteractiveShell.instance()
    api.reload_crane_llm()
    try:
        shell.run_cell("table = {'a': 1}", store_history=True, cell_id="c1")
        shell.run_cell("table = table.clear()", store_history=True, cell_id="c2")

        extension = api.get_extension()
        extension.assistant.call_llm = lambda prompt, **_: (_ for _ in ()).throw(
            AssertionError("a check should answer")
        )
        extension.set_target_cell("t", "table['a']")
        result = extension.assistant.run(shell=shell, include_runinfo=True)
    finally:
        api._dispose_instance()

    assert result.verdict.certain and result.verdict.rule == "not-subscriptable"
    (origin,) = result.origins
    assert origin.variable == "table"
    assert [(s.cell_id, s.role) for s in origin.steps] == [("c2", "assigned")]


def check_guard_stops_a_cell_that_would_crash():
    """With the guard on, a certain crash is reported and nothing in the cell runs."""

    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    from crane_llm.nb_extension import api, guard

    shell = InteractiveShell.instance()
    api.reload_crane_llm()
    try:
        api.set_guard(True)
        api.set_guard(True)
        assert len([t for t in shell.ast_transformers if isinstance(t, guard.CellGuard)]) == 1

        shell.run_cell("table = {'a': 1}", store_history=True, cell_id="g1")
        shell.run_cell("table = table.clear()", store_history=True, cell_id="g2")

        # The first line would run fine; it must not run either.
        with capture_output() as captured:
            result = shell.run_cell("guard_ran = True\ntable['a']", store_history=True, cell_id="g3")
        assert isinstance(result.error_before_exec, guard.CrashPrevented), result.error_before_exec
        assert "guard_ran" not in shell.user_ns
        assert "is not subscriptable" in str(result.error_before_exec)

        # One output: HTML for any frontend, and the verdict with its origins
        # for the JupyterLab extension to draw with links to the cells.
        (output,) = captured.outputs
        assert "text/html" in output.data
        payload = output.data[guard.MIME_TYPE]
        assert payload["verdict"]["certain"]
        (origin,) = payload["origins"]
        assert origin["variable"] == "table" and origin["steps"][0]["cell_id"] == "g2"

        # Cells the checker finds nothing in run as usual, and so does a cell
        # the user has marked to run anyway.
        result = shell.run_cell("fine = 1 + 1", store_history=True)
        assert result.success and shell.user_ns["fine"] == 2
        result = shell.run_cell("# crane: run\nguard_ran = True\ntable['a']", store_history=True)
        assert isinstance(result.error_in_exec, TypeError) and shell.user_ns["guard_ran"]

        # Executions that do not store history come from frontends, not the user.
        result = shell.run_cell("table['a']", store_history=False)
        assert isinstance(result.error_in_exec, TypeError)

        # The guard does not stop the checker from answering the toolbar button.
        extension = api.get_extension()
        extension.set_target_cell("t", "table['a']")
        assert extension.assistant.check(shell) is not None

        # Reloading the backend keeps the guard on.
        api.reload_crane_llm()
        assert guard.is_enabled(shell)

        api.set_guard(False)
        assert not guard.is_enabled(shell)
        result = shell.run_cell("table['a']", store_history=True)
        assert isinstance(result.error_in_exec, TypeError)
    finally:
        guard.disable(shell)
        api._dispose_instance()


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
        lambda prompt, **_: '{"reasoning": "fine", "prediction": false}'
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

    # A cell the guard stopped never ran, so it changed nothing.
    from crane_llm.nb_extension.guard import CrashPrevented

    stopped = FakeResult("a = 2", success=False)
    stopped.error_in_exec = None
    stopped.error_before_exec = CrashPrevented("it would crash")
    extension._mark_views_stale(stopped)
    assert "(stale)" not in handle.html, "a cell the guard stopped must not retire this one"

    extension._mark_views_stale(FakeResult("a = 2"))
    assert "(stale)" in handle.html
    assert not extension._live_views


def check_magic_shows_errors_instead_of_raising():
    from crane_llm.nb_extension import ui

    class FakeHandle:
        html = ""

        def update(self, obj):
            FakeHandle.html = obj.data

    def fail(prompt, **_):
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
    check_model_responses_are_read,
    check_builtin_checks_are_certain,
    check_builtin_checks_stop_at_unknown_code,
    check_builtin_checks_pass_working_code,
    check_scan_limit_skips_data_reading_checks,
    check_check_answers_without_calling_the_model,
    check_progress_says_the_checker_found_nothing,
    check_llm_switched_off_runs_only_the_checker,
    check_model_is_chosen_per_check,
    check_every_text_key_exists,
    check_cell_writes_are_recorded,
    check_origins_lead_to_the_responsible_cells,
    check_origins_are_traced_from_a_live_kernel,
    check_guard_stops_a_cell_that_would_crash,
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
