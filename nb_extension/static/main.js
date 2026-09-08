define(["base/js/namespace", "base/js/events", "jquery"], function (Jupyter, events, $) {
    "use strict";

    var PANEL_ID = "crane-llm-sidebar";

    function ensureSidebar() {
        var $panel = $("#" + PANEL_ID);
        if ($panel.length === 0) {
            $panel = $(
                '<div id="' + PANEL_ID + '" style="position:fixed;top:48px;right:0;width:420px;height:calc(100vh - 48px);' +
                'z-index:9999;background:#111827;color:#f9fafb;border-left:1px solid #374151;box-shadow:-8px 0 24px rgba(0,0,0,0.18);' +
                'padding:12px;overflow:auto;font-family:system-ui,sans-serif;">' +
                '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:10px;">' +
                '<div style="font-size:14px;font-weight:700;">CRANE-LLM</div>' +
                '<button id="crane-llm-close" style="background:#374151;color:#fff;border:0;border-radius:6px;padding:4px 8px;cursor:pointer;">Hide</button>' +
                '</div>' +
                '<div id="crane-llm-status" style="margin-bottom:10px;color:#93c5fd;">idle</div>' +
                '<div style="font-size:12px;font-weight:700;margin:8px 0 4px;">Prompt</div>' +
                '<pre id="crane-llm-prompt" style="white-space:pre-wrap;word-break:break-word;background:#0b1220;border:1px solid #243041;border-radius:8px;padding:10px;min-height:180px;margin:0 0 10px 0;"></pre>' +
                '<div style="font-size:12px;font-weight:700;margin:8px 0 4px;">Response</div>' +
                '<pre id="crane-llm-response" style="white-space:pre-wrap;word-break:break-word;background:#0b1220;border:1px solid #243041;border-radius:8px;padding:10px;min-height:120px;margin:0;"></pre>' +
                '</div>'
            );
            $("body").append($panel);
            $("#crane-llm-close").on("click", function () {
                $panel.hide();
            });
        }
        $panel.show();
        return $panel;
    }

    function setStatus(text) {
        $("#crane-llm-status").text(text);
    }

    function setPrompt(text) {
        $("#crane-llm-prompt").text(text || "");
    }

    function setResponse(text) {
        $("#crane-llm-response").text(text || "");
    }

    function appendToSelectedCell(responseText) {
        var cell = Jupyter.notebook.get_selected_cell();
        if (!cell || cell.cell_type !== "code") {
            return;
        }
        if (!cell.output_area) {
            return;
        }
        cell.output_area.clear_output();
        cell.output_area.append_output({
            output_type: "stream",
            name: "stdout",
            text: responseText + "\n"
        });
    }

    function runSelectedCellAnalysis() {
        var cell = Jupyter.notebook.get_selected_cell();
        if (!cell || cell.cell_type !== "code") {
            alert("Select a code cell first.");
            return;
        }

        var source = cell.get_text();
        var panel = ensureSidebar();
        setStatus("building prompt...");

        var pythonCode = [
            "from nb_extension.api import run_crane_llm",
            "import json",
            "source = " + JSON.stringify(source),
            "result = run_crane_llm(source=source)",
            "print(json.dumps({'prompt': result.prompt, 'response': result.response}, ensure_ascii=False))"
        ].join("\n");

        setStatus("calling LLM...");
        setPrompt("");
        setResponse("");

        var future = Jupyter.notebook.kernel.execute(pythonCode, {
            iopub: {
                output: function (msg) {
                    if (msg.msg_type === "stream") {
                        try {
                            var payload = JSON.parse(msg.content.text);
                            setPrompt(payload.prompt || "");
                            setResponse(payload.response || "");
                            setStatus("done");
                            appendToSelectedCell(payload.response || "");
                        } catch (err) {
                            setStatus("received output");
                            setResponse(msg.content.text || "");
                        }
                    } else if (msg.msg_type === "error") {
                        setStatus("error");
                        setResponse((msg.content.ename || "Error") + ": " + (msg.content.evalue || ""));
                    }
                }
            },
            shell: {
                reply: function () {
                    setStatus("running...");
                }
            }
        }, {
            silent: false,
            store_history: false,
            stop_on_error: true
        });

        return future;
    }

    function addToolbarButton() {
        if (!Jupyter || !Jupyter.toolbar) {
            return;
        }
        if (!$("#crane-llm-btn").length) {
            Jupyter.toolbar.add_buttons_group([
                {
                    label: "CRANE-LLM",
                    icon: "fa-commenting-o",
                    callback: runSelectedCellAnalysis,
                    id: "crane-llm-btn"
                }
            ]);
        }
    }

    function load_ipython_extension() {
        addToolbarButton();
        events.on("notebook_loaded.Notebook", function () {
            addToolbarButton();
        });
    }

    return {
        load_ipython_extension: load_ipython_extension,
        load: load_ipython_extension,
        runSelectedCellAnalysis: runSelectedCellAnalysis
    };
});
