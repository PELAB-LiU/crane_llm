import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, showErrorMessage, ToolbarButton } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';

const COMMAND_ID = 'crane-llm-jlab:run';
const TOGGLE_RUNINFO_COMMAND_ID = 'crane-llm-jlab:toggle-runinfo';
const SIDEBAR_ID = 'crane-llm-sidebar';

/** Must match api.PAYLOAD_BEGIN / api.PAYLOAD_END. */
const PAYLOAD_BEGIN = '<<<CRANE-LLM:BEGIN>>>';
const PAYLOAD_END = '<<<CRANE-LLM:END>>>';

/** Kernel calls are cheap, but a busy kernel queues them behind user code. */
const KERNEL_TIMEOUT_MS = 10 * 60 * 1000;

const INSTALLED_PANELS = new WeakSet<NotebookPanel>();
const RESPONSE_MANAGERS = new WeakMap<NotebookPanel, CraneResponseManager>();
const RUNNING_PANELS = new WeakSet<NotebookPanel>();

type PredictionTone = 'crash' | 'safe' | 'unknown' | 'stale';

interface Verdict {
  tone: PredictionTone;
  label: string;
}

interface ResponseEntry {
  container: HTMLDivElement;
  titleNode: HTMLDivElement;
  responseNode: HTMLPreElement;
  /** The cell widget's node, so the accent can be cleared when the entry goes. */
  cellNode: HTMLElement;
  stale: boolean;
}

interface CranePayload {
  ok: boolean;
  prompt: string;
  response: string;
  error: string;
}

/** Remembers the runtime-information switch across page reloads. */
const RUNINFO_STORAGE_KEY = 'crane-llm:include-runinfo';

function loadIncludeRuninfo(): boolean {
  // Default on. Storage can be unavailable in private windows, so never let a
  // failure here stop the extension from loading.
  try {
    return window.localStorage.getItem(RUNINFO_STORAGE_KEY) !== 'false';
  } catch (_error) {
    return true;
  }
}

function saveIncludeRuninfo(value: boolean): void {
  try {
    window.localStorage.setItem(RUNINFO_STORAGE_KEY, String(value));
  } catch (_error) {
    // Not worth surfacing; the switch simply will not persist.
  }
}

/**
 * The runtime-information switch, shared by every control that shows it.
 *
 * The sidebar checkbox, the toolbar popover and the command palette entry are
 * three views of this one value. Without a single owner they would drift apart
 * as soon as the user changed it from one of them.
 */
class RuninfoSetting {
  private value: boolean = loadIncludeRuninfo();
  private listeners = new Set<(value: boolean) => void>();

  get(): boolean {
    return this.value;
  }

  set(value: boolean): void {
    if (value === this.value) {
      return;
    }
    this.value = value;
    saveIncludeRuninfo(value);
    for (const listener of this.listeners) {
      listener(value);
    }
  }

  toggle(): void {
    this.set(!this.value);
  }

  /** Subscribe and receive the current value immediately. */
  subscribe(listener: (value: boolean) => void): () => void {
    this.listeners.add(listener);
    listener(this.value);
    return () => this.listeners.delete(listener);
  }
}

function runinfoHintText(includeRuninfo: boolean): string {
  return includeRuninfo
    ? 'Prompt: executed cells, runtime state, target cell.'
    : 'Prompt: executed cells and target cell only.';
}

class CraneSidebar extends Widget {
  private statusNode: HTMLDivElement;
  private promptNode: HTMLPreElement;
  private responseNode: HTMLPreElement;
  private runinfoCheckbox: HTMLInputElement;
  private runinfoHintNode: HTMLDivElement;

  constructor(private setting: RuninfoSetting) {
    super();
    this.addClass('crane-llm-sidebar');

    this.node.style.cssText = [
      'padding:12px',
      'height:100%',
      'overflow:auto',
      'box-sizing:border-box',
      'background:var(--jp-layout-color1)',
      'color:var(--jp-ui-font-color1)',
      'border-left:1px solid var(--jp-border-color2)'
    ].join(';');

    const header = document.createElement('div');
    header.style.cssText =
      'display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px;';

    const title = document.createElement('div');
    title.textContent = 'CRANE-LLM';
    title.style.cssText = 'font-size:14px;font-weight:700;';

    header.appendChild(title);

    this.statusNode = document.createElement('div');
    this.statusNode.style.cssText =
      'margin-bottom:10px;color:var(--jp-info-color1);font-weight:600;';
    this.statusNode.textContent = 'idle';

    // Runtime-information switch. On by default, which is the configuration
    // the approach is built around; turning it off falls back to a prompt of
    // executed cells and the target cell alone.
    const runinfoRow = document.createElement('label');
    runinfoRow.style.cssText =
      'display:flex;align-items:center;gap:8px;margin:0 0 4px;cursor:pointer;font-size:12px;';

    this.runinfoCheckbox = document.createElement('input');
    this.runinfoCheckbox.type = 'checkbox';
    this.runinfoCheckbox.style.cssText = 'margin:0;cursor:pointer;';

    const runinfoLabel = document.createElement('span');
    runinfoLabel.textContent = 'Include runtime information';
    runinfoLabel.style.cssText = 'font-weight:700;';

    runinfoRow.appendChild(this.runinfoCheckbox);
    runinfoRow.appendChild(runinfoLabel);

    this.runinfoHintNode = document.createElement('div');
    this.runinfoHintNode.style.cssText =
      'font-size:11px;opacity:0.75;margin:0 0 10px 24px;';

    this.runinfoCheckbox.onchange = () => this.setting.set(this.runinfoCheckbox.checked);

    // Reflect changes made from the toolbar popover or the command palette.
    this.setting.subscribe(value => {
      this.runinfoCheckbox.checked = value;
      this.runinfoHintNode.textContent = runinfoHintText(value);
    });

    const promptLabel = document.createElement('div');
    promptLabel.textContent = 'Prompt';
    promptLabel.style.cssText = 'font-size:12px;font-weight:700;margin:8px 0 4px;';

    this.promptNode = document.createElement('pre');
    this.promptNode.style.cssText = [
      'white-space:pre-wrap',
      'word-break:break-word',
      'background:var(--jp-layout-color2)',
      'border:1px solid var(--jp-border-color2)',
      'border-radius:8px',
      'padding:10px',
      'min-height:180px',
      'margin:0 0 10px 0'
    ].join(';');

    const responseLabel = document.createElement('div');
    responseLabel.textContent = 'Response';
    responseLabel.style.cssText = 'font-size:12px;font-weight:700;margin:8px 0 4px;';

    this.responseNode = document.createElement('pre');
    this.responseNode.style.cssText = [
      'white-space:pre-wrap',
      'word-break:break-word',
      'background:var(--jp-layout-color2)',
      'border:1px solid var(--jp-border-color2)',
      'border-radius:8px',
      'padding:10px',
      'min-height:120px',
      'margin:0'
    ].join(';');

    this.node.appendChild(header);
    this.node.appendChild(this.statusNode);
    this.node.appendChild(runinfoRow);
    this.node.appendChild(this.runinfoHintNode);
    this.node.appendChild(promptLabel);
    this.node.appendChild(this.promptNode);
    this.node.appendChild(responseLabel);
    this.node.appendChild(this.responseNode);
  }

  setStatus(text: string): void {
    this.statusNode.textContent = text;
  }

  setPrompt(text: string): void {
    this.promptNode.textContent = text;
  }

  setResponse(text: string): void {
    this.responseNode.textContent = text;
  }
}

/**
 * Hang the runtime-information switch off the toolbar button, revealed on hover.
 *
 * The sidebar has the same switch, but it is easy to never open the sidebar at
 * all, and a setting nobody can find is not really exposed. The popover is
 * appended to document.body and positioned with fixed coordinates so that no
 * ancestor's overflow can clip it out of the toolbar.
 *
 * Returns a teardown function; the caller runs it when the panel goes away.
 */
function installRuninfoPopover(anchor: HTMLElement, setting: RuninfoSetting): () => void {
  const popover = document.createElement('div');
  popover.className = 'crane-llm-runinfo-popover';
  popover.style.cssText = [
    'position:fixed',
    'z-index:10000',
    'display:none',
    'min-width:250px',
    'padding:10px 12px',
    'border-radius:8px',
    'background:var(--jp-layout-color1)',
    'color:var(--jp-ui-font-color1)',
    'border:1px solid var(--jp-border-color1)',
    'box-shadow:0 4px 14px rgba(0,0,0,0.22)',
    'font-size:12px'
  ].join(';');

  const row = document.createElement('label');
  row.style.cssText =
    'display:flex;align-items:center;gap:8px;cursor:pointer;font-weight:700;';

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.style.cssText = 'margin:0;cursor:pointer;';

  const labelText = document.createElement('span');
  labelText.textContent = 'Include runtime information';

  const hint = document.createElement('div');
  hint.style.cssText = 'margin-top:6px;opacity:0.75;';

  row.appendChild(checkbox);
  row.appendChild(labelText);
  popover.appendChild(row);
  popover.appendChild(hint);
  document.body.appendChild(popover);

  const unsubscribe = setting.subscribe(value => {
    checkbox.checked = value;
    hint.textContent = runinfoHintText(value);
  });

  checkbox.onchange = () => setting.set(checkbox.checked);

  // Clicking anywhere in the popover must not reach the button behind it.
  popover.addEventListener('click', event => event.stopPropagation());

  let hideTimer: number | undefined;

  const cancelHide = () => {
    if (hideTimer !== undefined) {
      window.clearTimeout(hideTimer);
      hideTimer = undefined;
    }
  };

  const show = () => {
    cancelHide();
    const rect = anchor.getBoundingClientRect();
    popover.style.display = 'block';
    // Measure after it is displayed, then keep it inside the viewport.
    const width = popover.offsetWidth;
    const left = Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8));
    popover.style.top = `${rect.bottom + 6}px`;
    popover.style.left = `${left}px`;
  };

  // A grace period, so moving the pointer from the button into the popover
  // does not close it on the way.
  const scheduleHide = () => {
    cancelHide();
    hideTimer = window.setTimeout(() => {
      popover.style.display = 'none';
    }, 250);
  };

  const hideNow = () => {
    cancelHide();
    popover.style.display = 'none';
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      hideNow();
    }
  };

  anchor.addEventListener('mouseenter', show);
  anchor.addEventListener('mouseleave', scheduleHide);
  // Keyboard users never fire mouseenter, so tabbing to the button opens it too.
  anchor.addEventListener('focusin', show);
  popover.addEventListener('mouseenter', cancelHide);
  popover.addEventListener('mouseleave', scheduleHide);
  popover.addEventListener('focusin', cancelHide);
  popover.addEventListener('focusout', scheduleHide);
  document.addEventListener('keydown', onKeyDown);

  return () => {
    unsubscribe();
    cancelHide();
    anchor.removeEventListener('mouseenter', show);
    anchor.removeEventListener('mouseleave', scheduleHide);
    anchor.removeEventListener('focusin', show);
    document.removeEventListener('keydown', onKeyDown);
    popover.remove();
  };
}

function getResponseManager(panel: NotebookPanel): CraneResponseManager {
  let manager = RESPONSE_MANAGERS.get(panel);
  if (!manager) {
    manager = new CraneResponseManager(panel);
    RESPONSE_MANAGERS.set(panel, manager);
  }
  return manager;
}

class CraneResponseManager {
  private responseEntries = new Map<any, ResponseEntry>();
  private trackedCellModels = new WeakSet<any>();
  private executionCounts = new Map<any, number | null | undefined>();
  private connectedCells: any = null;
  /** Cells scheduled by the user that have not reported completion yet. */
  private pendingExecutions = 0;

  constructor(private panel: NotebookPanel) {
    this.attachSessionSignals();

    // The document model is loaded asynchronously, so at the moment this
    // manager is created (when the panel is added to the tracker) the model is
    // usually still null. Attaching only once, synchronously, left the manager
    // permanently deaf to every cell signal: responses were then never cleared
    // when their cell was re-run. Attach now and again once the model arrives.
    this.attachNotebookSignals();
    this.panel.content.modelChanged.connect(() => this.attachNotebookSignals(), this);
    void this.panel.context.ready.then(() => this.attachNotebookSignals());

    // The authoritative signals for "this cell ran". Watching executionCount
    // alone is fragile, because it depends on those per-cell signals having
    // been connected in time.
    NotebookActions.executionScheduled.connect(this.handleExecutionScheduled, this);
    NotebookActions.executed.connect(this.handleExecuted, this);

    this.panel.disposed.connect(() => this.dispose());
  }

  private dispose(): void {
    NotebookActions.executionScheduled.disconnect(this.handleExecutionScheduled, this);
    NotebookActions.executed.disconnect(this.handleExecuted, this);
    this.clearAll();
  }

  /** True while the user has cells queued or running in this notebook. */
  get hasPendingExecutions(): boolean {
    return this.pendingExecutions > 0;
  }

  /** Re-running the analysed cell retires its prediction immediately. */
  private handleExecutionScheduled(_sender: unknown, args: { notebook: any; cell: any }): void {
    if (args?.notebook !== this.panel.content) {
      return;
    }
    this.pendingExecutions += 1;
    this.removeEntry(args.cell?.model);
  }

  /** Any completed execution changes the state the other predictions rested on. */
  private handleExecuted(_sender: unknown, args: { notebook: any; cell: any }): void {
    if (args?.notebook !== this.panel.content) {
      return;
    }
    this.pendingExecutions = Math.max(0, this.pendingExecutions - 1);
    this.removeEntry(args.cell?.model);
    this.markAllStale();
    this.refreshExecutionSnapshot();
  }

  renderResponse(cell: any, response: string): void {
    this.removeEntry(cell.model);

    const entry = this.createEntry(cell, response);
    this.responseEntries.set(cell.model, entry);
    this.refreshExecutionSnapshot();
  }

  clearResponseForCell(cell: any): void {
    this.removeEntry(cell?.model);
  }

  /** Idempotent: safe to call again whenever the model may have appeared. */
  private attachNotebookSignals(): void {
    const cells = this.panel.content.model?.cells;
    if (!cells) {
      return;
    }

    this.attachCellSignalsFromNotebook();

    if (this.connectedCells !== cells && typeof cells.changed?.connect === 'function') {
      cells.changed.connect(this.handleNotebookCellsChanged, this);
      this.connectedCells = cells;
    }

    this.refreshExecutionSnapshot();
  }

  private attachSessionSignals(): void {
    const sessionContext = this.panel.sessionContext as any;
    const statusChanged = sessionContext?.statusChanged;
    const kernelChanged = sessionContext?.kernelChanged;

    if (statusChanged && typeof statusChanged.connect === 'function') {
      statusChanged.connect(this.handleSessionStatusChanged, this);
    }

    if (kernelChanged && typeof kernelChanged.connect === 'function') {
      kernelChanged.connect(this.handleKernelChanged, this);
    }
  }

  private attachCellSignalsFromNotebook(): void {
    const cells = this.panel.content.model?.cells;
    if (!cells) {
      return;
    }

    for (let index = 0; index < cells.length; index += 1) {
      this.attachCellSignals(cells.get(index));
    }
  }

  private attachCellSignals(cell: any): void {
    if (!cell || cell.type !== 'code' || this.trackedCellModels.has(cell)) {
      return;
    }

    this.trackedCellModels.add(cell);

    // Execution counts change via stateChanged; source edits arrive on
    // contentChanged. Handling them separately keeps typing from triggering a
    // full notebook sweep on every keystroke.
    if (cell.stateChanged && typeof cell.stateChanged.connect === 'function') {
      cell.stateChanged.connect(this.handleCellStateChanged, this);
    }
    if (cell.contentChanged && typeof cell.contentChanged.connect === 'function') {
      cell.contentChanged.connect(this.handleCellContentChanged, this);
    }
  }

  private handleNotebookCellsChanged(): void {
    this.attachCellSignalsFromNotebook();
    this.reconcileDeletedCells();
    this.refreshExecutionSnapshot();
  }

  /** Drop entries whose cell is no longer part of the notebook. */
  private reconcileDeletedCells(): void {
    if (this.responseEntries.size === 0) {
      return;
    }

    const live = new Set<any>();
    const cells = this.panel.content.model?.cells;
    if (cells) {
      for (let index = 0; index < cells.length; index += 1) {
        live.add(cells.get(index));
      }
    }

    for (const cellModel of Array.from(this.responseEntries.keys())) {
      if (!live.has(cellModel)) {
        this.removeEntry(cellModel);
      }
    }
  }

  private handleCellStateChanged(_sender: any, args: any): void {
    if (args && args.name && args.name !== 'executionCount') {
      return;
    }
    this.syncExecutionCounts();
  }

  /**
   * Editing a cell invalidates any prediction made about it, because the
   * verdict on screen describes code the user has since changed.
   */
  private handleCellContentChanged(sender: any): void {
    const entry = this.responseEntries.get(sender);
    if (entry && !entry.stale) {
      entry.stale = true;
      this.applyEntryStyle(entry);
    }
  }

  private handleSessionStatusChanged(_sender: unknown, status: string): void {
    // 'autorestarting' is what a kernel that died on its own reports. Without
    // it, every response box survives a crash that wiped the namespace.
    if (status === 'restarting' || status === 'autorestarting' || status === 'dead') {
      // Executions in flight will never report completion now, so the counter
      // would otherwise stay above zero and block analysis forever.
      this.pendingExecutions = 0;
      this.clearAll();
    }
  }

  private handleKernelChanged(): void {
    const sessionContext = this.panel.sessionContext as any;
    if (!sessionContext?.session?.kernel) {
      this.clearAll();
    }
  }

  private syncExecutionCounts(): void {
    const cells = this.panel.content.model?.cells;
    if (!cells) {
      return;
    }

    const executedCellModels: any[] = [];
    for (let index = 0; index < cells.length; index += 1) {
      const cell = cells.get(index) as any;
      this.attachCellSignals(cell);

      if (cell.type !== 'code') {
        continue;
      }

      if (this.executionCounts.get(cell) !== cell.executionCount) {
        executedCellModels.push(cell);
      }
    }

    if (executedCellModels.length === 0) {
      this.refreshExecutionSnapshot();
      return;
    }

    // Re-running the analysed cell retires its prediction outright. Running any
    // other cell changes the kernel state the prediction rested on, so the
    // remaining predictions are only marked stale.
    let removedTrackedEntry = false;
    for (const cellModel of executedCellModels) {
      removedTrackedEntry = this.removeEntry(cellModel) || removedTrackedEntry;
    }

    if (removedTrackedEntry || this.responseEntries.size > 0) {
      this.markAllStale();
    }

    this.refreshExecutionSnapshot();
  }

  private refreshExecutionSnapshot(): void {
    const cells = this.panel.content.model?.cells;
    if (!cells) {
      return;
    }

    this.executionCounts.clear();
    for (let index = 0; index < cells.length; index += 1) {
      const cell = cells.get(index) as any;
      if (cell.type === 'code') {
        this.executionCounts.set(cell, cell.executionCount);
      }
    }
  }

  private removeEntry(cellModel: any): boolean {
    const entry = cellModel ? this.responseEntries.get(cellModel) : undefined;
    if (!entry) {
      return false;
    }

    entry.container.remove();
    clearCellAccent(entry.cellNode);
    this.responseEntries.delete(cellModel);
    return true;
  }

  private clearAll(): void {
    for (const entry of this.responseEntries.values()) {
      entry.container.remove();
      clearCellAccent(entry.cellNode);
    }

    this.responseEntries.clear();
    this.executionCounts.clear();
    this.refreshExecutionSnapshot();
  }

  private markAllStale(): void {
    for (const entry of this.responseEntries.values()) {
      if (!entry.stale) {
        entry.stale = true;
        this.applyEntryStyle(entry);
      }
    }
  }

  private createEntry(cell: any, response: string): ResponseEntry {
    const container = document.createElement('div');
    const header = document.createElement('div');
    const titleNode = document.createElement('div');
    const closeButton = document.createElement('button');
    const responseNode = document.createElement('pre');

    container.className = 'crane-llm-result';
    container.style.cssText = [
      'margin:8px 0 4px 0',
      'padding:10px 12px',
      'border-left:4px solid var(--jp-brand-color1)',
      'background:var(--jp-layout-color2)',
      'border-radius:0 8px 8px 0'
    ].join(';');

    header.style.cssText =
      'display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;';

    titleNode.style.cssText = 'font-size:12px;font-weight:700;';

    closeButton.type = 'button';
    closeButton.textContent = 'Close';
    closeButton.className = 'jp-mod-styled jp-Button';
    closeButton.style.cssText = 'font-size:11px;padding:2px 8px;min-height:24px;';
    closeButton.onclick = () => {
      this.removeEntry(cell.model);
    };

    responseNode.textContent = response;
    responseNode.style.cssText = 'margin:0;white-space:pre-wrap;word-break:break-word;';

    header.appendChild(titleNode);
    header.appendChild(closeButton);
    container.appendChild(header);
    container.appendChild(responseNode);

    // Appended inside the cell's own node rather than beside it. A sibling
    // belongs to the notebook's Lumino layout, which addresses its children by
    // index and detaches them under the windowed rendering modes: a foreign
    // sibling misplaces later insertions and is orphaned when the cell is
    // deleted. A child travels with the cell and dies with it.
    cell.node.appendChild(container);

    const entry: ResponseEntry = {
      container,
      titleNode,
      responseNode,
      cellNode: cell.node,
      stale: false
    };

    this.applyEntryStyle(entry, response);
    return entry;
  }

  private applyEntryStyle(
    entry: ResponseEntry,
    responseText: string = entry.responseNode.textContent || ''
  ): void {
    const verdict = readVerdict(responseText);
    const tone: PredictionTone = entry.stale ? 'stale' : verdict.tone;
    const color = getToneColor(tone);

    entry.container.style.borderLeftColor = color;
    entry.titleNode.textContent = entry.stale
      ? `CRANE-LLM: ${verdict.label} (stale)`
      : `CRANE-LLM: ${verdict.label}`;

    // Also mark the cell itself, which is where the verdict is actually
    // looked for. An inset shadow rather than a border, so nothing reflows.
    entry.cellNode.style.boxShadow = `inset 4px 0 0 0 ${color}`;
  }
}

function clearCellAccent(cellNode: HTMLElement | undefined): void {
  if (cellNode) {
    cellNode.style.boxShadow = '';
  }
}

function getToneColor(tone: PredictionTone): string {
  if (tone === 'crash') {
    return '#dc2626';
  }
  if (tone === 'safe') {
    return '#16a34a';
  }
  if (tone === 'stale') {
    return '#6b7280';
  }
  // 'unknown' must not reuse the brand colour, which is also the container's
  // default border: an unreadable response would then look like no verdict.
  return '#d97706';
}

/** Tolerate the model wrapping its JSON object in a Markdown code fence. */
function stripCodeFence(text: string): string {
  const match = /^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/i.exec(text);
  return match ? match[1] : text;
}

/** The outermost {...} span, for responses that carry prose around the object. */
function firstJsonObject(text: string): string | null {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  return start !== -1 && end > start ? text.slice(start, end + 1) : null;
}

function parseResponseJson(responseText: string): Record<string, unknown> | null {
  for (const candidate of [responseText, stripCodeFence(responseText), firstJsonObject(responseText)]) {
    if (!candidate) {
      continue;
    }
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === 'object') {
        return parsed as Record<string, unknown>;
      }
    } catch (_error) {
      // Try the next candidate.
    }
  }
  return null;
}

function readVerdict(responseText: string): Verdict {
  const parsed = parseResponseJson(responseText);

  if (parsed) {
    // `prediction` is this extension's schema; `detection` is the key the
    // offline experiment prompts use. Accept either.
    const raw = 'prediction' in parsed ? parsed.prediction : parsed.detection;
    if (raw === true || raw === 'true') {
      return { tone: 'crash', label: 'crash predicted' };
    }
    if (raw === false || raw === 'false') {
      return { tone: 'safe', label: 'no crash predicted' };
    }
  }

  return { tone: 'unknown', label: 'response not understood' };
}

function cellSource(cell: any): string {
  const model = cell?.model;
  if (!model) {
    return '';
  }

  if (model.sharedModel && typeof model.sharedModel.getSource === 'function') {
    return model.sharedModel.getSource();
  }

  const serialized = model.toJSON?.();
  if (serialized) {
    if (typeof serialized.source === 'string') {
      return serialized.source;
    }
    if (Array.isArray(serialized.source)) {
      return serialized.source.join('');
    }
  }

  if (model.value && typeof model.value.text === 'string') {
    return model.value.text;
  }

  return '';
}

/**
 * The notebook's own id for a cell. The kernel sees the same id on every
 * execute request, which lets the backend keep one ledger entry per cell
 * instead of one per execution.
 */
function cellId(cell: any): string {
  const model = cell?.model;
  const id = model?.sharedModel?.getId?.() ?? model?.id;
  return typeof id === 'string' && id ? id : 'active-cell';
}

/**
 * Run code in the notebook's kernel and return its stdout.
 *
 * Only `stdout` stream messages are collected. Jupyter delivers stderr as a
 * stream message too, so library warnings and progress bars would otherwise be
 * spliced into whatever the caller parses out of the result.
 */
function requestKernelText(panel: NotebookPanel, code: string): Promise<string> {
  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel) {
    return Promise.reject(new Error('No kernel is connected to this notebook.'));
  }

  return new Promise<string>((resolve, reject) => {
    let stdout = '';
    let stderr = '';

    const future = kernel.requestExecute({
      code,
      stop_on_error: true,
      store_history: false,
      silent: false
    });

    const timer = setTimeout(() => {
      future.dispose();
      reject(
        new Error(
          `The kernel did not respond within ${Math.round(
            KERNEL_TIMEOUT_MS / 1000
          )}s. It may still be busy running another cell.`
        )
      );
    }, KERNEL_TIMEOUT_MS);

    const settle = (fn: () => void) => {
      clearTimeout(timer);
      fn();
    };

    future.onIOPub = message => {
      const msgType = message.header.msg_type;
      if (msgType === 'stream') {
        const content = message.content as { name?: string; text?: string };
        if (content.name === 'stderr') {
          stderr += content.text ?? '';
        } else {
          stdout += content.text ?? '';
        }
      } else if (msgType === 'error') {
        const content = message.content as { ename?: string; evalue?: string };
        settle(() =>
          reject(new Error(`${content.ename ?? 'Error'}: ${content.evalue ?? ''}`))
        );
      }
    };

    future.done
      .then(() =>
        settle(() => {
          if (!stdout.trim() && stderr.trim()) {
            reject(new Error(stderr.trim()));
            return;
          }
          resolve(stdout);
        })
      )
      .catch(error => settle(() => reject(error)));
  });
}

function extractPayload(streamText: string): CranePayload {
  const start = streamText.indexOf(PAYLOAD_BEGIN);
  const end = streamText.lastIndexOf(PAYLOAD_END);

  if (start === -1 || end === -1 || end < start) {
    throw new Error(
      `The kernel did not return a CRANE-LLM payload. Output was:\n${streamText.trim()}`
    );
  }

  return JSON.parse(
    streamText.slice(start + PAYLOAD_BEGIN.length, end)
  ) as CranePayload;
}

/**
 * Why the notebook is not ready to be analysed, or null if it is.
 *
 * CRANE-LLM has to run code in the kernel to read the namespace. A kernel
 * serves execute requests in order, so starting while the user's cells are
 * running would queue the request behind them: the sidebar would sit there for
 * as long as the run takes, and the prompt would finally be built from the
 * namespace as it is *after* those cells, which is not the state the user was
 * asking about. Refuse instead of producing a misleading answer.
 */
function notebookNotReadyReason(
  panel: NotebookPanel,
  manager: CraneResponseManager
): string | null {
  const kernel = panel.sessionContext.session?.kernel;

  if (!kernel) {
    return 'No kernel is connected to this notebook. Start the kernel first.';
  }

  if (kernel.status === 'dead') {
    return 'The kernel is not running. Restart it first.';
  }

  if (kernel.status === 'restarting' || kernel.status === 'autorestarting') {
    return 'The kernel is restarting. Wait for it to come back, then try again.';
  }

  // Two independent signals. The counter catches cells queued through the
  // notebook UI; the kernel status also catches work started from elsewhere,
  // and covers the brief gap between two queued cells where the counter has
  // been decremented but the kernel is still busy.
  if (manager.hasPendingExecutions || kernel.status === 'busy') {
    return 'CRANE-LLM cannot run while the notebook is executing. Wait for the running cells to finish, then try again.';
  }

  return null;
}

async function runAnalysis(
  app: JupyterFrontEnd,
  panel: NotebookPanel,
  sidebar: CraneSidebar,
  setting: RuninfoSetting
): Promise<void> {
  const activeCell = panel.content.activeCell;
  if (!activeCell || activeCell.model.type !== 'code') {
    await showErrorMessage('CRANE-LLM', 'Select a code cell first.');
    return;
  }

  // One analysis per notebook at a time. Without this, a double click starts a
  // second run whose results race the first one into the sidebar.
  if (RUNNING_PANELS.has(panel)) {
    return;
  }

  const notReadyReason = notebookNotReadyReason(panel, getResponseManager(panel));
  if (notReadyReason) {
    sidebar.setStatus('not run: the notebook is busy');
    sidebar.setResponse(notReadyReason);
    await showErrorMessage('CRANE-LLM', notReadyReason);
    return;
  }

  RUNNING_PANELS.add(panel);

  try {
    const responseManager = getResponseManager(panel);
    responseManager.clearResponseForCell(activeCell);

    const includeRuninfo = setting.get();

    app.shell.activateById(SIDEBAR_ID);
    sidebar.setStatus(
      includeRuninfo
        ? 'building prompt with runtime information...'
        : 'building prompt from executed code only...'
    );
    sidebar.setPrompt('');
    sidebar.setResponse('');

    // A single round trip. Fetching the prompt into the browser and posting it
    // back for the LLM call doubled the latency and made the prompt itself
    // depend on whatever else the kernel happened to print.
    const streamText = await requestKernelText(
      panel,
      [
        'from nb_extension.api import run_crane_llm_payload',
        `print(run_crane_llm_payload(source=${JSON.stringify(
          cellSource(activeCell)
        )}, cell_id=${JSON.stringify(cellId(activeCell))}, include_runinfo=${
          includeRuninfo ? 'True' : 'False'
        }))`
      ].join('\n')
    );

    const payload = extractPayload(streamText);
    sidebar.setPrompt(payload.prompt ?? '');

    if (!payload.ok) {
      sidebar.setStatus('error');
      sidebar.setResponse(payload.error || 'The kernel reported an unknown error.');
      return;
    }

    sidebar.setStatus(includeRuninfo ? 'done' : 'done, without runtime information');
    sidebar.setResponse(payload.response);
    responseManager.renderResponse(activeCell, payload.response);
  } finally {
    RUNNING_PANELS.delete(panel);
  }
}

function reportError(sidebar: CraneSidebar, error: unknown): void {
  sidebar.setStatus('error');
  sidebar.setResponse(error instanceof Error ? error.message : String(error));
}

function installToolbarButton(
  panel: NotebookPanel,
  app: JupyterFrontEnd,
  sidebar: CraneSidebar,
  setting: RuninfoSetting
): void {
  if (INSTALLED_PANELS.has(panel)) {
    return;
  }

  INSTALLED_PANELS.add(panel);
  getResponseManager(panel);

  const button = new ToolbarButton({
    label: 'CRANE-LLM',
    tooltip: 'Predict whether the selected cell will crash',
    onClick: () => {
      void runAnalysis(app, panel, sidebar, setting).catch(error =>
        reportError(sidebar, error)
      );
    }
  });

  panel.toolbar.addItem('crane-llm', button);

  // The switch also lives here, so it is discoverable without opening the
  // sidebar. Torn down with the panel to avoid orphaning it on document.body.
  const teardown = installRuninfoPopover(button.node, setting);
  panel.disposed.connect(() => teardown());
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'crane-llm-jlab:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette],
  activate: (
    app: JupyterFrontEnd,
    tracker: INotebookTracker,
    palette: ICommandPalette | null
  ) => {
    const setting = new RuninfoSetting();
    const sidebar = new CraneSidebar(setting);
    sidebar.id = SIDEBAR_ID;
    sidebar.title.label = 'CRANE-LLM';
    sidebar.title.caption = 'CRANE-LLM prompt and response';
    sidebar.title.closable = true;
    app.shell.add(sidebar, 'right');

    const execute = async () => {
      const panel = tracker.currentWidget;
      if (!panel) {
        await showErrorMessage('CRANE-LLM', 'Open a notebook first.');
        return;
      }

      // The command registry swallows rejections, so a failure triggered from
      // the palette has to be surfaced here.
      try {
        await runAnalysis(app, panel, sidebar, setting);
      } catch (error) {
        reportError(sidebar, error);
        await showErrorMessage(
          'CRANE-LLM',
          error instanceof Error ? error.message : String(error)
        );
      }
    };

    app.commands.addCommand(COMMAND_ID, {
      label: 'Run CRANE-LLM',
      caption: 'Build the CRANE prompt from the active notebook cell and run the LLM',
      execute
    });

    app.commands.addCommand(TOGGLE_RUNINFO_COMMAND_ID, {
      label: 'CRANE-LLM: Include Runtime Information',
      caption:
        'Include the live kernel state in the prompt. When off, only the executed cells and the target cell are sent.',
      isToggleable: true,
      isToggled: () => setting.get(),
      execute: () => setting.toggle()
    });

    // Keep the palette's checkmark correct when the switch is changed from the
    // sidebar or the toolbar popover instead.
    setting.subscribe(() => {
      app.commands.notifyCommandChanged(TOGGLE_RUNINFO_COMMAND_ID);
    });

    if (palette) {
      palette.addItem({ command: COMMAND_ID, category: 'Notebook' });
      palette.addItem({ command: TOGGLE_RUNINFO_COMMAND_ID, category: 'Notebook' });
    }

    tracker.widgetAdded.connect((_sender, panel) => {
      installToolbarButton(panel, app, sidebar, setting);
    });

    tracker.currentChanged.connect(() => {
      const panel = tracker.currentWidget;
      if (panel) {
        installToolbarButton(panel, app, sidebar, setting);
      }
    });

    tracker.forEach(panel => {
      installToolbarButton(panel, app, sidebar, setting);
    });
  }
};

export default plugin;
