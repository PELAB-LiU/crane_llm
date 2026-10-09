import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, showErrorMessage, ToolbarButton } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';

import TEXTS from './ui_texts.json';

const COMMAND_ID = 'crane-llm-jlab:run';
const TOGGLE_RUNINFO_COMMAND_ID = 'crane-llm-jlab:toggle-runinfo';
const TOGGLE_LLM_COMMAND_ID = 'crane-llm-jlab:toggle-llm';
const TOGGLE_GUARD_COMMAND_ID = 'crane-llm-jlab:toggle-guard';
const SIDEBAR_ID = 'crane-llm-sidebar';

/** Must match api.PAYLOAD_BEGIN / api.PAYLOAD_END / api.STAGE_MARKER. */
const PAYLOAD_BEGIN = '<<<CRANE-LLM:BEGIN>>>';
const PAYLOAD_END = '<<<CRANE-LLM:END>>>';
const STAGE_MARKER = '<<<CRANE-LLM:STAGE>>>';

/**
 * The text at a dotted key of ui_texts.json, with its {placeholders} filled
 * in. Every user-facing text lives in that file. Mirrors texts.text in the
 * backend: a placeholder with no value is left as written.
 */
function t(key: string, values: Record<string, string | number> = {}): string {
  let node: any = TEXTS;
  for (const part of key.split('.')) {
    node = node?.[part];
  }
  if (typeof node !== 'string') {
    return key;
  }
  return node.replace(/\{(\w+)\}/g, (match: string, name: string) =>
    name in values ? String(values[name]) : match
  );
}

/** Shown until the backend reports its first step; never listed as a finished step. */
const STARTING = t('progress.starting');

/**
 * One progress step, printed by the backend while it works, already worded
 * for the user. Must match api.progress.
 */
interface StageEvent {
  stage: string;
  message: string;
  prompt: string;
}

/** Whether a kernel status means the namespace is gone. */
function isKernelGone(status: string): boolean {
  return status === 'restarting' || status === 'autorestarting' || status === 'dead';
}

/** Kernel calls are cheap, but a busy kernel queues them behind user code. */
const KERNEL_TIMEOUT_MS = 10 * 60 * 1000;

const INSTALLED_PANELS = new WeakSet<NotebookPanel>();
const RESPONSE_MANAGERS = new WeakMap<NotebookPanel, CraneResponseManager>();
const RUNNING_PANELS = new WeakSet<NotebookPanel>();

/** 'none': the checker ran alone, with the LLM off, and found nothing. */
type PredictionTone = 'crash' | 'safe' | 'unknown' | 'none' | 'stale';

/** Must match verdict.Verdict.to_json in the backend. */
interface Verdict {
  tone: 'crash' | 'safe' | 'unknown' | 'none';
  label: string;
  reasoning: string;
  /** 'check': a built-in check, certain. 'model': an LLM prediction. */
  source: 'check' | 'model';
  certain: boolean;
  model: string;
  variables: string[];
  /** Says who gave the verdict. */
  badge: string;
  /** Says how far the verdict can be trusted. */
  note: string;
}

/** Must match provenance.OriginStep.to_json. */
interface OriginStep {
  cell_id: string;
  execution_count: number | null;
  role: string;
  line: number;
  line_text: string;
  /** The cell's source as it ran, to find it again and to notice later edits. */
  source: string;
  note: string;
}

interface Origin {
  variable: string;
  summary: string;
  steps: OriginStep[];
}

interface ResponseEntry {
  container: HTMLDivElement;
  titleNode: HTMLSpanElement;
  originsNode: HTMLDivElement;
  /** The cell widget's node, so the accent can be cleared when the entry goes. */
  cellNode: HTMLElement | null;
  /** The code the verdict is about; the verdict goes stale once the cell's code differs. */
  source: string;
  verdict: Verdict;
  /** Notes placed on the cells the crash comes from, removed with the entry. */
  originMarks: OriginMark[];
  stale: boolean;
}

interface OriginMark {
  cellNode: HTMLElement;
  noteNode: HTMLDivElement;
}

interface CranePayload {
  ok: boolean;
  prompt: string;
  response: string;
  error: string;
  include_runinfo: boolean;
  use_llm: boolean;
  verdict: Verdict | null;
  origins: Origin[];
}

/** A new element with a class, and its text if given. The look is in style/index.css. */
function element<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className: string,
  text?: string
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  node.className = className;
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

/** How many live verdicts mark each origin cell, so one closing keeps the others' outline. */
const ORIGIN_MARK_COUNTS = new WeakMap<HTMLElement, number>();

/**
 * An on/off switch shared by every control that shows it, remembered across
 * page reloads.
 *
 * The sidebar checkbox, the toolbar popover and the command palette entry are
 * three views of one value. Without a single owner they would drift apart as
 * soon as the user changed it from one of them.
 */
class ToggleSetting {
  private value: boolean;
  private listeners = new Set<(value: boolean) => void>();

  /** The stored value wins over the default once the user has changed it. */
  constructor(
    private storageKey: string,
    private defaultValue = true
  ) {
    this.value = this.load();
  }

  private load(): boolean {
    // Storage can be unavailable in private windows, so never let a failure
    // here stop the extension from loading.
    try {
      const stored = window.localStorage.getItem(this.storageKey);
      return stored === null ? this.defaultValue : stored !== 'false';
    } catch (_error) {
      return this.defaultValue;
    }
  }

  get(): boolean {
    return this.value;
  }

  set(value: boolean): void {
    if (value === this.value) {
      return;
    }
    this.value = value;
    try {
      window.localStorage.setItem(this.storageKey, String(value));
    } catch (_error) {
      // Not worth surfacing; the switch simply will not persist.
    }
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

/**
 * The switches. Runtime information decides what the LLM is sent; the LLM
 * switch decides whether it is asked at all, so with it off the runtime
 * switch has nothing to act on. Both apply to the check the toolbar button
 * starts. The guard is separate: it has the kernel run the built-in checker
 * on every cell before it runs, and never involves the LLM.
 */
interface CraneSettings {
  runinfo: ToggleSetting;
  llm: ToggleSetting;
  guard: ToggleSetting;
}

/**
 * One switch as a checkbox row with a hint below it. Used by the sidebar and
 * the toolbar popover alike. Returns the row and an unsubscribe function.
 */
function switchRow(
  setting: ToggleSetting,
  label: string,
  hint: (value: boolean) => string
): { node: HTMLDivElement; teardown: () => void; setEnabled: (enabled: boolean, note: string) => void } {
  const node = document.createElement('div');
  const row = element('label', 'crane-llm-switch');
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  const hintNode = element('div', 'crane-llm-switch-hint');

  row.appendChild(checkbox);
  row.appendChild(element('span', 'crane-llm-switch-label', label));
  node.appendChild(row);
  node.appendChild(hintNode);

  let enabled = true;
  let disabledNote = '';
  const render = (value: boolean) => {
    checkbox.checked = value;
    hintNode.textContent = enabled ? hint(value) : disabledNote;
  };
  checkbox.onchange = () => setting.set(checkbox.checked);
  const teardown = setting.subscribe(render);

  const setEnabled = (value: boolean, note: string) => {
    enabled = value;
    disabledNote = note;
    checkbox.disabled = !value;
    row.classList.toggle('crane-llm-disabled', !value);
    render(setting.get());
  };

  return { node, teardown, setEnabled };
}

function runinfoHintText(includeRuninfo: boolean): string {
  return includeRuninfo ? t('runinfo.hint_on') : t('runinfo.hint_off');
}

function llmHintText(useLlm: boolean): string {
  return useLlm ? t('llm.hint_on') : t('llm.hint_off');
}

function guardHintText(guard: boolean): string {
  return guard ? t('guard.hint_on') : t('guard.hint_off');
}

/** The title over a group of switches. */
function switchGroupHeading(text: string, divider: boolean): HTMLDivElement {
  const node = element('div', 'crane-llm-switch-heading', text);
  node.classList.toggle('crane-llm-divider', divider);
  return node;
}

/**
 * All switches, in two groups, because they apply at different moments. The
 * first two decide how the toolbar button checks the selected cell: the LLM
 * switch first, since it decides whether the runtime-information one matters,
 * whose row is greyed out while the LLM is off. The guard applies whenever a
 * cell runs, and uses the built-in checker alone.
 */
function switchRows(settings: CraneSettings): { nodes: HTMLDivElement[]; teardown: () => void } {
  const llm = switchRow(settings.llm, t('llm.label'), llmHintText);
  const runinfo = switchRow(settings.runinfo, t('runinfo.label'), runinfoHintText);
  const guard = switchRow(settings.guard, t('guard.label'), guardHintText);
  const unsubscribe = settings.llm.subscribe(useLlm =>
    runinfo.setEnabled(useLlm, t('runinfo.hint_llm_off'))
  );
  return {
    nodes: [
      switchGroupHeading(t('switches.button_heading'), false),
      llm.node,
      runinfo.node,
      switchGroupHeading(t('switches.guard_heading'), true),
      guard.node
    ],
    teardown: () => {
      llm.teardown();
      runinfo.teardown();
      guard.teardown();
      unsubscribe();
    }
  };
}

class CraneSidebar extends Widget {
  /** Steps of the current run that are done, listed above the status. */
  private stepsNode: HTMLDivElement;
  private statusNode: HTMLDivElement;
  private promptNode: HTMLPreElement;
  private responseNode: HTMLPreElement;
  constructor(settings: CraneSettings) {
    super();
    this.addClass('crane-llm-sidebar');

    this.stepsNode = element('div', 'crane-llm-sidebar-steps');
    this.statusNode = element('div', 'crane-llm-sidebar-status', t('sidebar.idle'));
    this.promptNode = element('pre', 'crane-llm-sidebar-text crane-llm-prompt');
    this.responseNode = element('pre', 'crane-llm-sidebar-text');

    // The switches. The LLM and runtime information are on by default, which
    // is the configuration the approach is built around; the guard is off
    // until the user asks for it. They reflect changes made from the toolbar
    // popover or the command palette too.
    const switches = switchRows(settings);

    this.node.append(
      element('div', 'crane-llm-sidebar-title', t('sidebar.title')),
      this.stepsNode,
      this.statusNode,
      ...switches.nodes,
      element('div', 'crane-llm-sidebar-heading', t('sidebar.prompt_heading')),
      this.promptNode,
      element('div', 'crane-llm-sidebar-heading', t('sidebar.response_heading')),
      this.responseNode
    );
  }

  setStatus(text: string): void {
    this.statusNode.textContent = text;
  }

  /**
   * Show the steps already done, ticked, above the one now under way, so the
   * user sees that the checker ran and found nothing before the model was
   * asked.
   */
  showSteps(done: string[], current: string): void {
    this.stepsNode.replaceChildren(
      ...done.map(step => element('div', '', `${t('progress.done_mark')} ${step}`))
    );
    this.setStatus(current);
  }

  setPrompt(text: string): void {
    this.promptNode.textContent = text;
  }

  setResponse(text: string): void {
    this.responseNode.textContent = text;
  }

  /** Show a finished verdict: who gave it, what was sent, what came back. */
  showVerdict(payload: CranePayload): void {
    const verdict = payload.verdict as Verdict;
    if (verdict.certain) {
      this.setStatus(t('sidebar.status_check_done'));
      this.setPrompt(t('sidebar.no_prompt_check'));
      this.setResponse(verdict.reasoning);
      return;
    }
    if (verdict.source === 'check') {
      this.setStatus(t('sidebar.status_check_only_done'));
      this.setPrompt(t('sidebar.no_prompt_llm_off'));
      this.setResponse(verdict.note);
      return;
    }
    this.setStatus(
      (verdict.model
        ? t('sidebar.status_model_done', { model: verdict.model })
        : t('sidebar.status_model_done_no_name')) +
        (payload.include_runinfo ? '' : t('sidebar.status_without_runinfo'))
    );
    this.setPrompt(payload.prompt ?? '');
    this.setResponse(payload.response);
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
function installSwitchPopover(anchor: HTMLElement, settings: CraneSettings): () => void {
  const popover = element('div', 'crane-llm-popover');
  const switches = switchRows(settings);
  popover.append(...switches.nodes);
  document.body.appendChild(popover);
  const unsubscribe = switches.teardown;

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
    popover.classList.add('crane-llm-open');
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
      popover.classList.remove('crane-llm-open');
    }, 250);
  };

  const hideNow = () => {
    cancelHide();
    popover.classList.remove('crane-llm-open');
  };

  // A click starts a check, and a check that cannot run opens a dialog. When
  // the dialog closes, JupyterLab gives the focus back to the button, which
  // must not reopen the panel: the pointer is elsewhere by then, so nothing
  // would ever close it. Focus opens the panel again only once the pointer
  // has come back to the button, or the user moves the focus with Tab.
  let clicked = false;
  const onClick = () => {
    clicked = true;
    hideNow();
  };
  const onMouseEnter = () => {
    clicked = false;
    show();
  };
  const onFocusIn = () => {
    if (!clicked) {
      show();
    }
  };
  const onFocusOut = (event: FocusEvent) => {
    if (!popover.contains(event.relatedTarget as Node | null)) {
      scheduleHide();
    }
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') {
      hideNow();
    } else if (event.key === 'Tab') {
      clicked = false;
    }
  };

  anchor.addEventListener('mouseenter', onMouseEnter);
  anchor.addEventListener('mouseleave', scheduleHide);
  // Keyboard users never fire mouseenter, so tabbing to the button opens it too.
  anchor.addEventListener('focusin', onFocusIn);
  anchor.addEventListener('focusout', onFocusOut);
  // Clicking the button starts a check; the switch has done its job by then,
  // and left open it would cover the progress shown under the toolbar.
  anchor.addEventListener('click', onClick);
  popover.addEventListener('mouseenter', cancelHide);
  popover.addEventListener('mouseleave', scheduleHide);
  popover.addEventListener('focusin', cancelHide);
  popover.addEventListener('focusout', scheduleHide);
  document.addEventListener('keydown', onKeyDown);

  return () => {
    unsubscribe();
    cancelHide();
    anchor.removeEventListener('mouseenter', onMouseEnter);
    anchor.removeEventListener('mouseleave', scheduleHide);
    anchor.removeEventListener('focusin', onFocusIn);
    anchor.removeEventListener('focusout', onFocusOut);
    anchor.removeEventListener('click', onClick);
    document.removeEventListener('keydown', onKeyDown);
    popover.remove();
  };
}

/** What NotebookActions.executed reports about a finished cell. */
interface ExecutedArgs {
  notebook: any;
  cell: any;
  success?: boolean;
  error?: { errorName?: string } | null;
}

/**
 * Whether the guard stopped the cell, so none of it ran and the kernel state
 * is what it was. Must match the name of guard.CrashPrevented.
 */
function stoppedByGuard(args: ExecutedArgs): boolean {
  return args.success === false && args.error?.errorName === 'CrashPrevented';
}

function getResponseManager(panel: NotebookPanel, settings: CraneSettings): CraneResponseManager {
  let manager = RESPONSE_MANAGERS.get(panel);
  if (!manager) {
    manager = new CraneResponseManager(panel, settings.guard);
    RESPONSE_MANAGERS.set(panel, manager);
  }
  return manager;
}

class CraneResponseManager {
  private responseEntries = new Map<any, ResponseEntry>();
  /** Progress boxes of checks still under way, by cell model. */
  private progressBoxes = new Map<any, HTMLDivElement>();
  private trackedCellModels = new WeakSet<any>();
  private connectedCells: any = null;
  /** Cells scheduled by the user that have not reported completion yet. */
  private pendingExecutions = 0;
  /** The kernel the backend has been loaded into, so it is loaded once per kernel. */
  private backendKernelId: string | null = null;
  /** A kernel the backend failed to load into, so the load is not retried on every idle. */
  private failedKernelId: string | null = null;
  /** The kernel the user was last told cannot run the guard, so they are told once. */
  private warnedKernelId: string | null = null;
  private loadingBackend = false;
  /** The guard switch as last sent to the kernel, null when not sent to this kernel yet. */
  private sentGuard: boolean | null = null;
  private unsubscribeGuard: () => void;

  constructor(
    private panel: NotebookPanel,
    private guard: ToggleSetting
  ) {
    this.attachSessionSignals();
    void this.panel.sessionContext.ready.then(() => this.ensureBackendLoaded());
    this.unsubscribeGuard = guard.subscribe(() => this.handleGuardChanged());

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
    this.unsubscribeGuard();
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
    if (args.cell?.model) {
      LIVE_CELLS.add(args.cell.model);
    }
    this.removeEntry(args.cell?.model);
  }

  /**
   * Any completed execution changes the state the other predictions rested
   * on, except one the guard stopped before any of it ran.
   */
  private handleExecuted(_sender: unknown, args: ExecutedArgs): void {
    if (args?.notebook !== this.panel.content) {
      return;
    }
    this.pendingExecutions = Math.max(0, this.pendingExecutions - 1);
    this.removeEntry(args.cell?.model);
    if (!stoppedByGuard(args)) {
      this.markAllStale();
    }
  }

  /**
   * Show, under the cell, the steps of a check still under way: finished ones
   * ticked, the current one last. Replaced by the verdict when it arrives.
   */
  showProgress(cell: any, done: string[], current: string): void {
    let box = this.progressBoxes.get(cell.model);
    if (!box) {
      box = element('div', 'crane-llm-progress');
      // Inside the cell's node, for the same reason as the verdict box.
      cell.node.appendChild(box);
      this.progressBoxes.set(cell.model, box);
    }
    box.replaceChildren(
      ...done.map(step => element('div', 'crane-llm-step-done', `${t('progress.done_mark')} ${step}`)),
      element('div', 'crane-llm-step-current', t('progress.current', { step: current }))
    );
  }

  clearProgress(cell: any): void {
    const box = this.progressBoxes.get(cell?.model);
    if (box) {
      box.remove();
      this.progressBoxes.delete(cell.model);
    }
  }

  renderResponse(cell: any, verdict: Verdict, origins: Origin[]): void {
    this.clearProgress(cell);
    this.removeEntry(cell.model);

    const entry = this.createEntry(cell, verdict, origins);
    this.responseEntries.set(cell.model, entry);
  }

  /** Clear the cell's verdicts, the guard's included, before it is checked again. */
  clearResponseForCell(cell: any): void {
    this.removeEntry(cell?.model);
    retireGuardOutputs(cell?.model);
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

    // Source edits arrive on contentChanged. Executions are followed through
    // NotebookActions, which also says whether the guard stopped the cell;
    // a changed execution count alone cannot tell.
    if (cell.contentChanged && typeof cell.contentChanged.connect === 'function') {
      cell.contentChanged.connect(this.handleCellContentChanged, this);
    }
  }

  private handleNotebookCellsChanged(): void {
    this.attachCellSignalsFromNotebook();
    this.reconcileDeletedCells();
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

  /**
   * Editing a cell invalidates any prediction made about it, because the
   * verdict on screen describes code the user has since changed. The signal
   * also fires when the cell's outputs change, which is not an edit, so the
   * code is compared.
   */
  private handleCellContentChanged(sender: any): void {
    const entry = this.responseEntries.get(sender);
    if (entry && !entry.stale && editedSince(entry, sender)) {
      entry.stale = true;
      applyEntryStyle(entry);
    }
  }

  private handleSessionStatusChanged(_sender: unknown, status: string): void {
    // 'autorestarting' is what a kernel that died on its own reports. Without
    // it, every response box survives a crash that wiped the namespace.
    if (isKernelGone(status)) {
      // Executions in flight will never report completion now, so the counter
      // would otherwise stay above zero and block analysis forever.
      this.pendingExecutions = 0;
      this.clearAll();
      // A restarted kernel keeps its id but has lost the backend, and the
      // guard with it. A load that failed is tried again, since crane_llm
      // may have been installed before the restart.
      this.backendKernelId = null;
      this.failedKernelId = null;
      this.warnedKernelId = null;
      this.sentGuard = null;
    } else if (status === 'idle') {
      this.ensureBackendLoaded();
    }
  }

  /**
   * Load the backend into the kernel as soon as it is idle, not at the first
   * check.
   *
   * The backend records what every cell does to the namespace, which is how a
   * crash is traced back to the cell it comes from, but only for cells run
   * after it is loaded. Cells run before that are known only from their code.
   * A kernel without crane_llm installed fails the import. That is not
   * retried until the kernel restarts or the guard is switched on, and it is
   * reported only when the guard is on: the user has asked for nothing else
   * yet, but with the guard on they believe their cells are being checked.
   *
   * The guard lives in the kernel and is lost when it restarts, so the
   * switch's value goes along with every load.
   */
  private ensureBackendLoaded(): void {
    const kernel = this.panel.sessionContext.session?.kernel;
    if (
      !kernel ||
      kernel.status !== 'idle' ||
      this.loadingBackend ||
      this.backendKernelId === kernel.id ||
      this.failedKernelId === kernel.id
    ) {
      return;
    }
    const kernelId = kernel.id;
    const guard = this.guard.get();
    this.loadingBackend = true;
    // The trailing semicolons keep IPython from displaying the return value,
    // which would also overwrite the user's `_`.
    requestKernelText(
      this.panel,
      'from crane_llm.nb_extension.api import load_crane_llm, set_guard\n' +
        `load_crane_llm();\nset_guard(${guard ? 'True' : 'False'});`
    )
      .then(() => {
        this.backendKernelId = kernelId;
        this.sentGuard = guard;
      })
      .catch(error => {
        this.failedKernelId = kernelId;
        this.warnGuardUnavailable(error);
      })
      .finally(() => {
        this.loadingBackend = false;
        // The switch may have changed while the load was under way.
        this.sendGuard();
      });
  }

  /**
   * Switching the guard on asks for it to work, so a kernel the backend
   * failed to load into gets another try, in case crane_llm has been
   * installed since. Otherwise the kernel is just told the new value.
   */
  private handleGuardChanged(): void {
    const kernel = this.panel.sessionContext.session?.kernel;
    if (kernel && this.guard.get() && this.failedKernelId === kernel.id) {
      this.failedKernelId = null;
      this.warnedKernelId = null;
      this.ensureBackendLoaded();
      return;
    }
    this.sendGuard();
  }

  /** Tell the kernel the guard switch's value, once the backend is loaded there. */
  private sendGuard(): void {
    const kernel = this.panel.sessionContext.session?.kernel;
    const guard = this.guard.get();
    if (!kernel || this.backendKernelId !== kernel.id || this.sentGuard === guard) {
      return;
    }
    this.sentGuard = guard;
    requestKernelText(
      this.panel,
      `from crane_llm.nb_extension.api import set_guard\nset_guard(${guard ? 'True' : 'False'});`
    ).catch(error => {
      this.sentGuard = null;
      this.warnGuardUnavailable(error);
    });
  }

  /** Say, once per kernel, that the guard is on but cannot work in this kernel. */
  private warnGuardUnavailable(error: unknown): void {
    const kernelId = this.panel.sessionContext.session?.kernel?.id ?? null;
    if (!this.guard.get() || kernelId === null || this.warnedKernelId === kernelId) {
      return;
    }
    this.warnedKernelId = kernelId;
    void showErrorMessage(
      t('guard.unavailable_title'),
      t('guard.unavailable', { error: error instanceof Error ? error.message : String(error) })
    );
  }

  private handleKernelChanged(): void {
    const sessionContext = this.panel.sessionContext as any;
    if (!sessionContext?.session?.kernel) {
      this.clearAll();
    }
  }

  private removeEntry(cellModel: any): boolean {
    const entry = cellModel ? this.responseEntries.get(cellModel) : undefined;
    if (!entry) {
      return false;
    }

    entry.container.remove();
    clearCellAccent(entry.cellNode);
    clearOriginMarks(entry);
    this.responseEntries.delete(cellModel);
    return true;
  }

  private clearAll(): void {
    for (const cellModel of Array.from(this.responseEntries.keys())) {
      this.removeEntry(cellModel);
    }
    for (const box of this.progressBoxes.values()) {
      box.remove();
    }
    this.progressBoxes.clear();
  }

  private markAllStale(): void {
    for (const entry of this.responseEntries.values()) {
      if (!entry.stale) {
        entry.stale = true;
        applyEntryStyle(entry);
      }
    }
  }

  private createEntry(cell: any, verdict: Verdict, origins: Origin[]): ResponseEntry {
    const entry = buildEntry(this.panel, cell, verdict, origins, {
      onClose: () => this.removeEntry(cell.model),
      markOrigins: true
    });
    // Appended inside the cell's own node rather than beside it. A sibling
    // belongs to the notebook's Lumino layout, which addresses its children by
    // index and detaches them under the windowed rendering modes: a foreign
    // sibling misplaces later insertions and is orphaned when the cell is
    // deleted. A child travels with the cell and dies with it.
    cell.node.appendChild(entry.container);
    applyEntryStyle(entry);
    return entry;
  }
}

interface EntryOptions {
  /** Adds a Close button that calls this. */
  onClose?: () => void;
  /** A last line under the verdict; `backticked` parts are shown as code. */
  footer?: string;
  /** Outline the origin cells and put a note in each. */
  markOrigins: boolean;
}

/**
 * A verdict box, not yet placed anywhere: the verdict, who gave it, and the
 * cells the crash comes from. Shared by the toolbar button's check and the
 * guard's output. ``panel`` and ``cell`` are null when the box is shown
 * outside a notebook; the origins are then listed without links.
 */
function buildEntry(
  panel: NotebookPanel | null,
  cell: any | null,
  verdict: Verdict,
  origins: Origin[],
  options: EntryOptions
): ResponseEntry {
  const container = element('div', 'crane-llm-result');
  const header = element('div', 'crane-llm-result-header');
  const heading = element('div', 'crane-llm-result-heading');
  const titleNode = element('span', 'crane-llm-result-title');
  const originsNode = element('div', 'crane-llm-origins');

  // Who gave the verdict is the first thing to know about it: a built-in
  // check is certain, a model prediction can be wrong.
  const badgeNode = element('span', 'crane-llm-badge', verdict.badge);
  badgeNode.classList.toggle('crane-llm-certain', verdict.certain);
  badgeNode.title = verdict.note;

  heading.append(titleNode, badgeNode);
  header.appendChild(heading);
  if (options.onClose) {
    const closeButton = element('button', 'jp-mod-styled jp-Button crane-llm-close', t('verdict.close'));
    closeButton.type = 'button';
    closeButton.onclick = options.onClose;
    header.appendChild(closeButton);
  }
  container.appendChild(header);
  if (verdict.reasoning) {
    container.appendChild(element('div', 'crane-llm-reasoning', verdict.reasoning));
  }
  container.appendChild(element('div', 'crane-llm-note', verdict.note));
  if (options.footer) {
    const footerNode = element('div', 'crane-llm-footer');
    footerNode.append(
      ...options.footer.split('`').map((piece, index) => element(index % 2 ? 'code' : 'span', '', piece))
    );
    container.appendChild(footerNode);
  }
  container.appendChild(originsNode);

  const entry: ResponseEntry = {
    container,
    titleNode,
    originsNode,
    cellNode: cell?.node ?? null,
    source: cell ? cellSource(cell) : '',
    verdict,
    originMarks: [],
    stale: false
  };

  renderOrigins(panel, entry, cell, origins, options.markOrigins);
  return entry;
}

/**
 * List the cells the crash comes from, and mark those cells in the notebook.
 *
 * The cell under analysis is seldom where the mistake was made: the list
 * names the cell that last set each blamed variable and every cell that
 * changed it since, and each entry jumps to its cell.
 */
function renderOrigins(
  panel: NotebookPanel | null,
  entry: ResponseEntry,
  targetCell: any | null,
  origins: Origin[],
  mark: boolean
): void {
  if (!origins.length) {
    return;
  }

  entry.originsNode.appendChild(element('div', 'crane-llm-origins-heading', t('origins.heading')));

  const targetLabel = cellLabel(panel, targetCell, null);

  for (const origin of origins) {
    entry.originsNode.appendChild(element('div', 'crane-llm-origin-summary', origin.summary));

    for (const step of origin.steps) {
      const cell = panel ? findCell(panel, step) : null;
      const edited = cell !== null && step.role !== 'defines' && cellSource(cell) !== step.source;
      const label =
        cell || !panel ? cellLabel(panel, cell, step.execution_count) : t('origins.cell_missing');
      const role = t(`origins.roles.${step.role}`);

      const row = element('div', 'crane-llm-origin-step');
      row.classList.toggle('crane-llm-linked', cell !== null);
      row.appendChild(
        element(
          'span',
          'crane-llm-origin-where',
          step.line
            ? t('origins.step_line', { role, cell: label, line: step.line })
            : t('origins.step', { role, cell: label })
        )
      );
      if (step.line_text) {
        row.appendChild(element('code', 'crane-llm-origin-code', step.line_text));
      }
      const notes = [step.note, edited ? t('origins.edited') : ''].filter(Boolean);
      if (notes.length) {
        row.appendChild(element('span', 'crane-llm-origin-step-note', ` (${notes.join('; ')})`));
      }

      if (panel && cell) {
        row.title = t('origins.go_to_cell');
        row.onclick = () => revealCell(panel, cell);
        if (mark) {
          entry.originMarks.push(markOriginCell(cell, origin.variable, step, targetLabel));
        }
      }
      entry.originsNode.appendChild(row);
    }
  }
}

function applyEntryStyle(entry: ResponseEntry): void {
  const verdict = entry.verdict;
  const tone: PredictionTone = entry.stale ? 'stale' : verdict.tone;

  entry.container.dataset.tone = tone;
  entry.titleNode.textContent = t(entry.stale ? 'verdict.title_stale' : 'verdict.title', {
    label: verdict.label
  });

  // Also mark the cell itself, which is where the verdict is actually
  // looked for.
  if (entry.cellNode) {
    entry.cellNode.dataset.craneLlmTone = tone;
  }

  // Once stale, the origins describe a kernel state that has moved on.
  if (entry.stale) {
    clearOriginMarks(entry);
  }
}

/** Must match guard.MIME_TYPE in the backend. */
const GUARD_MIME = 'application/vnd.crane-llm.guard+json';

/** Must match the payload guard.CellGuard displays. */
interface GuardPayload {
  verdict: Verdict;
  origins: Origin[];
  footer: string;
}

/**
 * Code cells run in this browser session. A guard output in any other cell
 * was saved with the notebook and describes a kernel that is gone.
 */
const LIVE_CELLS = new WeakSet<any>();

/** The guard outputs shown in each cell, by cell model, so a new check can retire them. */
const GUARD_OUTPUTS = new WeakMap<any, Set<GuardOutput>>();

/**
 * Remove the guard's verdict from a cell that is being checked again with
 * the toolbar button: the new verdict replaces it. The CrashPrevented error
 * under it stays, as the record of the run that was stopped.
 */
function retireGuardOutputs(cellModel: any): void {
  for (const output of GUARD_OUTPUTS.get(cellModel) ?? []) {
    output.retire();
  }
}

/**
 * The guard's output, drawn as the same verdict box the toolbar button gives.
 *
 * The kernel sends it as HTML too, which is what other frontends show, but
 * HTML cannot reach the notebook: here the origin entries jump to their
 * cells, and those cells are outlined until another cell runs. The box is
 * built when the output is attached, since only then is it inside its cell.
 */
class GuardOutput extends Widget {
  private payload: GuardPayload | null = null;
  private entry: ResponseEntry | null = null;
  private cell: any = null;

  constructor(private panel: NotebookPanel) {
    super();
    this.addClass('crane-llm-guard-output');
  }

  renderModel(model: any): Promise<void> {
    this.payload = model.data[GUARD_MIME] as GuardPayload;
    this.build();
    return Promise.resolve();
  }

  protected onAfterAttach(msg: any): void {
    super.onAfterAttach(msg);
    this.build();
  }

  private build(): void {
    if (this.entry || !this.payload || !this.isAttached) {
      return;
    }
    this.cell = this.panel.content.widgets.find(widget => widget.node.contains(this.node)) ?? null;
    const live = this.cell !== null && LIVE_CELLS.has(this.cell.model);
    const entry = buildEntry(this.panel, this.cell, this.payload.verdict, this.payload.origins, {
      footer: this.payload.footer,
      markOrigins: live
    });
    if (!live) {
      // Leave the cell itself alone: the output only records an old run.
      entry.cellNode = null;
      entry.stale = true;
    }
    this.node.appendChild(entry.container);
    applyEntryStyle(entry);
    this.entry = entry;
    if (live) {
      NotebookActions.executed.connect(this.handleExecuted, this);
      this.panel.sessionContext.statusChanged.connect(this.handleStatus, this);
      this.cell.model.contentChanged.connect(this.handleEdited, this);
      let outputs = GUARD_OUTPUTS.get(this.cell.model);
      if (!outputs) {
        outputs = new Set();
        GUARD_OUTPUTS.set(this.cell.model, outputs);
      }
      outputs.add(this);
    }
  }

  /** Stale once the cell's code is no longer the code that was checked; outputs do not count. */
  private handleEdited(): void {
    if (this.entry && editedSince(this.entry, this.cell?.model)) {
      this.markStale();
    }
  }

  /** Hide the verdict and take its marks off the notebook. */
  retire(): void {
    this.cleanUp();
    this.hide();
  }

  /**
   * Like the button's verdicts, this one goes stale when another cell runs,
   * but not when the guard stopped that cell too.
   */
  private handleExecuted(_sender: unknown, args: ExecutedArgs): void {
    if (args?.notebook === this.panel.content && args.cell !== this.cell && !stoppedByGuard(args)) {
      this.markStale();
    }
  }

  private handleStatus(_sender: unknown, status: string): void {
    if (isKernelGone(status)) {
      this.markStale();
    }
  }

  private markStale(): void {
    this.disconnect();
    if (this.entry && !this.entry.stale) {
      this.entry.stale = true;
      applyEntryStyle(this.entry);
    }
  }

  private disconnect(): void {
    NotebookActions.executed.disconnect(this.handleExecuted, this);
    this.panel.sessionContext.statusChanged.disconnect(this.handleStatus, this);
    this.cell?.model?.contentChanged?.disconnect(this.handleEdited, this);
  }

  private cleanUp(): void {
    this.disconnect();
    GUARD_OUTPUTS.get(this.cell?.model)?.delete(this);
    if (this.entry) {
      clearOriginMarks(this.entry);
      clearCellAccent(this.entry.cellNode);
      this.entry.cellNode = null;
    }
  }

  /** Disposed when the output is cleared, for one when the cell runs again. */
  dispose(): void {
    if (this.isDisposed) {
      return;
    }
    this.cleanUp();
    super.dispose();
  }
}

/** Draw guard outputs in this notebook with GuardOutput instead of their HTML. */
function installGuardRenderer(panel: NotebookPanel): void {
  panel.content.rendermime.addFactory(
    {
      safe: true,
      mimeTypes: [GUARD_MIME],
      defaultRank: 0,
      createRenderer: () => new GuardOutput(panel)
    },
    0
  );
}

function clearCellAccent(cellNode: HTMLElement | null | undefined): void {
  if (cellNode) {
    delete cellNode.dataset.craneLlmTone;
  }
}

/** The cell a step refers to: by notebook id, or by source for cells replayed from history. */
function findCell(panel: NotebookPanel, step: OriginStep): any | null {
  const widgets = panel.content.widgets;
  for (const widget of widgets) {
    if (widget.model.type === 'code' && cellId(widget) === step.cell_id) {
      return widget;
    }
  }
  for (const widget of widgets) {
    if (widget.model.type === 'code' && cellSource(widget) === step.source) {
      return widget;
    }
  }
  return null;
}

/**
 * "cell [7]" after the execution count, as the kernel names cells in the
 * summary too, or "cell #3" after its position for a cell that has none.
 */
function cellLabel(panel: NotebookPanel | null, cell: any | null, executionCount: number | null): string {
  const count = executionCount ?? cell?.model?.executionCount;
  if (typeof count === 'number') {
    return t('origins.cell_with_count', { count });
  }
  const index = panel && cell ? panel.content.widgets.indexOf(cell) : -1;
  return index >= 0 ? t('origins.cell_with_position', { position: index + 1 }) : t('origins.cell_unknown');
}

function revealCell(panel: NotebookPanel, cell: any): void {
  const index = panel.content.widgets.indexOf(cell);
  if (index < 0) {
    return;
  }
  panel.content.activeCellIndex = index;
  const notebook = panel.content as any;
  if (typeof notebook.scrollToCell === 'function') {
    void notebook.scrollToCell(cell, 'center');
  } else {
    cell.node.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
}

/** Outline an origin cell and say, inside it, what it did. */
function markOriginCell(cell: any, variable: string, step: OriginStep, targetLabel: string): OriginMark {
  const cellNode: HTMLElement = cell.node;
  const count = ORIGIN_MARK_COUNTS.get(cellNode) ?? 0;
  ORIGIN_MARK_COUNTS.set(cellNode, count + 1);
  cellNode.classList.add('crane-llm-origin-cell');

  const action =
    step.role === 'defines' ? t('origins.cell_note_action_defines') : t(`origins.roles.${step.role}`);
  const noteNode = element(
    'div',
    'crane-llm-origin-note',
    step.line
      ? t('origins.cell_note_line', { action, variable, line: step.line, target: targetLabel })
      : t('origins.cell_note', { action, variable, target: targetLabel })
  );
  cellNode.appendChild(noteNode);
  return { cellNode, noteNode };
}

function clearOriginMarks(entry: ResponseEntry): void {
  for (const mark of entry.originMarks) {
    mark.noteNode.remove();
    const count = (ORIGIN_MARK_COUNTS.get(mark.cellNode) ?? 1) - 1;
    ORIGIN_MARK_COUNTS.set(mark.cellNode, count);
    if (count <= 0) {
      mark.cellNode.classList.remove('crane-llm-origin-cell');
    }
  }
  entry.originMarks = [];
}

/** Whether the cell's code differs from the code a verdict was given for. */
function editedSince(entry: ResponseEntry, cellModel: any): boolean {
  return cellSource({ model: cellModel }) !== entry.source;
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
function requestKernelText(
  panel: NotebookPanel,
  code: string,
  onStdout?: (text: string) => void
): Promise<string> {
  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel) {
    return Promise.reject(new Error(t('errors.no_kernel')));
  }

  return new Promise<string>((resolve, reject) => {
    let stdout = '';
    let stderr = '';

    // stop_on_error must stay off. With it on, a failure here (crane_llm not
    // installed in this kernel, say) makes the kernel abort every request
    // queued behind this one, which includes cells the user has just run.
    const future = kernel.requestExecute({
      code,
      stop_on_error: false,
      store_history: false,
      silent: false
    });

    const timer = setTimeout(() => {
      future.dispose();
      reject(
        new Error(t('errors.kernel_timeout', { seconds: Math.round(KERNEL_TIMEOUT_MS / 1000) }))
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
          onStdout?.(content.text ?? '');
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
    throw new Error(t('errors.no_payload', { output: streamText.trim() }));
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
    return t('errors.no_kernel');
  }

  if (kernel.status === 'dead') {
    return t('errors.kernel_dead');
  }

  if (kernel.status === 'restarting' || kernel.status === 'autorestarting') {
    return t('errors.kernel_restarting');
  }

  // Two independent signals. The counter catches cells queued through the
  // notebook UI; the kernel status also catches work started from elsewhere,
  // and covers the brief gap between two queued cells where the counter has
  // been decremented but the kernel is still busy.
  if (manager.hasPendingExecutions || kernel.status === 'busy') {
    return t('errors.notebook_busy');
  }

  return null;
}

async function runAnalysis(
  app: JupyterFrontEnd,
  panel: NotebookPanel,
  sidebar: CraneSidebar,
  settings: CraneSettings
): Promise<void> {
  const activeCell = panel.content.activeCell;
  if (!activeCell || activeCell.model.type !== 'code') {
    await showErrorMessage(t('errors.dialog_title'), t('errors.select_code_cell'));
    return;
  }

  // One analysis per notebook at a time. Without this, a double click starts a
  // second run whose results race the first one into the sidebar.
  if (RUNNING_PANELS.has(panel)) {
    return;
  }

  const notReadyReason = notebookNotReadyReason(panel, getResponseManager(panel, settings));
  if (notReadyReason) {
    sidebar.setStatus(t('sidebar.status_busy'));
    sidebar.setResponse(notReadyReason);
    await showErrorMessage(t('errors.dialog_title'), notReadyReason);
    return;
  }

  RUNNING_PANELS.add(panel);

  try {
    const responseManager = getResponseManager(panel, settings);
    responseManager.clearResponseForCell(activeCell);

    const includeRuninfo = settings.runinfo.get();
    const useLlm = settings.llm.get();

    app.shell.activateById(SIDEBAR_ID);
    sidebar.setPrompt('');
    sidebar.setResponse('');

    // Each step is shown in the sidebar and under the cell as the backend
    // reports it, so a check that falls through to the model says so. The
    // step before the current one moves to the list of finished ones.
    const done: string[] = [];
    let current = '';
    const showStep = (text: string) => {
      if (current && current !== STARTING) {
        done.push(current);
      }
      current = text;
      sidebar.showSteps(done, text);
      responseManager.showProgress(activeCell, done, text);
    };
    showStep(STARTING);

    let pending = '';
    const onStdout = (text: string) => {
      pending += text;
      const lines = pending.split('\n');
      pending = lines.pop() ?? '';
      for (const line of lines) {
        const at = line.indexOf(STAGE_MARKER);
        if (at === -1) {
          continue;
        }
        try {
          const event = JSON.parse(line.slice(at + STAGE_MARKER.length)) as StageEvent;
          showStep(event.message);
          if (event.prompt) {
            sidebar.setPrompt(event.prompt);
          }
        } catch (_error) {
          // A malformed progress line only costs the progress display.
        }
      }
    };

    // Every code cell of the notebook, so that a variable that does not exist
    // can be traced to the cell that would define it.
    const notebookCells = panel.content.widgets
      .filter(widget => widget.model.type === 'code')
      .map(widget => ({ id: cellId(widget), source: cellSource(widget) }));

    // A single round trip. Fetching the prompt into the browser and posting it
    // back for the LLM call doubled the latency and made the prompt itself
    // depend on whatever else the kernel happened to print.
    const streamText = await requestKernelText(
      panel,
      [
        'from crane_llm.nb_extension.api import run_crane_llm_payload',
        `print(run_crane_llm_payload(source=${JSON.stringify(
          cellSource(activeCell)
        )}, cell_id=${JSON.stringify(cellId(activeCell))}, include_runinfo=${
          includeRuninfo ? 'True' : 'False'
        }, use_llm=${useLlm ? 'True' : 'False'}, notebook_cells_json=${JSON.stringify(
          JSON.stringify(notebookCells)
        )}))`
      ].join('\n'),
      onStdout
    );

    const payload = extractPayload(streamText);
    sidebar.setPrompt(payload.prompt ?? '');

    if (!payload.ok || !payload.verdict) {
      sidebar.setStatus(t('sidebar.status_error'));
      sidebar.setResponse(payload.error || t('errors.unknown_error'));
      return;
    }

    showStep('');
    sidebar.showVerdict(payload);
    responseManager.renderResponse(activeCell, payload.verdict, payload.origins ?? []);
  } finally {
    // On failure the error is in the sidebar; the progress box must not stay.
    getResponseManager(panel, settings).clearProgress(activeCell);
    RUNNING_PANELS.delete(panel);
  }
}

function reportError(sidebar: CraneSidebar, error: unknown): void {
  sidebar.setStatus(t('sidebar.status_error'));
  sidebar.setResponse(error instanceof Error ? error.message : String(error));
}

function installToolbarButton(
  panel: NotebookPanel,
  app: JupyterFrontEnd,
  sidebar: CraneSidebar,
  settings: CraneSettings
): void {
  if (INSTALLED_PANELS.has(panel)) {
    return;
  }

  INSTALLED_PANELS.add(panel);
  getResponseManager(panel, settings);
  installGuardRenderer(panel);

  const button = new ToolbarButton({
    label: t('toolbar.label'),
    tooltip: t('toolbar.tooltip'),
    onClick: () => {
      void runAnalysis(app, panel, sidebar, settings).catch(error =>
        reportError(sidebar, error)
      );
    }
  });

  panel.toolbar.addItem('crane-llm', button);

  // The switch also lives here, so it is discoverable without opening the
  // sidebar. Torn down with the panel to avoid orphaning it on document.body.
  const teardown = installSwitchPopover(button.node, settings);
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
    const settings: CraneSettings = {
      runinfo: new ToggleSetting('crane-llm:include-runinfo'),
      llm: new ToggleSetting('crane-llm:use-llm'),
      // Off by default: it stops cells from running, which nobody should
      // meet before choosing it.
      guard: new ToggleSetting('crane-llm:guard', false)
    };
    const sidebar = new CraneSidebar(settings);
    sidebar.id = SIDEBAR_ID;
    sidebar.title.label = t('sidebar.title');
    sidebar.title.caption = t('sidebar.caption');
    sidebar.title.closable = true;
    app.shell.add(sidebar, 'right');

    const execute = async () => {
      const panel = tracker.currentWidget;
      if (!panel) {
        await showErrorMessage(t('errors.dialog_title'), t('errors.open_notebook'));
        return;
      }

      // The command registry swallows rejections, so a failure triggered from
      // the palette has to be surfaced here.
      try {
        await runAnalysis(app, panel, sidebar, settings);
      } catch (error) {
        reportError(sidebar, error);
        await showErrorMessage(
          t('errors.dialog_title'),
          error instanceof Error ? error.message : String(error)
        );
      }
    };

    app.commands.addCommand(COMMAND_ID, {
      label: t('commands.run_label'),
      caption: t('commands.run_caption'),
      execute
    });

    app.commands.addCommand(TOGGLE_RUNINFO_COMMAND_ID, {
      label: t('commands.toggle_label'),
      caption: t('commands.toggle_caption'),
      isToggleable: true,
      isToggled: () => settings.runinfo.get(),
      execute: () => settings.runinfo.toggle()
    });

    app.commands.addCommand(TOGGLE_LLM_COMMAND_ID, {
      label: t('commands.toggle_llm_label'),
      caption: t('commands.toggle_llm_caption'),
      isToggleable: true,
      isToggled: () => settings.llm.get(),
      execute: () => settings.llm.toggle()
    });

    app.commands.addCommand(TOGGLE_GUARD_COMMAND_ID, {
      label: t('commands.toggle_guard_label'),
      caption: t('commands.toggle_guard_caption'),
      isToggleable: true,
      isToggled: () => settings.guard.get(),
      execute: () => settings.guard.toggle()
    });

    // Keep the palette's checkmark correct when the switch is changed from the
    // sidebar or the toolbar popover instead.
    settings.runinfo.subscribe(() => {
      app.commands.notifyCommandChanged(TOGGLE_RUNINFO_COMMAND_ID);
    });
    settings.llm.subscribe(() => {
      app.commands.notifyCommandChanged(TOGGLE_LLM_COMMAND_ID);
    });
    settings.guard.subscribe(() => {
      app.commands.notifyCommandChanged(TOGGLE_GUARD_COMMAND_ID);
    });

    if (palette) {
      palette.addItem({ command: COMMAND_ID, category: 'Notebook' });
      palette.addItem({ command: TOGGLE_RUNINFO_COMMAND_ID, category: 'Notebook' });
      palette.addItem({ command: TOGGLE_LLM_COMMAND_ID, category: 'Notebook' });
      palette.addItem({ command: TOGGLE_GUARD_COMMAND_ID, category: 'Notebook' });
    }

    tracker.widgetAdded.connect((_sender, panel) => {
      installToolbarButton(panel, app, sidebar, settings);
    });

    tracker.currentChanged.connect(() => {
      const panel = tracker.currentWidget;
      if (panel) {
        installToolbarButton(panel, app, sidebar, settings);
      }
    });

    tracker.forEach(panel => {
      installToolbarButton(panel, app, sidebar, settings);
    });
  }
};

export default plugin;
