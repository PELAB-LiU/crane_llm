import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { ICommandPalette, showErrorMessage, ToolbarButton } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';

const COMMAND_ID = 'crane-llm-jlab:run';
const INSTALLED_PANELS = new WeakSet<NotebookPanel>();
const RESPONSE_MANAGERS = new WeakMap<NotebookPanel, CraneResponseManager>();

type PredictionTone = 'blue' | 'green' | 'red' | 'gray';

interface ResponseEntry {
  container: HTMLDivElement;
  titleNode: HTMLDivElement;
  responseNode: HTMLPreElement;
  stale: boolean;
}

class CraneSidebar extends Widget {
  private statusNode: HTMLDivElement;
  private promptNode: HTMLPreElement;
  private responseNode: HTMLPreElement;

  constructor() {
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
    header.style.cssText = 'display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px;';

    const title = document.createElement('div');
    title.textContent = 'CRANE-LLM';
    title.style.cssText = 'font-size:14px;font-weight:700;';

    header.appendChild(title);

    this.statusNode = document.createElement('div');
    this.statusNode.style.cssText = 'margin-bottom:10px;color:var(--jp-info-color1);font-weight:600;';
    this.statusNode.textContent = 'idle';

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

  constructor(private panel: NotebookPanel) {
    this.attachNotebookSignals();
    this.attachSessionSignals();
    this.syncExecutionCounts();
    this.panel.disposed.connect(() => this.clearAll());
  }

  renderResponse(cell: any, response: string): void {
    this.removeEntry(cell.model);

    const entry = this.createEntry(cell, response);
    this.responseEntries.set(cell.model, entry);
    this.syncExecutionCounts();
  }

  private attachNotebookSignals(): void {
    const cells = this.panel.content.model?.cells;
    if (!cells) {
      return;
    }

    this.attachCellSignalsFromNotebook();

    if (typeof cells.changed?.connect === 'function') {
      cells.changed.connect(this.handleNotebookCellsChanged, this);
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

    const possibleSignals = [cell.stateChanged, cell.contentChanged, cell.sharedModel?.changed];
    for (const signal of possibleSignals) {
      if (signal && typeof signal.connect === 'function') {
        signal.connect(this.handleCellMaybeChanged, this);
      }
    }
  }

  private handleNotebookCellsChanged(): void {
    this.attachCellSignalsFromNotebook();
    this.syncExecutionCounts();
  }

  private handleCellMaybeChanged(): void {
    this.syncExecutionCounts();
  }

  private handleSessionStatusChanged(_sender: unknown, status: string): void {
    if (status === 'restarting' || status === 'dead') {
      this.clearAll();
    }
  }

  private handleKernelChanged(): void {
    const sessionContext = this.panel.sessionContext as any;
    const kernel = sessionContext?.session?.kernel;
    if (!kernel) {
      this.clearAll();
    }
  }

  clearResponseForCell(cell: any): void {
    this.removeEntry(cell?.model);
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

      const previousExecutionCount = this.executionCounts.get(cell);
      const currentExecutionCount = cell.executionCount;
      if (previousExecutionCount !== currentExecutionCount) {
        executedCellModels.push(cell);
      }
    }

    if (executedCellModels.length === 0) {
      this.refreshExecutionSnapshot();
      return;
    }

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
    const entry = this.responseEntries.get(cellModel);
    if (!entry) {
      return false;
    }

    entry.container.remove();
    this.responseEntries.delete(cellModel);
    return true;
  }

  private clearAll(): void {
    for (const entry of this.responseEntries.values()) {
      entry.container.remove();
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
    const titleNode = document.createElement('div');
    const header = document.createElement('div');
    const closeButton = document.createElement('button');
    const responseNode = document.createElement('pre');

    container.className = 'crane-llm-result';
    container.style.cssText = [
      'margin:8px 0 16px 0',
      'padding:10px 12px',
      'border-left:4px solid var(--jp-brand-color1)',
      'background:var(--jp-layout-color2)',
      'border-radius:0 8px 8px 0'
    ].join(';');

    header.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:6px;';

    titleNode.style.cssText = 'font-size:12px;font-weight:700;';

    closeButton.type = 'button';
    closeButton.textContent = 'Close';
    closeButton.className = 'jp-mod-styled jp-Button';
    closeButton.style.cssText = 'font-size:11px;padding:2px 8px;min-height:24px;';

    responseNode.textContent = response;
    responseNode.style.cssText = 'margin:0;white-space:pre-wrap;word-break:break-word;';

    closeButton.onclick = () => {
      this.removeEntry(cell.model);
    };

    header.appendChild(titleNode);
    header.appendChild(closeButton);
    container.appendChild(header);
    container.appendChild(responseNode);
    cell.node.insertAdjacentElement('afterend', container);

    const entry: ResponseEntry = {
      container,
      titleNode,
      responseNode,
      stale: false
    };

    this.applyEntryStyle(entry, response);
    return entry;
  }

  private applyEntryStyle(entry: ResponseEntry, responseText: string = entry.responseNode.textContent || ''): void {
    const tone = entry.stale ? 'gray' : getResponseTone(responseText);
    const borderColor = getToneColor(tone);
    entry.container.style.borderLeftColor = borderColor;
    entry.titleNode.textContent = entry.stale ? 'CRANE-LLM response (stale)' : 'CRANE-LLM response';
  }
}

function getToneColor(tone: PredictionTone): string {
  if (tone === 'red') {
    return '#dc2626';
  }
  if (tone === 'green') {
    return '#16a34a';
  }
  if (tone === 'gray') {
    return '#6b7280';
  }
  return 'var(--jp-brand-color1)';
}

function getResponseTone(responseText: string): PredictionTone {
  try {
    const parsed = JSON.parse(responseText);
    if (parsed && typeof parsed === 'object' && 'prediction' in parsed) {
      const prediction = (parsed as { prediction?: unknown }).prediction;
      if (prediction === true || prediction === 'true') {
        return 'red';
      }
      if (prediction === false || prediction === 'false') {
        return 'green';
      }
    }
  } catch (_error) {
    // Fall through to the default tone.
  }

  return 'blue';
}

function cellSource(cell: any): string {
  if (!cell || !cell.model) {
    return '';
  }

  const serialized = cell.model.toJSON?.();
  if (serialized && typeof serialized.source === 'string') {
    return serialized.source;
  }

  if (cell.model.sharedModel && typeof cell.model.sharedModel.getSource === 'function') {
    return cell.model.sharedModel.getSource();
  }

  if (cell.model.value && typeof cell.model.value.text === 'string') {
    return cell.model.value.text;
  }

  return '';
}

function requestKernelText(panel: NotebookPanel, code: string): Promise<string> {
  const kernel = panel.sessionContext.session?.kernel;
  if (!kernel) {
    return Promise.reject(new Error('No kernel is connected to this notebook.'));
  }

  return new Promise((resolve, reject) => {
    let streamText = '';
    const future = kernel.requestExecute({
      code,
      stop_on_error: true,
      store_history: false
    });

    future.onIOPub = message => {
      const msgType = message.header.msg_type;
      if (msgType === 'stream') {
        const content = message.content as { text?: string };
        streamText += content.text ?? '';
      } else if (msgType === 'error') {
        const content = message.content as { ename?: string; evalue?: string };
        reject(new Error(`${content.ename ?? 'Error'}: ${content.evalue ?? ''}`));
      }
    };

    future.done
      .then(() => resolve(streamText.trim()))
      .catch(error => reject(error));
  });
}

async function runAnalysis(app: JupyterFrontEnd, panel: NotebookPanel, sidebar: CraneSidebar): Promise<void> {
  const activeCell = panel.content.activeCell;
  if (!activeCell || activeCell.model.type !== 'code') {
    await showErrorMessage('CRANE-LLM', 'Select a code cell first.');
    return;
  }

  const responseManager = getResponseManager(panel);
  responseManager.clearResponseForCell(activeCell);

  const targetCellSource = cellSource(activeCell);

  sidebar.setStatus('building prompt...');
  sidebar.setResponse('');

  const prompt = await requestKernelText(
    panel,
    [
      'from nb_extension.api import get_prompt',
      `source = ${JSON.stringify(targetCellSource)}`,
      'print(get_prompt(source))'
    ].join('\n')
  );

  sidebar.setPrompt(prompt);

  sidebar.setStatus('calling LLM...');
  const response = await requestKernelText(
    panel,
    [
      'from nb_extension.api import run_prompt',
      `prompt = ${JSON.stringify(prompt)}`,
      'print(run_prompt(prompt))'
    ].join('\n')
  );

  sidebar.setStatus('done');
  sidebar.setResponse(response);
  responseManager.renderResponse(activeCell, response);
}

function installToolbarButton(panel: NotebookPanel, app: JupyterFrontEnd, sidebar: CraneSidebar): void {
  if (INSTALLED_PANELS.has(panel)) {
    return;
  }

  INSTALLED_PANELS.add(panel);
  getResponseManager(panel);

  panel.toolbar.addItem(
    'crane-llm',
    new ToolbarButton({
      label: 'CRANE-LLM',
      onClick: () => {
        void runAnalysis(app, panel, sidebar).catch(error => {
          sidebar.setStatus('error');
          sidebar.setResponse(error instanceof Error ? error.message : String(error));
        });
      }
    })
  );
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'crane-llm-jlab:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ICommandPalette],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker, palette: ICommandPalette | null) => {
    const sidebar = new CraneSidebar();
    sidebar.id = 'crane-llm-sidebar';
    sidebar.title.label = 'CRANE-LLM';
    sidebar.title.closable = true;
    app.shell.add(sidebar, 'right');

    const execute = async () => {
      const panel = tracker.currentWidget;
      if (!panel) {
        await showErrorMessage('CRANE-LLM', 'Open a notebook first.');
        return;
      }

      await runAnalysis(app, panel, sidebar);
    };

    app.commands.addCommand(COMMAND_ID, {
      label: 'Run CRANE-LLM',
      caption: 'Build the CRANE prompt from the active notebook cell and run the LLM',
      execute
    });

    if (palette) {
      palette.addItem({ command: COMMAND_ID, category: 'Notebook' });
    }

    tracker.widgetAdded.connect((_sender, panel) => {
      installToolbarButton(panel, app, sidebar);
    });

    tracker.currentChanged.connect(() => {
      const panel = tracker.currentWidget;
      if (panel) {
        installToolbarButton(panel, app, sidebar);
      }
    });

    tracker.forEach(panel => {
      installToolbarButton(panel, app, sidebar);
    });
  }
};

export default plugin;
