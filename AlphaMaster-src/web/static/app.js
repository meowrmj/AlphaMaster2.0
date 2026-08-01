const API = "";
let selectedDataFile = null;
let selectedSymbol = null;
let dataRootDir = "";
let selectedStrategyFile = null;
let selectedStrategySymbol = null;
let selectedBacktestDataFile = null;
let evalMode = localStorage.getItem("alphamaster_eval_mode") || "cpu_batch";
let algorithmMode = localStorage.getItem("alphamaster_algorithm_mode") || "rl";
let replayPolicy = localStorage.getItem("alphamaster_replay_policy") || "qd_incubation";
let replayConfig = null;
let searchConfig = null;
let replayPolicyPage = 0;
let chart = null;
let chartSymbol = null;
let chartZoom = { min: null, max: null };
let chartDrag = null;
let chartAutoFollow = true;
let chartZoomHandlersReady = false;
let trainingStartPending = false;
let trainingPendingAction = null;
let trainingRequestInFlight = false;
let trainingModeApplyInFlight = false;
let pollTimer = null;
let clientErrors = [];
let debugMode = false;
let lastDebugViewContent = "";

// 分页与回测状态
let currentPage = "train";
let btActive = false;
let btBuster = "";      // 图表缓存刷新键（用 job 时间戳）
let btPortfolioSig = ""; // 绩效卡签名：变化时才重建 + 播放数字动画，避免每次轮询重播
let lastEquityData = null; // 最近一次资金曲线数据，供绩效卡 sparkline 复用
let lastTrainingActive = false;
let lastTrainingStatus = null;
let btLastAlertKey = "";
let lastErrorPopupText = "";
let lastErrorPopupAt = 0;

const $ = (id) => document.getElementById(id);
const DEFAULT_CHART_WINDOW = 360;
const MIN_LAUNCH_PENDING_MS = 650;
const START_BTN_IDLE_TEXT = "开始训练";
const RETRAIN_BTN_TEXT = "重新训练";
const STOP_BTN_TEXT = "停止";

const EVAL_MODE_LABELS = {
  cpu_batch: "CPU batch",
  cuda_batch: "CUDA batch",
  legacy_cpu: "旧 CPU",
};

const REPLAY_POLICY_LABELS = {
  qd_incubation: "QD + 新方向孵化",
  qd: "QD 优秀池",
  incubation: "新方向孵化",
  none: "关闭回放",
};

const REPLAY_MODULES = [
  { key: "qd", label: "QD 优秀池" },
  { key: "incubation", label: "新方向孵化" },
];

const REPLAY_PAGE_LABELS = ["回放记忆", "搜索增强"];
const SEARCH_MODULES = [
  { key: "annealing", label: "退火" },
  { key: "genetic", label: "遗传" },
];

replayConfig = loadReplayConfig();
searchConfig = loadSearchConfig();

function getEvalMode() {
  const select = $("evalModeSelect");
  const value = select?.value || evalMode || "cpu_batch";
  if (!["cpu_batch", "cuda_batch", "legacy_cpu"].includes(value)) return "cpu_batch";
  return value;
}

function getAlgorithmMode() {
  const select = $("algorithmModeSelect");
  const value = select?.value || algorithmMode || "rl";
  if (!["rl", "ga", "hybrid"].includes(value)) return "rl";
  return value;
}

function algorithmQueryParam() {
  return `algorithm_mode=${encodeURIComponent(getAlgorithmMode())}`;
}

function algorithmModeLabel(mode) {
  return { rl: "强化学习 RL", ga: "独立遗传算法 GA", hybrid: "混合搜索 Hybrid" }[mode] || "强化学习 RL";
}

function syncEvalOptionsForAlgorithm() {
  const select = $("evalModeSelect");
  if (!select) return;
  const legacy = Array.from(select.options).find((option) => option.value === "legacy_cpu");
  const isGa = getAlgorithmMode() === "ga";
  if (legacy) legacy.disabled = isGa;
  if (isGa && select.value === "legacy_cpu") {
    select.value = "cpu_batch";
    evalMode = "cpu_batch";
    localStorage.setItem("alphamaster_eval_mode", evalMode);
  }
}

function scopedReplayConfigForAlgorithm() {
  if (getAlgorithmMode() === "ga") {
    return normalizeReplayConfig({ modules: { qd: false, incubation: false } });
  }
  return getReplayConfig();
}

function scopedSearchConfigForAlgorithm() {
  if (getAlgorithmMode() === "ga") {
    return normalizeSearchConfig({ modules: { annealing: false, genetic: false } });
  }
  return getSearchConfig();
}

function syncAlgorithmScopedControls() {
  syncEvalOptionsForAlgorithm();
  const mode = getAlgorithmMode();
  const isGa = mode === "ga";
  const replayPanel = $("replayPolicyPanel");
  const label = $("evalModeLabelText");
  if (replayPanel) {
    replayPanel.classList.toggle("hidden", isGa);
    if (isGa) replayPanel.open = false;
  }
  if (label) label.textContent = isGa ? "GA 评估模式" : "评估加速模式";
  updateReplayPolicySummary();
}

function evalModeLabel(mode) {
  return EVAL_MODE_LABELS[mode] || EVAL_MODE_LABELS.cpu_batch;
}

function replayModulesFromLegacy(policy) {
  const value = String(policy || "qd_incubation").toLowerCase();
  return {
    qd: value === "qd_incubation" || value === "qd",
    incubation: value === "qd_incubation" || value === "incubation",
  };
}

function replayPolicyFromModules(modules) {
  const qd = Boolean(modules?.qd);
  const incubation = Boolean(modules?.incubation);
  if (qd && incubation) return "qd_incubation";
  if (qd) return "qd";
  if (incubation) return "incubation";
  return "none";
}

function normalizeReplayConfig(value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const rawModules = value.modules && typeof value.modules === "object"
      ? value.modules
      : replayModulesFromLegacy(value.name || value.policy || replayPolicy);
    const modules = {};
    for (const mod of REPLAY_MODULES) modules[mod.key] = Boolean(rawModules[mod.key]);
    return { version: 1, modules };
  }
  return { version: 1, modules: replayModulesFromLegacy(value || replayPolicy) };
}

function loadReplayConfig() {
  const raw = localStorage.getItem("alphamaster_replay_config");
  if (raw) {
    try {
      return normalizeReplayConfig(JSON.parse(raw));
    } catch (_) {
      // Fall back to the legacy string below.
    }
  }
  return normalizeReplayConfig(localStorage.getItem("alphamaster_replay_policy") || "qd_incubation");
}

function saveReplayConfig(config) {
  replayConfig = normalizeReplayConfig(config);
  replayPolicy = replayPolicyFromModules(replayConfig.modules);
  localStorage.setItem("alphamaster_replay_config", JSON.stringify(replayConfig));
  localStorage.setItem("alphamaster_replay_policy", replayPolicy);
  updateReplayPolicySummary();
}

function normalizeSearchConfig(value) {
  const modules = {};
  const rawModules = value && typeof value === "object" && !Array.isArray(value) && value.modules
    ? value.modules
    : {};
  for (const mod of SEARCH_MODULES) modules[mod.key] = Boolean(rawModules[mod.key]);
  return { version: 1, modules };
}

function loadSearchConfig() {
  const raw = localStorage.getItem("alphamaster_search_config");
  if (raw) {
    try {
      return normalizeSearchConfig(JSON.parse(raw));
    } catch (_) {
      // Fall back to disabled modules.
    }
  }
  return normalizeSearchConfig(null);
}

function saveSearchConfig(config) {
  searchConfig = normalizeSearchConfig(config);
  localStorage.setItem("alphamaster_search_config", JSON.stringify(searchConfig));
  updateReplayPolicySummary();
}

function getSearchConfig() {
  const modules = {};
  for (const mod of SEARCH_MODULES) {
    const input = document.querySelector(`[data-search-module="${mod.key}"]`);
    modules[mod.key] = input ? Boolean(input.checked) : Boolean(searchConfig.modules?.[mod.key]);
  }
  return normalizeSearchConfig({ modules });
}

function searchPluginSummary(config = searchConfig) {
  const normalized = normalizeSearchConfig(config);
  const enabled = SEARCH_MODULES.filter((mod) => normalized.modules[mod.key]).map((mod) => mod.label);
  return enabled.length ? enabled.join(" + ") : "无搜索增强";
}

function getReplayConfig() {
  const modules = {};
  for (const mod of REPLAY_MODULES) {
    const input = document.querySelector(`[data-replay-module="${mod.key}"]`);
    modules[mod.key] = input ? Boolean(input.checked) : Boolean(replayConfig.modules?.[mod.key]);
  }
  return normalizeReplayConfig({ modules });
}

function getReplayPolicy() {
  return replayPolicyFromModules(getReplayConfig().modules);
}

function replayPolicyLabel(policy) {
  return REPLAY_POLICY_LABELS[policy] || REPLAY_POLICY_LABELS.qd_incubation;
}

function replayPolicyLabelFromConfig(config) {
  return replayPolicyLabel(replayPolicyFromModules(normalizeReplayConfig(config).modules));
}

function updateReplayPolicySummary() {
  const summary = $("replayPolicySummary");
  if (getAlgorithmMode() === "ga") {
    if (summary) summary.textContent = "GA 独立种群，不使用 RL 回放";
    return;
  }
  if (summary) summary.textContent = `${replayPolicyLabel(replayPolicy)} / ${searchPluginSummary()}`;
}

function renderReplayPolicyPage() {
  const pages = Array.from(document.querySelectorAll(".replay-policy-page"));
  if (!pages.length) return;
  replayPolicyPage = Math.max(0, Math.min(replayPolicyPage, pages.length - 1));
  pages.forEach((page, index) => page.classList.toggle("active", index === replayPolicyPage));
  const label = $("replayPolicyPageLabel");
  if (label) label.textContent = `${REPLAY_PAGE_LABELS[replayPolicyPage] || "搜索增强"} ${replayPolicyPage + 1}/${pages.length}`;
  const prev = $("replayPolicyPrevBtn");
  const next = $("replayPolicyNextBtn");
  if (prev) prev.disabled = replayPolicyPage <= 0;
  if (next) next.disabled = replayPolicyPage >= pages.length - 1;
}

function initEvalModeSelect() {
  const algorithmSelect = $("algorithmModeSelect");
  const select = $("evalModeSelect");
  const replayInputs = Array.from(document.querySelectorAll("[data-replay-module]"));
  const searchInputs = Array.from(document.querySelectorAll("[data-search-module]"));
  if (algorithmSelect) {
    algorithmSelect.value = ["rl", "ga", "hybrid"].includes(algorithmMode) ? algorithmMode : "rl";
    algorithmMode = getAlgorithmMode();
    localStorage.setItem("alphamaster_algorithm_mode", algorithmMode);
    algorithmSelect.addEventListener("change", () => {
      algorithmMode = getAlgorithmMode();
      localStorage.setItem("alphamaster_algorithm_mode", algorithmMode);
      syncAlgorithmScopedControls();
      handleAlgorithmModeSelectionChange();
    });
  }
  if (select) {
    select.value = ["cpu_batch", "cuda_batch", "legacy_cpu"].includes(evalMode) ? evalMode : "cpu_batch";
    syncAlgorithmScopedControls();
    evalMode = getEvalMode();
    localStorage.setItem("alphamaster_eval_mode", evalMode);
    select.addEventListener("change", () => {
      evalMode = getEvalMode();
      localStorage.setItem("alphamaster_eval_mode", evalMode);
      handleTrainingModeSelectionChange({ userInitiated: true });
    });
  }
  if (replayInputs.length) {
    replayConfig = normalizeReplayConfig(replayConfig);
    for (const input of replayInputs) {
      input.checked = Boolean(replayConfig.modules?.[input.dataset.replayModule]);
    }
    saveReplayConfig(getReplayConfig());
    const onReplayPolicyChange = () => {
      saveReplayConfig(getReplayConfig());
      handleTrainingModeSelectionChange({ userInitiated: true });
    };
    replayInputs.forEach((input) => input.addEventListener("change", onReplayPolicyChange));
    searchConfig = normalizeSearchConfig(searchConfig);
    for (const input of searchInputs) {
      input.checked = Boolean(searchConfig.modules?.[input.dataset.searchModule]);
    }
    saveSearchConfig(getSearchConfig());
    const onSearchConfigChange = () => {
      saveSearchConfig(getSearchConfig());
      handleTrainingModeSelectionChange({ userInitiated: true });
    };
    searchInputs.forEach((input) => input.addEventListener("change", onSearchConfigChange));
    $("replayPolicyPrevBtn")?.addEventListener("click", () => {
            replayPolicyPage -= 1;
            renderReplayPolicyPage();
    });
    $("replayPolicyNextBtn")?.addEventListener("click", () => {
      replayPolicyPage += 1;
      renderReplayPolicyPage();
    });
    renderReplayPolicyPage();
  }
  syncAlgorithmScopedControls();
}

const CPU_TRAINING_NOTE = `暂无报错

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【为什么用 CPU 训练，不用 GPU？】

你可以把 GPU 想象成一辆超大的货车，CPU 想象成一辆灵活的小电瓶车。

我们这个项目的训练，就像要做很多很多道「小题」：
每道题只算一点点数字，算完马上换下一道。
货车虽然一次能装很多，但每装卸一次都要准备很久才能再出发；
电瓶车一次装的少，但说走就走，一道接一道做，反而更快。

再打个比方：
GPU 像很多厨师一起做大锅饭，适合一次炒一大锅；
我们这个训练更像一道道菜分开炒，而且每道菜份量很小。
大锅饭团队每次开火、洗锅、集合都要时间，小菜一碟反而耽误在「准备」上。

所以具体原因是：
1. 每次要算的数据不多，GPU「启动一次计算」的等待，有时比真正算数还久。
2. 训练是一步接一步、一条公式接一条公式地指挥，GPU 经常闲着等下一道题，没法一直满负荷。
3. 数据还要在 CPU 和 GPU 之间来回搬运，也要花时间。

我们实测过（同样训练 50 步）：GPU 大约 4.5 秒一步，CPU 大约 1.9 秒一步。
这不是显卡坏了，也不是没装驱动，而是这个项目的做题方式，更适合 CPU。

说白了就是这个项目用CPU训练的速度比用GPU训练的速度更快`;

function emptyDebugMessage() {
  return debugMode ? "暂无日志" : CPU_TRAINING_NOTE;
}

function formatApiError(data, status, path) {
  const d = data?.detail;
  let detail = "";
  if (Array.isArray(d)) {
    detail = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  } else if (typeof d === "string") {
    detail = d;
  } else if (d) {
    detail = JSON.stringify(d);
  }
  if (data?.traceback) {
    detail += `\n\n${data.traceback}`;
  }
  return detail || `HTTP ${status} ${path}`;
}

async function logClientError(message, context = {}) {
  const entry = `[${new Date().toLocaleString()}] ${message}`;
  clientErrors.push(entry);
  if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
  renderDebugView();
  const silent = !!context.silent;
  if (!silent) {
    const detail = context.detail ? `${message}\n\n${context.detail}` : message;
    showErrorPopup("出错了", detail);
  }
  try {
    await fetch(API + "/api/debug/client-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ level: "error", message, context }),
    });
  } catch (_) {
    /* server may be down */
  }
}

function showErrorPopup(title, detail) {
  const modal = $("errorModal");
  const titleEl = $("errorModalTitle");
  const detailEl = $("errorModalDetail");
  if (!modal || !detailEl) {
    window.alert(`${title}\n\n${detail}`);
    return;
  }
  const text = String(detail || "").trim() || "未知错误";
  const now = Date.now();
  if (text === lastErrorPopupText && now - lastErrorPopupAt < 2500) return;
  lastErrorPopupText = text;
  lastErrorPopupAt = now;
  if (titleEl) titleEl.textContent = title || "出错了";
  detailEl.textContent = text;
  modal.hidden = false;
}

function closeErrorPopup() {
  const modal = $("errorModal");
  if (modal) modal.hidden = true;
}

async function copyErrorPopupDetail() {
  const text = $("errorModalDetail")?.textContent || "";
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

function isViewAtBottom(el, threshold = 40) {
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function renderDebugView(serverLines = [], errorLines = []) {
  const parts = [];
  if (clientErrors.length) {
    parts.push("=== 前端报错 ===", ...clientErrors);
  }
  if (errorLines.length) {
    parts.push("\n=== 服务端错误日志 (logs/web_errors.log) ===", ...errorLines);
  }
  if (debugMode && serverLines.length) {
    parts.push("\n=== 服务端运行日志 (logs/web_server.log) ===", ...serverLines);
  }
  const el = $("debugView");
  const atBottom = isViewAtBottom(el);
  const next = parts.length ? parts.join("\n") : emptyDebugMessage();
  const changed = next !== lastDebugViewContent;
  el.textContent = next;
  if (changed && atBottom && lastDebugViewContent) {
    el.scrollTop = el.scrollHeight;
  }
  lastDebugViewContent = next;
}

async function setDebugMode(enabled) {
  debugMode = !!enabled;
  $("debugModeCheck").checked = debugMode;
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ debug_mode: debugMode }),
    });
  } catch (e) {
    await logClientError("切换调试模式失败: " + e.message);
  }
  if (!debugMode) {
    renderDebugView([], []);
  } else {
    await refreshDebugLogs();
  }
}

async function refreshDebugLogs() {
  try {
    const data = await fetch(API + "/api/debug/logs?lines=120").then(async (res) => {
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(formatApiError(json, res.status, "/api/debug/logs"));
      return json;
    });
    $("debugLogPaths").textContent = `本地: ${data.error_log}`;
    renderDebugView(data.server_tail || [], data.error_tail || []);
  } catch (e) {
    renderDebugView();
  }
}

async function fetchJSON(path, opts = {}) {
  const silent = !!opts.silent;
  const maxRetries = opts.retries != null ? Number(opts.retries) : 5;
  const retryDelayMs = opts.retryDelayMs != null ? Number(opts.retryDelayMs) : 2000;
  const fetchOpts = { ...opts };
  delete fetchOpts.silent;
  delete fetchOpts.retries;
  delete fetchOpts.retryDelayMs;

  let lastNetworkMsg = null;
  for (let attempt = 1; attempt <= Math.max(1, maxRetries); attempt++) {
    let res;
    try {
      res = await fetch(API + path, fetchOpts);
    } catch (e) {
      lastNetworkMsg = `网络错误 ${path}: ${e.message}`;
      if (attempt < maxRetries) {
        await new Promise((r) => setTimeout(r, retryDelayMs));
        continue;
      }
      await logClientError(lastNetworkMsg, { path, silent, attempts: attempt });
      throw new Error(lastNetworkMsg);
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = formatApiError(data, res.status, path);
      await logClientError(`${path} -> ${msg}`, { path, status: res.status, silent });
      if (!silent) await refreshDebugLogs();
      throw new Error(msg);
    }
    return data;
  }
  throw new Error(lastNetworkMsg || `网络错误 ${path}`);
}

function formatScore(v) {
  if (v == null || Number.isNaN(v)) return "—";
  return Number(v).toFixed(4);
}

function waitForNextPaint() {
  return new Promise((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(resolve));
  });
}

async function waitForLaunchPendingMinimum(startedAt) {
  const elapsed = Date.now() - startedAt;
  const remaining = MIN_LAUNCH_PENDING_MS - elapsed;
  if (remaining > 0) {
    await new Promise((resolve) => setTimeout(resolve, remaining));
  }
}

function setTrainingActionPending(action) {
  const pending = Boolean(action);
  const startedAt = Date.now();
  trainingStartPending = Boolean(pending);
  trainingPendingAction = action || null;
  const launchPending = action === "start" || action === "retrain";
  const startBtn = $("startBtn");
  const retrainBtn = $("retrainBtn");
  const stopBtn = $("stopBtn");
  const pill = $("jobPill");
  if (startBtn) {
    startBtn.disabled = pending || !selectedDataFile;
    startBtn.textContent = launchPending ? "等待中" : START_BTN_IDLE_TEXT;
    startBtn.classList.toggle("is-pending", launchPending);
  }
  if (retrainBtn) {
    retrainBtn.disabled = !selectedDataFile;
    retrainBtn.textContent = RETRAIN_BTN_TEXT;
    retrainBtn.classList.remove("is-pending");
  }
  if (stopBtn) {
    stopBtn.disabled = action === "stop" || !(lastTrainingStatus?.active);
    stopBtn.textContent = action === "stop" ? "停止中" : STOP_BTN_TEXT;
    stopBtn.classList.toggle("is-pending", action === "stop");
  }
  if (pill && pending) {
    const pendingText = action === "stop" ? "停止中" : "等待中";
    pill.innerHTML = `<i class="pill-dot"></i>${pendingText}`;
    pill.className = "pill running";
  }
  const hint = $("logHint");
  if (hint && pending) {
    hint.textContent = action === "stop"
        ? "正在停止训练进程..."
        : "正在提交训练启动请求...";
  }
  return startedAt;
}

function renderDataFileCard(info) {
  const card = $("dataFileCard");
  const startBtn = $("startBtn");

  if (!info || !info.data_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择数据文件</div>';
    selectedDataFile = null;
    selectedSymbol = null;
    startBtn.disabled = true;
    if ($("retrainBtn")) $("retrainBtn").disabled = true;
    if ($("exportBtn")) $("exportBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  selectedDataFile = info.data_file;
  selectedSymbol = info.symbol || null;

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.data_file}</div>
    `;
    startBtn.disabled = true;
    if ($("retrainBtn")) $("retrainBtn").disabled = true;
    if ($("exportTrainingBtn")) $("exportTrainingBtn").disabled = true;
    if ($("importTrainingBtn")) $("importTrainingBtn").disabled = true;
    return;
  }

  card.className = "data-file-card valid";
  const yearsText = info.years_h1 != null ? `${info.years_h1} 年` : "—";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol}</span></div>
      <div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>
      <div class="item"><span class="label">K线</span><span class="value">${info.bars?.toLocaleString()}</span></div>
      <div class="item"><span class="label">数据年限</span><span class="value">${yearsText}</span></div>
      <div class="item"><span class="label">进度</span><span class="value" id="fileProgressPct">—</span></div>
      <div class="item"><span class="label">本次训练时长</span><span class="value" id="fileElapsedTime">—</span></div>
      <div class="item"><span class="label">历史训练总时长</span><span class="value" id="fileHistoryElapsedTime">—</span></div>
      <div class="item"><span class="label">保存冠军分</span><span class="value score-best" id="fileBestScore">—</span></div>
      <div class="item"><span class="label">当前候选分</span><span class="value score-val" id="fileValScore">—</span></div>
      <div class="item"><span class="label">未突破步数</span><span class="value" id="fileStagnationSteps">—</span></div>
    </div>
    <div class="path" title="${info.data_file}">${info.filename || info.data_file}</div>
  `;
  if (!trainingStartPending) {
    startBtn.disabled = false;
    if ($("retrainBtn")) $("retrainBtn").disabled = false;
  }
}

function updateBtStartBtn() {
  const startBtn = $("btStartBtn");
  if (!startBtn) return;
  startBtn.disabled = btActive || !selectedStrategyFile;
  ["btCommissionInput", "btSlippageInput"].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = btActive;
  });
}

function renderStrategyFileCard(info) {
  const card = $("btStrategyCard");
  if (!card) return;

  if (!info || !info.strategy_file) {
    card.className = "data-file-card";
    card.innerHTML = '<div class="data-file-empty">尚未选择策略文件</div>';
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    updateBtStartBtn();
    return;
  }

  if (info.valid === false) {
    card.className = "data-file-card invalid";
    card.innerHTML = `
      <div class="data-file-error">${info.message || "文件无效"}</div>
      <div class="data-file-path">${info.strategy_file}</div>
    `;
    selectedStrategyFile = null;
    selectedStrategySymbol = null;
    updateBtStartBtn();
    return;
  }

  selectedStrategyFile = info.strategy_file;
  selectedStrategySymbol = info.symbol || null;
  card.className = "data-file-card valid";
  const timeframeItem = info.timeframe
    ? `<div class="item"><span class="label">周期</span><span class="value">${info.timeframe}</span></div>`
    : "";
  const dataPath = info.data_file || "";
  const dataOk = info.data_file_exists;
  const dataHint = dataPath
    ? (dataOk ? dataPath : `（文件不存在）${dataPath}`)
    : "未记录数据路径 — 回测前请先在训练页选择同品种 Parquet";
  card.innerHTML = `
    <div class="data-file-row">
      <div class="item"><span class="label">品种</span><span class="value sym">${info.symbol || "—"}</span></div>
      ${timeframeItem}
      <div class="item"><span class="label">最优分数</span><span class="value score-best">${formatScore(info.best_score)}</span></div>
      <div class="item"><span class="label">词表版本</span><span class="value">${info.vocab_version || "—"}</span></div>
      <div class="item"><span class="label">公式长度</span><span class="value">${info.formula_decoded ? info.formula_decoded.split("→").length : "—"}</span></div>
    </div>
    <div class="path" title="${info.strategy_file}">策略: ${info.filename || info.strategy_file}</div>
    <div class="path ${dataPath && dataOk ? "" : "data-file-missing"}" title="${dataPath || ""}">数据: ${dataHint}</div>
  `;
  updateBtStartBtn();
}

function formatElapsed(startedAtIso, endAtIso) {
  if (!startedAtIso) return "—";
  const started = new Date(startedAtIso).getTime();
  if (Number.isNaN(started)) return "—";
  const end = endAtIso ? new Date(endAtIso).getTime() : Date.now();
  if (Number.isNaN(end)) return "—";
  return formatDurationSeconds(Math.max(0, Math.floor((end - started) / 1000)));
}

function formatDurationSeconds(secs) {
  if (secs == null || secs < 0) return "—";
  const total = Math.floor(secs);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  if (h > 0) return `${h}小时${m}分`;
  if (m > 0) return `${m}分钟`;
  return `${total}秒`;
}

function updateTrainingTimeFields(progress, training) {
  const sessionEl = $("fileElapsedTime");
  const historyEl = $("fileHistoryElapsedTime");
  if (!sessionEl && !historyEl) return;

  if (historyEl) {
    const hist = progress?.history_total_seconds;
    historyEl.textContent = hist != null ? formatDurationSeconds(hist) : "—";
  }

  const job = training?.job;
  const active = !!training?.active;
  if (!sessionEl) return;

  if (!job || job.state === "idle") {
    sessionEl.textContent = "—";
    return;
  }

  const elapsed = formatElapsed(job.started_at, active ? null : job.finished_at);
  sessionEl.textContent = active || elapsed === "—" ? elapsed : `${elapsed}（已停）`;
}

function updateFileProgress(progress) {
  const el = document.getElementById("fileProgressPct");
  if (el && progress) {
    el.textContent = `${progress.current_step} / ${progress.train_steps} (${progress.progress_pct}%)`;
  }
  const bestEl = document.getElementById("fileBestScore");
  if (bestEl) {
    bestEl.textContent = progress ? formatScore(progress.champion_score ?? progress.best_score) : "—";
  }
  const valEl = document.getElementById("fileValScore");
  if (valEl) {
    let val = progress?.candidate_val_score ?? progress?.val_score;
    if (val == null && progress?.history?.val_score?.length) {
      val = progress.history.val_score[progress.history.val_score.length - 1];
    }
    valEl.textContent = progress ? formatScore(val) : "—";
  }
  const stagnationEl = document.getElementById("fileStagnationSteps");
  if (stagnationEl) {
    const steps = progress?.stagnation_steps;
    stagnationEl.textContent = Number.isFinite(Number(steps)) ? Number(steps).toLocaleString() : "—";
  }
}

const CHART_SERIES = [
  { key: "best_score", label: "本轮最优分", borderColor: "#34f5c8", fillRGB: "52, 245, 200", yAxisID: "y" },
  { key: "new_candidate_best_val_score", label: "新候选最高分（不含精英）", borderColor: "#f97316", fillRGB: "249, 115, 22", yAxisID: "y", pointRadius: 1.4 },
  { key: "batch_best_val_score", label: "本批最高分（含精英）", borderColor: "#facc15", fillRGB: "250, 204, 21", yAxisID: "y", pointRadius: 1.2 },
  { key: "val_score", label: "当前候选分", borderColor: "#38bdf8", fillRGB: "56, 189, 248", yAxisID: "y" },
];

// 让曲线在填充区形成竖向渐变
function makeGradient(ctx, area, rgb, alpha = 0.16) {
  if (!area) return `rgba(${rgb}, 0.08)`;
  const g = ctx.createLinearGradient(0, area.top, 0, area.bottom);
  g.addColorStop(0, `rgba(${rgb}, ${alpha})`);
  g.addColorStop(0.6, `rgba(${rgb}, ${Math.max(alpha * 0.3, 0.03)})`);
  g.addColorStop(1, `rgba(${rgb}, 0)`);
  return g;
}

function finiteSeries(history, key) {
  const values = history?.[key];
  if (!Array.isArray(values)) return [];
  return values.map((v) => {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  });
}

function hasFiniteSeries(history, key) {
  return finiteSeries(history, key).some((v) => v !== null);
}

function sameFiniteSeries(a, b) {
  const x = Array.isArray(a) ? a : [];
  const y = Array.isArray(b) ? b : [];
  if (!x.length || x.length !== y.length) return false;
  for (let i = 0; i < x.length; i += 1) {
    const xv = Number(x[i]);
    const yv = Number(y[i]);
    if (!Number.isFinite(xv) && !Number.isFinite(yv)) continue;
    if (!Number.isFinite(xv) || !Number.isFinite(yv)) return false;
    if (Math.abs(xv - yv) > 1e-10) return false;
  }
  return true;
}

// 发光效果：在每条数据线绘制前设置对应颜色的柔和阴影
const glowPlugin = {
  id: "neonGlow",
  beforeDatasetDraw(chart, args) {
    const color = args?.meta?.dataset?.options?.borderColor;
    const blur = args?.meta?.dataset?.options?.glowBlur ?? 5;
    const ctx = chart.ctx;
    ctx.save();
    if (typeof color === "string" && blur > 0) {
      ctx.shadowColor = color;
      ctx.shadowBlur = blur;
    }
  },
  afterDatasetDraw(chart) {
    chart.ctx.restore();
  },
};
if (window.Chart) Chart.register(glowPlugin);

const CHART_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 450, easing: "easeOutQuart" },
  transitions: {
    active: { animation: { duration: 450, easing: "easeOutQuart" } },
  },
  plugins: {
    legend: {
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 16,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      backgroundColor: "rgba(8, 12, 20, 0.92)",
      borderColor: "rgba(94, 234, 212, 0.35)",
      borderWidth: 1,
      titleColor: "#e8edf4",
      bodyColor: "#a9bccf",
      titleFont: { family: "'JetBrains Mono'", size: 11 },
      bodyFont: { family: "'JetBrains Mono'", size: 11 },
      padding: 10,
      cornerRadius: 8,
      displayColors: true,
      usePointStyle: true,
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      title: { display: true, text: "分数（最优 / 验证）", color: "#7dd3fc", font: { family: "'DM Sans'", size: 10, weight: "600" } },
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

function buildChartDatasets(history) {
  const hiddenDuplicateKeys = new Set();
  if (sameFiniteSeries(history?.new_candidate_best_val_score, history?.batch_best_val_score)) {
    hiddenDuplicateKeys.add("new_candidate_best_val_score");
  }
  return CHART_SERIES
    .filter((s) => !hiddenDuplicateKeys.has(s.key) && hasFiniteSeries(history, s.key))
    .map((s) => {
    const isBatchBest = s.key === "batch_best_val_score";
    const isNewBest = s.key === "new_candidate_best_val_score";
    const fillAlpha = isBatchBest ? 0.06 : isNewBest ? 0.08 : 0.16;
    return {
      label: s.label,
      data: finiteSeries(history, s.key),
      borderColor: s.borderColor,
      borderWidth: 2,
      tension: 0.35,
      pointRadius: isBatchBest || isNewBest ? 0 : s.pointRadius ?? 0,
      pointHitRadius: 8,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: s.borderColor,
      pointHoverBorderColor: "#05070d",
      spanGaps: false,
      fill: false,
      glowBlur: isBatchBest ? 2 : isNewBest ? 3 : 5,
      backgroundColor: (context) => {
        const { ctx, chartArea } = context.chart;
        return makeGradient(ctx, chartArea, s.fillRGB, fillAlpha);
      },
      yAxisID: s.yAxisID,
    };
  });
}

function destroyChart() {
  if (chart) {
    chart.destroy();
    chart = null;
  }
  chartSymbol = null;
  chartZoom = { min: null, max: null };
  chartAutoFollow = true;
}

function createChart(ctx, steps, history) {
  return new Chart(ctx, {
    type: "line",
    data: { labels: steps, datasets: buildChartDatasets(history) },
    options: CHART_OPTIONS,
  });
}

function latestHistoryValue(history, key) {
  const arr = history?.[key];
  if (!Array.isArray(arr) || !arr.length) return null;
  for (let i = arr.length - 1; i >= 0; i -= 1) {
    const value = Number(arr[i]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function formatTimingMs(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 1 : 2)}s`;
  return `${Math.round(value)}ms`;
}

function renderTimingSummary(history) {
  const box = $("timingSummary");
  if (!box) return;
  const total = latestHistoryValue(history, "timing_total_ms");
  const values = {
    timingTotal: total,
    timingAB: latestHistoryValue(history, "timing_sample_elite_ms"),
    timingEval: latestHistoryValue(history, "timing_eval_ms"),
    timingGrad: latestHistoryValue(history, "timing_grad_ms"),
    timingRest: latestHistoryValue(history, "timing_rest_ms"),
  };
  box.hidden = !Number.isFinite(total);
  let visibleItems = 0;
  for (const [id, value] of Object.entries(values)) {
    const el = $(id);
    if (!el) continue;
    const visible = Number.isFinite(value);
    if (el.parentElement) el.parentElement.hidden = !visible;
    if (visible) visibleItems += 1;
    el.textContent = formatTimingMs(value);
  }
  box.hidden = visibleItems === 0;
}

function clampChartWindow(min, max, total) {
  if (!Number.isFinite(total) || total <= 0) return { min: null, max: null };
  const maxIndex = total - 1;
  let nextMin = Math.max(0, Math.min(maxIndex, Math.round(min)));
  let nextMax = Math.max(0, Math.min(maxIndex, Math.round(max)));
  if (nextMax < nextMin) [nextMin, nextMax] = [nextMax, nextMin];
  const minSpan = Math.min(maxIndex, 20);
  if (nextMax - nextMin < minSpan) {
    const mid = (nextMin + nextMax) / 2;
    nextMin = Math.max(0, Math.round(mid - minSpan / 2));
    nextMax = Math.min(maxIndex, nextMin + minSpan);
    nextMin = Math.max(0, nextMax - minSpan);
  }
  return { min: nextMin, max: nextMax };
}

function applyChartZoom(mode = "none") {
  if (!chart) return;
  const x = chart.options.scales.x;
  if (chartZoom.min == null || chartZoom.max == null) {
    delete x.min;
    delete x.max;
  } else {
    x.min = chartZoom.min;
    x.max = chartZoom.max;
  }
  chart.update(mode);
}

function resetChartZoom() {
  chartZoom = { min: null, max: null };
  chartAutoFollow = false;
  applyChartZoom("active");
}

function currentChartWindowSpan(total) {
  if (!Number.isFinite(total) || total <= 1) return DEFAULT_CHART_WINDOW - 1;
  const min = chartZoom.min ?? Math.max(0, total - DEFAULT_CHART_WINDOW);
  const max = chartZoom.max ?? total - 1;
  const span = max - min;
  return Number.isFinite(span) && span > 0 ? span : Math.min(DEFAULT_CHART_WINDOW - 1, total - 1);
}

function chartWindowTouchesLatest(total, tolerance = 2) {
  if (!Number.isFinite(total) || total <= 0) return true;
  if (chartZoom.min == null || chartZoom.max == null) return true;
  return chartZoom.max >= total - 1 - tolerance;
}

function followLatestChartWindow(total, spanOverride = null) {
  if (!Number.isFinite(total) || total <= DEFAULT_CHART_WINDOW) {
    chartZoom = { min: null, max: null };
    return;
  }
  const span = Number.isFinite(spanOverride)
    ? Math.max(2, Math.min(spanOverride, total - 1))
    : currentChartWindowSpan(total);
  chartZoom = {
    min: Math.max(0, total - 1 - span),
    max: total - 1,
  };
}

function resetChartToLatest() {
  chartAutoFollow = true;
  followLatestChartWindow(chart?.data?.labels?.length || 0);
  applyChartZoom("active");
}

function zoomChartAt(canvasX, factor) {
  if (!chart) return;
  const total = chart.data.labels.length;
  if (total < 3) return;
  const area = chart.chartArea;
  if (!area || canvasX < area.left || canvasX > area.right) return;
  const currentMin = chartZoom.min ?? 0;
  const currentMax = chartZoom.max ?? total - 1;
  const span = currentMax - currentMin;
  const ratio = (canvasX - area.left) / Math.max(1, area.right - area.left);
  const center = currentMin + span * ratio;
  const nextSpan = span * factor;
  const wasFollowingLatest = chartAutoFollow || chartWindowTouchesLatest(total);
  chartZoom = clampChartWindow(center - nextSpan * ratio, center + nextSpan * (1 - ratio), total);
  if (wasFollowingLatest && chartWindowTouchesLatest(total)) {
    chartAutoFollow = true;
    followLatestChartWindow(total, currentChartWindowSpan(total));
  } else {
    chartAutoFollow = false;
  }
  applyChartZoom("none");
}

function panChartByPixels(deltaX) {
  if (!chart || !chartDrag) return;
  const total = chart.data.labels.length;
  const area = chart.chartArea;
  if (!area || total < 3) return;
  const span = chartDrag.max - chartDrag.min;
  const pointsPerPixel = span / Math.max(1, area.right - area.left);
  const shift = -deltaX * pointsPerPixel;
  chartZoom = clampChartWindow(chartDrag.min + shift, chartDrag.max + shift, total);
  chartAutoFollow = chartWindowTouchesLatest(total);
  applyChartZoom("none");
}

function installChartZoomHandlers() {
  if (chartZoomHandlersReady) return;
  const canvas = $("mainChart");
  if (!canvas) return;
  chartZoomHandlersReady = true;
  canvas.addEventListener("wheel", (event) => {
    if (!chart) return;
    event.preventDefault();
    const rect = canvas.getBoundingClientRect();
    zoomChartAt(event.clientX - rect.left, event.deltaY < 0 ? 0.82 : 1.22);
  }, { passive: false });
  canvas.addEventListener("pointerdown", (event) => {
    if (!chart || event.button !== 0) return;
    const total = chart.data.labels.length;
    chartDrag = {
      startX: event.clientX,
      min: chartZoom.min ?? 0,
      max: chartZoom.max ?? total - 1,
    };
    canvas.setPointerCapture?.(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!chartDrag) return;
    panChartByPixels(event.clientX - chartDrag.startX);
  });
  const endDrag = (event) => {
    chartDrag = null;
    canvas.releasePointerCapture?.(event.pointerId);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("dblclick", resetChartToLatest);
}

function updateChartInPlace(steps, history) {
  const prevLen = chart.data.labels.length;
  chart.data.labels = steps;

  const next = buildChartDatasets(history);
  for (const ds of next) {
    const existing = chart.data.datasets.find((d) => d.label === ds.label);
    if (existing) {
      existing.data = ds.data;
    } else {
      chart.data.datasets.push(ds);
    }
  }

  const nextLabels = new Set(next.map((d) => d.label));
  chart.data.datasets = chart.data.datasets.filter((d) => nextLabels.has(d.label));

  const grew = steps.length > prevLen;
  if (chartAutoFollow && grew) followLatestChartWindow(steps.length, currentChartWindowSpan(prevLen || steps.length));
  applyChartZoom(grew ? "active" : "none");
}

function renderChart(history, label, progress) {
  const ctx = $("mainChart").getContext("2d");
  installChartZoomHandlers();
  const steps = history?.step || [];
  if (!steps.length) {
    destroyChart();
    renderTimingSummary(null);
    if (progress?.current_step > 0) {
      $("chartHint").textContent = `训练中 第 ${progress.current_step}/${progress.train_steps} 步，曲线每步更新`;
    } else {
      $("chartHint").textContent = "暂无历史数据（首步约需 15–30 秒）";
    }
    return;
  }

  const sameSymbol = chart && chartSymbol === label;
  if (sameSymbol) {
    updateChartInPlace(steps, history);
    renderTimingSummary(history);
  } else {
    destroyChart();
    chartAutoFollow = true;
    followLatestChartWindow(steps.length);
    chart = createChart(ctx, steps, history);
    chartSymbol = label;
    applyChartZoom("none");
    renderTimingSummary(history);
  }

  $("chartTitle").textContent = `${label} 训练曲线`;
  const windowText = chartZoom.min == null
    ? `${steps.length} 个记录点`
    : `${steps.length} 个记录点 · 显示 ${chartZoom.min + 1}-${chartZoom.max + 1}`;
  const followText = chartAutoFollow ? "跟随最新" : "查看历史";
  $("chartHint").textContent = `${windowText} · ${followText} · 滚轮缩放，拖动平移，双击回到最新`;
}

async function loadSymbolChart(symbol, progress) {
  if (!symbol) return;
  try {
    const timeframe = progress?.timeframe || null;
    const qs = timeframe
      ? `?timeframe=${encodeURIComponent(timeframe)}&${algorithmQueryParam()}`
      : `?${algorithmQueryParam()}`;
    const data = await fetchJSON(`/api/symbols/${encodeURIComponent(symbol)}${qs}`);
    const labelBase = data.timeframe ? `${symbol} ${data.timeframe}` : symbol;
    const label = `${labelBase} ${algorithmModeLabel(data.algorithm_mode || getAlgorithmMode())}`;
    renderChart(data.history, label, progress || data);
    $("formulaText").textContent = data.formula_decoded || "—";
  } catch (e) {
    $("formulaText").textContent = "—";
  }
}

function renderStrategies(rows) {
  const tbody = $("strategiesBody");
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="4">暂无已保存策略</td></tr>';
    return;
  }
  tbody.innerHTML = rows
    .map(
      (r) => `
    <tr>
      <td>${r.symbol}</td>
      <td>${r.timeframe || "—"}</td>
      <td>${formatScore(r.best_score)}</td>
      <td><code>${r.formula_decoded || "—"}</code></td>
    </tr>`
    )
    .join("");
}

function updateTrainingUI(training, progress) {
  lastTrainingStatus = training;
  const job = training?.job;
  const active = training?.active;
  if (active) {
    syncRunningTrainingControls(job);
  } else {
    syncAlgorithmScopedControls();
  }
  updateEvalModeApplyState(training);
  const pill = $("jobPill");
  const startBtn = $("startBtn");
  const retrainBtn = $("retrainBtn");
  const stopBtn = $("stopBtn");

  if (trainingStartPending && !trainingRequestInFlight) {
    setTrainingActionPending(null);
  }

  if (trainingStartPending) {
    updateTrainingTimeFields(progress, training);
    return;
  }

  if (!job || job.state === "idle") {
    pill.innerHTML = '<i class="pill-dot"></i>空闲';
    pill.className = "pill";
    startBtn.textContent = START_BTN_IDLE_TEXT;
    startBtn.disabled = !selectedDataFile;
    startBtn.classList.remove("is-pending");
    if (retrainBtn) {
      retrainBtn.textContent = RETRAIN_BTN_TEXT;
      retrainBtn.disabled = !selectedDataFile;
      retrainBtn.classList.remove("is-pending");
    }
    stopBtn.disabled = true;
    $("logHint").textContent = "—";
    updateTrainingTimeFields(progress, training);
    return;
  }

  const stateLabel = {
    running: "训练中",
    completed: "已完成",
    failed: "失败",
    stopped: "已停止",
  };
  const label = job.symbol ? `${job.symbol} ${job.timeframe || ""}`.trim() : "训练";
  const stateText = stateLabel[job.state] || job.state;
  pill.innerHTML = `<i class="pill-dot"></i>${stateText} · ${label}`;
  pill.className = "pill " + (job.state === "running" ? "running" : job.state);

  startBtn.textContent = active ? "训练中" : START_BTN_IDLE_TEXT;
  startBtn.disabled = active;
  startBtn.classList.remove("is-pending");
  if (retrainBtn) {
    retrainBtn.textContent = RETRAIN_BTN_TEXT;
    retrainBtn.disabled = !selectedDataFile;
    retrainBtn.classList.remove("is-pending");
  }
  stopBtn.disabled = !active;
  stopBtn.textContent = STOP_BTN_TEXT;
  stopBtn.classList.remove("is-pending");
  $("logHint").textContent = job.log_path || "—";
  updateTrainingTimeFields(progress, training);

  const logView = $("logView");
  const atBottom = isViewAtBottom(logView);
  logView.textContent = (training.log_tail || []).join("\n") || "等待输出…";
  if (atBottom) logView.scrollTop = logView.scrollHeight;
}

async function refreshOverview() {
  let overview = { data_file: null, progress: null };
  let strategies = { strategies: [] };
  let training = { active: false, job: null, log_tail: [] };

  try {
    overview = await fetchJSON(`/api/overview?${algorithmQueryParam()}`, { silent: true });
  } catch (_) {}

  try {
    strategies = await fetchJSON("/api/strategies", { silent: true });
  } catch (_) {}

  try {
    training = await fetchJSON("/api/training/status", { silent: true });
  } catch (_) {}

  if (overview.data_file) renderDataFileCard(overview.data_file);
  updateFileProgress(overview.progress);
  updateExportBtn(overview.progress, strategies.strategies);
  updateTrainingBtns(overview.progress, training);
  updateTrainingUI(training, overview.progress);
  renderStrategies(strategies.strategies);

  const sym = overview.progress?.symbol || selectedSymbol || training?.job?.symbol;
  const trainingActive = !!training?.active;
  if (lastTrainingActive && !trainingActive && sym) {
    await applyBestStrategyForBacktest(sym, null);
  }
  lastTrainingActive = trainingActive;

  if (sym && (training?.active || overview.progress)) {
    await loadSymbolChart(sym, overview.progress);
  }

  await refreshDebugLogs();
  refreshAiProviderStatus();
}

async function loadConfig() {
  const health = await fetch(API + "/api/health").then((r) => r.json()).catch(() => ({}));
  if (!health.version) {
    await logClientError(
      "后端版本过旧或未启动新版服务。请关闭旧进程后重新运行: python run_web.py",
      { health }
    );
  }

  const cfg = await fetchJSON("/api/config");
  debugMode = !!cfg.debug_mode;
  $("debugModeCheck").checked = debugMode;
  $("deviceMeta").textContent = `${cfg.train_steps} steps · batch ${cfg.batch_size} · ${cfg.device}`;
  if (cfg.error_log) {
    $("debugLogPaths").textContent = `本地: ${cfg.error_log}`;
  }
  dataRootDir = cfg.data_root_dir || "";
  if ($("dataRootInput")) $("dataRootInput").value = dataRootDir;
  if (cfg.data_file) renderDataFileCard(cfg.data_file);
  if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  applyBacktestCostDefaults(cfg);
  await initAiPanel(cfg);
}

function applyBacktestCostDefaults(cfg) {
  const cIn = $("btCommissionInput");
  const sIn = $("btSlippageInput");
  if (cIn && cfg.bt_commission_pct != null) cIn.value = Number(cfg.bt_commission_pct);
  if (sIn && cfg.bt_slippage_pct != null) sIn.value = Number(cfg.bt_slippage_pct);
  updateBtCostHint();
}

function readBacktestCosts() {
  const cRaw = Number($("btCommissionInput")?.value);
  const sRaw = Number($("btSlippageInput")?.value);
  const commission = Number.isFinite(cRaw) && cRaw >= 0 ? cRaw : 0.02;
  const slippage = Number.isFinite(sRaw) && sRaw >= 0 ? sRaw : 0.01;
  return { commission_pct: commission, slippage_pct: slippage };
}

function updateBtCostHint() {
  const hint = $("btCostSumHint");
  if (!hint) return;
  const { commission_pct, slippage_pct } = readBacktestCosts();
  const fee = Number((commission_pct + slippage_pct).toFixed(4));
  hint.textContent = `单边成本 ${fee}%`;
}

async function refreshAiProviderStatus() {
  try {
    const status = await fetchJSON("/api/ai/providers", { silent: true });
    window.__aiProviderStatus = status;
  } catch (_) {
    /* keep previous snapshot */
  }
  updateAiChannelHint();
}

async function initAiPanel(cfg) {
  const keyInput = $("aiApiKeyInput");
  if (!keyInput) return;

  if (cfg?.ai_api_key) keyInput.value = cfg.ai_api_key;
  else if (cfg?.ai_provider === "openclaw" || cfg?.ai_provider === "openclaw_wb") {
    keyInput.value = cfg.ai_provider;
  }

  await refreshAiProviderStatus();
  if (!keyInput.dataset.aiStatusBound) {
    keyInput.dataset.aiStatusBound = "1";
    keyInput.addEventListener("input", () => {
      updateAiChannelHint();
      refreshAiProviderStatus();
    });
  }
}

function resolveAiFromKey(raw) {
  const v = (raw || "").trim().toLowerCase();
  // openclaw_wb 必须先于 openclaw，避免前缀误匹配
  if (v === "openclaw_wb" || v.startsWith("openclaw_wb/")) {
    return { provider: "openclaw_wb", apiKey: raw.trim(), isAlias: true };
  }
  if (v === "openclaw" || v.startsWith("openclaw/")) {
    return { provider: "openclaw", apiKey: raw.trim(), isAlias: true };
  }
  return { provider: "deepseek", apiKey: (raw || "").trim(), isAlias: false };
}

function updateAiChannelHint() {
  const hint = $("aiChannelHint");
  const headHint = $("aiProviderHint");
  const keyInput = $("aiApiKeyInput");
  if (!hint || !keyInput) return;

  const resolved = resolveAiFromKey(keyInput.value);
  const status = window.__aiProviderStatus;
  const row = (status?.providers || []).find((p) => p.id === resolved.provider);

  if (resolved.provider === "deepseek") {
    if (headHint) headHint.textContent = "DeepSeek · deepseek-v4-flash";
    hint.textContent = "当前：DeepSeek（deepseek-v4-flash · https://api.deepseek.com）。";
  } else if (resolved.provider === "openclaw") {
    if (headHint) {
      headHint.textContent = row?.available ? "openclaw (QClaw) · 已匹配" : "openclaw (QClaw) · 未就绪";
    }
    hint.textContent = row?.hint || "已匹配 openclaw：将自动使用本地 QClaw token。";
  } else {
    if (headHint) headHint.textContent = row?.available ? "openclaw_wb · 已匹配" : "openclaw_wb · 未就绪";
    hint.textContent = row?.hint || "已匹配 openclaw_wb：将自动使用 WorkBuddy token。";
  }
}

function openUnlimitedModal() {
  const modal = $("aiUnlimitedModal");
  if (modal) modal.hidden = false;
}

function closeUnlimitedModal() {
  const modal = $("aiUnlimitedModal");
  if (modal) modal.hidden = true;
}

async function runAiAnalyze() {
  const btn = $("aiAnalyzeBtn");
  const view = $("aiAnswerView");
  const rawKey = $("aiApiKeyInput")?.value || "";
  const resolved = resolveAiFromKey(rawKey);
  if (!view) return;

  if (resolved.provider === "deepseek" && !resolved.apiKey) {
    view.className = "ai-answer error";
    view.textContent = "请填写 DeepSeek API Key";
    return;
  }

  await refreshAiProviderStatus();

  if (btn) btn.disabled = true;
  view.className = "ai-answer loading";
  view.textContent = `正在通过 ${resolved.provider} 连接并流式分析…`;

  let header = "";
  let answer = "";

  try {
    const res = await fetch(API + "/api/ai/analyze-training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: resolved.provider,
        api_key: resolved.apiKey,
        symbol: selectedSymbol || null,
      }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, "/api/ai/analyze-training"));
    }
    if (!res.body) throw new Error("浏览器不支持流式响应");

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    view.className = "ai-answer streaming";
    view.textContent = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";
      for (const block of chunks) {
        const line = block
          .split("\n")
          .map((l) => l.trim())
          .find((l) => l.startsWith("data:"));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch (_) {
          continue;
        }
        if (event.type === "meta") {
          header =
            `[${event.label || event.provider || resolved.provider} · ${event.model || ""} · ${event.symbol || ""}${event.timeframe ? " " + event.timeframe : ""}]` +
            (event.prior_count
              ? ` · 已带入前 ${event.prior_count} 次同品种同周期分析`
              : " · 首次分析") +
            `\n\n`;
          view.textContent = header;
          view.scrollTop = view.scrollHeight;
        } else if (event.type === "delta") {
          answer += event.text || "";
          view.textContent = header + answer;
          view.scrollTop = view.scrollHeight;
        } else if (event.type === "error") {
          throw new Error(event.message || "分析失败");
        } else if (event.type === "done") {
          answer = event.answer || answer;
          view.className = "ai-answer";
          view.textContent = header + (answer || "（无内容）");
        }
      }
    }
    if (!answer && view.className.includes("streaming")) {
      throw new Error("流式分析中断，未收到完整回复");
    }
    view.className = "ai-answer";
  } catch (e) {
    view.className = "ai-answer error";
    view.textContent = `分析失败: ${e.message}`;
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function browseStrategyFile() {
  await loadBacktestOptions();
}

async function loadBacktestOptions() {
  let data;
  try {
    const sym = selectedStrategySymbol || selectedSymbol;
    const url = sym ? `/api/backtest/options?symbol=${encodeURIComponent(sym)}` : "/api/backtest/options";
    data = await fetchJSON(url, { silent: true });
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
    return;
  }
  const stratSel = $("btStrategySelect");
  if (stratSel) {
    const opts = ['<option value="">— 选择回测策略 —</option>'];
    (data.strategies || []).forEach((s) => {
      opts.push(`<option value="${escHtml(s.strategy_file)}" data-symbol="${escHtml(s.symbol || "")}" data-timeframe="${escHtml(s.timeframe || "")}">${escHtml(s.label || s.strategy_file)}</option>`);
    });
    const prev = stratSel.value || selectedStrategyFile || data.last_strategy_file || "";
    stratSel.innerHTML = opts.join("");
    if (prev && [...stratSel.options].some((o) => o.value === prev)) stratSel.value = prev;
    onBacktestStrategySelect();
  }
  const dataSel = $("btDataFileSelect");
  if (dataSel) {
    const opts = ['<option value="">— 选择回测数据源 —</option>'];
    (data.data_files || []).forEach((d) => {
      const bars = d.bars != null ? ` · ${Number(d.bars).toLocaleString()}K` : "";
      opts.push(`<option value="${escHtml(d.data_file)}" data-symbol="${escHtml(d.symbol || "")}" data-timeframe="${escHtml(d.timeframe || "")}">${escHtml(d.symbol || "?")} ${escHtml(d.timeframe || "")} · ${escHtml(d.filename || d.relative_path || d.data_file)}${bars}</option>`);
    });
    const prev = dataSel.value || selectedBacktestDataFile || data.last_data_file || selectedDataFile || "";
    dataSel.innerHTML = opts.join("");
    if (prev && [...dataSel.options].some((o) => o.value === prev)) dataSel.value = prev;
    onBacktestDataFileSelect();
  }
}

function onBacktestStrategySelect() {
  const sel = $("btStrategySelect");
  if (!sel) return;
  const opt = sel.options[sel.selectedIndex];
  selectedStrategyFile = sel.value || null;
  selectedStrategySymbol = opt?.dataset?.symbol || null;
  const card = $("btStrategyCard");
  if (card) {
    card.className = selectedStrategyFile ? "data-file-card valid" : "data-file-card";
    card.innerHTML = selectedStrategyFile
      ? `<div class="data-file-row"><div class="item"><span class="label">策略</span><span class="value sym">${escHtml(selectedStrategySymbol || "—")}</span></div><div class="item"><span class="label">周期</span><span class="value">${escHtml(opt?.dataset?.timeframe || "—")}</span></div></div><div class="path" title="${escHtml(selectedStrategyFile)}">${escHtml(opt?.textContent || selectedStrategyFile)}</div>`
      : '<div class="data-file-empty">尚未选择策略</div>';
  }
  updateBtStartBtn();
}

function onBacktestDataFileSelect() {
  const sel = $("btDataFileSelect");
  selectedBacktestDataFile = sel?.value || null;
}

async function applyBestStrategyForBacktest(symbol, strategyFile) {
  if (strategyFile) {
    renderStrategyFileCard(strategyFile);
    return;
  }
  if (!symbol) return;
  try {
    const res = await fetchJSON(
      `/api/strategy-file/sync-best?symbol=${encodeURIComponent(symbol)}`,
      { method: "POST" }
    );
    renderStrategyFileCard(res);
  } catch (_) {
    await loadBacktestStrategyContext();
  }
}

async function loadBacktestStrategyContext() {
  await loadBacktestOptions();
  return;
  const sym = selectedStrategySymbol || selectedSymbol;
  if (sym) {
    await applyBestStrategyForBacktest(sym, null);
    return;
  }
  try {
    const cfg = await fetchJSON("/api/config");
    if (cfg.strategy_file) renderStrategyFileCard(cfg.strategy_file);
  } catch (_) {
    /* ignore */
  }
}

async function saveDataRootDir() {
  const input = $("dataRootInput");
  const value = (input?.value || "").trim();
  try {
    const saved = await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_root_dir: value }),
    });
    dataRootDir = saved.data_root_dir || "";
    if (input) input.value = dataRootDir;
    selectedBacktestDataFile = null;
    await loadBacktestOptions();
  } catch (e) {
    await logClientError("保存数据源文件夹失败: " + e.message);
  }
}

async function browseDataRootDir() {
  try {
    const res = await fetchJSON("/api/data-root/browse", { method: "POST" });
    if (res.cancelled) return;
    dataRootDir = res.data_root_dir || "";
    if ($("dataRootInput")) $("dataRootInput").value = dataRootDir;
    selectedBacktestDataFile = null;
    await loadBacktestOptions();
  } catch (e) {
    await logClientError("选择数据源文件夹失败: " + e.message);
  }
}

async function browseDataFile() {
  try {
    const res = await fetchJSON("/api/data-file/browse", { method: "POST" });
    if (res.cancelled) return;
    renderDataFileCard(res);
    selectedSymbol = res.symbol;
    if (res.stopped_training?.ok) {
      await logClientError(
        `已停止旧训练：${res.stopped_training.previous_symbol || "—"} ` +
        `${res.stopped_training.previous_timeframe || ""}，因为切换了数据文件`,
        { silent: true }
      );
    }
    await loadSymbolChart(res.symbol);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function applyEvalModeToCurrentTraining() {
  const mode = getEvalMode();
  const algorithm = getAlgorithmMode();
  const replayPayload = scopedReplayConfigForAlgorithm();
  const searchPayload = scopedSearchConfigForAlgorithm();
  const replay = replayPolicyFromModules(replayPayload.modules);
  const job = lastTrainingStatus?.job;
  const current = job?.eval_mode;
  const currentReplay = job?.replay_config
    ? replayPolicyFromModules(normalizeReplayConfig(job.replay_config).modules)
    : (job?.replay_policy || "qd_incubation");
  if (!lastTrainingStatus?.active) {
    if (!job?.data_file) {
      updateEvalModeApplyState(lastTrainingStatus);
      return;
    }
    try {
      const res = await fetchJSON("/api/training/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data_file: job.data_file, from_scratch: false, algorithm_mode: algorithm, eval_mode: mode, replay_policy: replayPayload, search_plugins: searchPayload }),
      });
      selectedSymbol = res.data_file?.symbol || res.job?.symbol || selectedSymbol;
      renderDataFileCard(res.data_file);
      await refreshOverview();
    } catch (e) {
      $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    return;
  }
  if ((job?.algorithm_mode || "rl") !== algorithm) {
    await stopTraining();
    return;
  }
  if (current === mode && currentReplay === replay && (job?.algorithm_mode || "rl") === algorithm) {
    updateEvalModeApplyState(lastTrainingStatus);
    return;
  }
  const ok = window.confirm(
    `要把当前训练从 ${algorithmModeLabel(job?.algorithm_mode || "rl")} / ${evalModeLabel(current)} / ${replayPolicyLabel(currentReplay)} / ${searchPluginSummary(job?.search_config)} 切到 ${algorithmModeLabel(algorithm)} / ${evalModeLabel(mode)} / ${replayPolicyLabel(replayPolicyFromModules(replayPayload.modules))} / ${searchPluginSummary(searchPayload)} 吗？\n\n` +
      "程序会先停止当前训练进程，再用同一个数据文件从检查点继续训练。"
  );
  if (!ok) return;
  try {
    const res = await fetchJSON("/api/training/apply-eval-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm_mode: algorithm, eval_mode: mode, replay_policy: replayPayload, search_plugins: searchPayload }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol || selectedSymbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function handleAlgorithmModeSelectionChange() {
  updateEvalModeApplyState(lastTrainingStatus);
  const job = lastTrainingStatus?.job;
  const activeAlgorithm = job?.algorithm_mode || "rl";
  if (lastTrainingStatus?.active && job && activeAlgorithm !== getAlgorithmMode()) {
    await stopTraining();
    return;
  }
  await refreshOverview();
}

function updateEvalModeApplyState(training) {
  const hint = $("evalModeHint");
  if (!hint) return;
  const selectedMode = getEvalMode();
  const selectedAlgorithm = getAlgorithmMode();
  const selectedReplayConfig = scopedReplayConfigForAlgorithm();
  const selectedSearchConfig = scopedSearchConfigForAlgorithm();
  const selectedReplay = replayPolicyFromModules(selectedReplayConfig.modules);
  const selectedSearch = searchPluginSummary(selectedSearchConfig);
  const job = training?.job || null;
  const active = Boolean(training?.active && job);
  const activeAlgorithm = job?.algorithm_mode || "rl";
  const activeMode = job?.eval_mode || "cpu_batch";
  const activeReplay = job?.replay_config
    ? replayPolicyFromModules(normalizeReplayConfig(job.replay_config).modules)
    : (job?.replay_policy || "qd_incubation");
  const activeSearch = searchPluginSummary(job?.search_config);
  const changed = active && (
    selectedAlgorithm !== activeAlgorithm ||
    selectedMode !== activeMode ||
    selectedReplay !== activeReplay ||
    JSON.stringify(selectedSearchConfig.modules) !== JSON.stringify(normalizeSearchConfig(job?.search_config).modules)
  );
  const resumable = Boolean(!active && job?.data_file);

  if (trainingModeApplyInFlight) {
    hint.textContent = `正在应用：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。会先保存当前节点，再按新设置续跑。`;
    return;
  }

  if (active) {
    if (selectedAlgorithm !== activeAlgorithm) {
      hint.textContent = `当前正在运行：${algorithmModeLabel(activeAlgorithm)} / ${evalModeLabel(activeMode)}。你正在查看：${algorithmModeLabel(selectedAlgorithm)}。切换算法会先停止当前训练，不会自动启动新算法；停稳后手动点击“开始训练”继续所选算法。`;
      return;
    }
    hint.textContent = changed
      ? `当前训练：${algorithmModeLabel(activeAlgorithm)} / ${evalModeLabel(activeMode)} / ${replayPolicyLabel(activeReplay)} / ${activeSearch}；已选择：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。同一算法内切换会自动保存当前节点，再按新设置续跑。`
      : `当前训练已使用：${algorithmModeLabel(activeAlgorithm)} / ${evalModeLabel(activeMode)} / ${replayPolicyLabel(activeReplay)} / ${activeSearch}。`;
    return;
  }

  if (job?.data_file) {
    hint.textContent = `当前没有运行训练；已选择：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。下次点“开始训练”才会按这个设置续跑。`;
    return;
  }

  hint.textContent = selectedAlgorithm === "ga"
    ? `当前未运行；下次启动：独立遗传算法 GA / ${evalModeLabel(selectedMode)}。GA 使用独立种群和独立历史，不使用 RL 的精英回放插件。`
    : `当前未运行；下次启动：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。`;
  return;

  if (trainingModeApplyInFlight) {
    hint.textContent = `正在应用：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。会先保存当前节点，再按新设置续跑。`;
    return;
  }

  if (active) {
    hint.textContent = changed
      ? `当前训练：${algorithmModeLabel(activeAlgorithm)} / ${evalModeLabel(activeMode)} / ${replayPolicyLabel(activeReplay)} / ${activeSearch}；已选择：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。切换后会自动保存当前节点，再按新设置续跑。`
      : `当前训练已使用：${algorithmModeLabel(activeAlgorithm)} / ${evalModeLabel(activeMode)} / ${replayPolicyLabel(activeReplay)} / ${activeSearch}。`;
    return;
  }

  if (job?.data_file) {
    hint.textContent = `当前没有运行训练；已选择：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。下次点“开始训练”才会按这个设置续跑。`;
    return;
  }

  hint.textContent = selectedAlgorithm === "ga"
    ? `当前未运行；下次启动：独立遗传算法 GA / ${evalModeLabel(selectedMode)}。GA 使用独立种群和独立历史，不使用 RL 的精英回放插件。`
    : `当前未运行；下次启动：${algorithmModeLabel(selectedAlgorithm)} / ${evalModeLabel(selectedMode)} / ${replayPolicyLabel(selectedReplay)} / ${selectedSearch}。`;
}

function restoreControlsFromTrainingJob(job) {
  if (!job) return;
  const alg = job.algorithm_mode || "rl";
  const mode = job.eval_mode || "cpu_batch";
  const replay = normalizeReplayConfig(job.replay_config || { modules: { qd: true, incubation: true } });
  const search = normalizeSearchConfig(job.search_config || { modules: { annealing: false, genetic: false } });
  const algSelect = $("algorithmModeSelect");
  const evalSelect = $("evalModeSelect");
  if (algSelect) algSelect.value = alg;
  if (evalSelect) evalSelect.value = mode;
  algorithmMode = getAlgorithmMode();
  evalMode = getEvalMode();
  localStorage.setItem("alphamaster_algorithm_mode", algorithmMode);
  localStorage.setItem("alphamaster_eval_mode", evalMode);
  saveReplayConfig(replay);
  saveSearchConfig(search);
  for (const input of Array.from(document.querySelectorAll("[data-replay-module]"))) {
    input.checked = Boolean(replay.modules?.[input.dataset.replayModule]);
  }
  for (const input of Array.from(document.querySelectorAll("[data-search-module]"))) {
    input.checked = Boolean(search.modules?.[input.dataset.searchModule]);
  }
  syncAlgorithmScopedControls();
  updateEvalModeApplyState(lastTrainingStatus);
}

function syncRunningTrainingControls(job) {
  if (!job || trainingModeApplyInFlight || trainingStartPending) return;
  const alg = job.algorithm_mode || "rl";
  const mode = job.eval_mode || "cpu_batch";
  const algSelect = $("algorithmModeSelect");
  const evalSelect = $("evalModeSelect");
  if (algSelect && algSelect.value !== alg) {
    updateEvalModeApplyState(lastTrainingStatus);
    return;
  }
  if (evalSelect && evalSelect.value !== mode) {
    evalSelect.value = mode;
    evalMode = getEvalMode();
    localStorage.setItem("alphamaster_eval_mode", evalMode);
  }
  const replay = normalizeReplayConfig(job.replay_config || job.replay_policy || { modules: { qd: true, incubation: true } });
  const search = normalizeSearchConfig(job.search_config || { modules: { annealing: false, genetic: false } });
  replayConfig = replay;
  replayPolicy = replayPolicyFromModules(replay.modules);
  searchConfig = search;
  localStorage.setItem("alphamaster_replay_config", JSON.stringify(replayConfig));
  localStorage.setItem("alphamaster_replay_policy", replayPolicy);
  localStorage.setItem("alphamaster_search_config", JSON.stringify(searchConfig));
  for (const input of Array.from(document.querySelectorAll("[data-replay-module]"))) {
    input.checked = Boolean(replay.modules?.[input.dataset.replayModule]);
  }
  for (const input of Array.from(document.querySelectorAll("[data-search-module]"))) {
    input.checked = Boolean(search.modules?.[input.dataset.searchModule]);
  }
  syncAlgorithmScopedControls();
}

async function handleTrainingModeSelectionChange(options = {}) {
  updateEvalModeApplyState(lastTrainingStatus);
  if (!lastTrainingStatus?.active) {
    await refreshOverview();
    return;
  }
  if (!options.userInitiated || trainingStartPending || trainingModeApplyInFlight) return;
  if ((lastTrainingStatus?.job?.algorithm_mode || "rl") !== getAlgorithmMode()) return;
  await applySelectedModeToActiveTraining();
}

async function applySelectedModeToActiveTraining() {
  if (trainingModeApplyInFlight) return;
  const mode = getEvalMode();
  const algorithm = getAlgorithmMode();
  const replayPayload = scopedReplayConfigForAlgorithm();
  const searchPayload = scopedSearchConfigForAlgorithm();
  const job = lastTrainingStatus?.job;
  if (!lastTrainingStatus?.active || !job) return;
  if ((job?.algorithm_mode || "rl") !== algorithm) {
    updateEvalModeApplyState(lastTrainingStatus);
    return;
  }

  const currentReplay = job?.replay_config
    ? replayPolicyFromModules(normalizeReplayConfig(job.replay_config).modules)
    : (job?.replay_policy || "qd_incubation");
  const currentSearch = JSON.stringify(normalizeSearchConfig(job?.search_config).modules);
  const nextReplay = replayPolicyFromModules(replayPayload.modules);
  const nextSearch = JSON.stringify(searchPayload.modules);
  if ((job?.algorithm_mode || "rl") === algorithm && job?.eval_mode === mode && currentReplay === nextReplay && currentSearch === nextSearch) {
    updateEvalModeApplyState(lastTrainingStatus);
    return;
  }

  trainingModeApplyInFlight = true;
  updateEvalModeApplyState(lastTrainingStatus);
  try {
    const res = await fetchJSON("/api/training/apply-eval-mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ algorithm_mode: algorithm, eval_mode: mode, replay_policy: replayPayload, search_plugins: searchPayload }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol || selectedSymbol;
    renderDataFileCard(res.data_file);
    await refreshOverview();
    if (res.job) {
      updateTrainingUI({ active: true, job: res.job, log_tail: [] }, null);
    }
  } catch (e) {
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    trainingModeApplyInFlight = false;
    updateEvalModeApplyState(lastTrainingStatus);
  }
}

async function startTraining() {
  if (trainingRequestInFlight || lastTrainingStatus?.active) return;
  trainingRequestInFlight = true;
  if (!selectedDataFile) {
    trainingRequestInFlight = false;
    await logClientError("请先选择数据文件");
    return;
  }
  const pendingAt = setTrainingActionPending("start");
  try {
    await waitForNextPaint();
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: selectedDataFile, from_scratch: false, algorithm_mode: getAlgorithmMode(), eval_mode: getEvalMode(), replay_policy: scopedReplayConfigForAlgorithm(), search_plugins: scopedSearchConfigForAlgorithm() }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await waitForLaunchPendingMinimum(pendingAt);
    setTrainingActionPending(null);
    if (res.job) {
      updateTrainingUI({ active: true, job: res.job, log_tail: [] }, null);
    }
    await refreshOverview();
  } catch (e) {
    await waitForLaunchPendingMinimum(pendingAt);
    setTrainingActionPending(null);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    trainingRequestInFlight = false;
    if (trainingStartPending) setTrainingActionPending(null);
  }
}

async function waitForTrainingInactive(timeoutMs = 20000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const status = await fetchJSON("/api/training/status", { silent: true });
    lastTrainingStatus = status;
    if (!status?.active) return status;
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error("等待当前训练停止超时");
}

async function retrainFromScratch() {
  if (trainingRequestInFlight) return;
  const activeJob = lastTrainingStatus?.active ? lastTrainingStatus.job : null;
  const dataFile = activeJob?.data_file || selectedDataFile;
  if (!dataFile) {
    await logClientError("请先选择数据文件");
    return;
  }
  const ok = window.confirm(
    (activeJob ? "当前训练会先停止，然后重新训练。\n" : "") +
      "重新训练会清除该品种的检查点，从第 0 步重新搜索。\n" +
      "已有的更优策略会保留，只有挖到更高分才会覆盖。\n\n" +
      "确定要重新训练吗？"
  );
  if (!ok) return;
  trainingRequestInFlight = true;
  const pendingAt = setTrainingActionPending("retrain");
  try {
    await waitForNextPaint();
    if (activeJob) {
      await fetchJSON("/api/training/stop", { method: "POST" });
      await waitForTrainingInactive();
    }
    const res = await fetchJSON("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data_file: dataFile, from_scratch: true, algorithm_mode: getAlgorithmMode(), eval_mode: getEvalMode(), replay_policy: scopedReplayConfigForAlgorithm(), search_plugins: scopedSearchConfigForAlgorithm() }),
    });
    selectedSymbol = res.data_file?.symbol || res.job?.symbol;
    renderDataFileCard(res.data_file);
    await waitForLaunchPendingMinimum(pendingAt);
    setTrainingActionPending(null);
    if (res.job) {
      updateTrainingUI({ active: true, job: res.job, log_tail: [] }, null);
    }
    await refreshOverview();
  } catch (e) {
    await waitForLaunchPendingMinimum(pendingAt);
    setTrainingActionPending(null);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    trainingRequestInFlight = false;
    if (trainingStartPending) setTrainingActionPending(null);
  }
}

function updateExportBtn(progress, strategies) {
  const sym = progress?.symbol || selectedSymbol;
  const hasStrategy = progress?.has_strategy || (strategies || []).some((s) => s.symbol === sym);
  const btn = $("exportBtn");
  if (btn) btn.disabled = !sym || !hasStrategy;
}

function updateTrainingBtns(progress, training) {
  if (trainingStartPending) return;
  const sym = progress?.symbol || selectedSymbol;
  const active = training?.active;
  const hasCheckpoint = Boolean(progress?.has_checkpoint);
  const exportBtn = $("exportTrainingBtn");
  const importBtn = $("importTrainingBtn");

  let exportTitle = "打包 checkpoint、训练曲线与策略为 zip";
  if (!sym) {
    exportTitle = "请先选择数据文件";
  } else if (active) {
    exportTitle = "训练进行中，请停止后再导出";
  } else if (!hasCheckpoint) {
    exportTitle = "该品种尚无检查点：至少训练满 20 步后才会生成（每 20 步保存一次）";
  }

  if (exportBtn) {
    exportBtn.disabled = !sym || !hasCheckpoint || !!active;
    exportBtn.title = exportTitle;
  }
  if (importBtn) {
    importBtn.disabled = !sym || !!active;
    importBtn.title = active ? "训练进行中，请停止后再导入" : "上传 .zip 或 .pt，下次训练断点续训";
  }
}

async function exportTraining() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/training/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const disp = res.headers.get("Content-Disposition") || "";
    const m = /filename="([^"]+)"/.exec(disp);
    const filename = m ? m[1] : `training_${sym.replace(/\./g, "_")}.zip`;
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function triggerImportTraining() {
  const input = $("importTrainingFile");
  if (input) {
    input.value = "";
    input.click();
  }
}

async function handleImportTrainingFile(event) {
  const input = event.target;
  const file = input.files?.[0];
  if (!file) return;

  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch(`${API}/api/training/import?symbol=${encodeURIComponent(sym)}`, {
      method: "POST",
      body: form,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(formatApiError(data, res.status, "/api/training/import"));
    }
    if (data.symbol && data.symbol !== sym) {
      selectedSymbol = data.symbol;
    }
    clientErrors.push(`[${new Date().toLocaleString()}] ${data.message || "训练文件导入成功"}`);
    if (clientErrors.length > 80) clientErrors = clientErrors.slice(-80);
    renderDebugView();
    await refreshOverview();
  } catch (e) {
    await logClientError(`导入训练失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    input.value = "";
  }
}

function parseContentDispositionFilename(header) {
  if (!header) return null;
  const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(header);
  if (utf8) return decodeURIComponent(utf8[1]);
  const plain = /filename="([^"]+)"/i.exec(header) || /filename=([^;]+)/i.exec(header);
  return plain ? plain[1].trim() : null;
}

async function exportStrategy() {
  const sym = selectedSymbol;
  if (!sym) {
    await logClientError("请先选择数据文件");
    return;
  }
  const path = `/api/strategies/${encodeURIComponent(sym)}/export`;
  try {
    const res = await fetch(API + path);
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(formatApiError(data, res.status, path));
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download =
      parseContentDispositionFilename(res.headers.get("Content-Disposition")) ||
      `strategy_${sym.replace(/\./g, "_")}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (e) {
    await logClientError(`导出策略失败: ${e.message}`);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

async function stopTraining() {
  if (trainingRequestInFlight || !lastTrainingStatus?.active) return;
  trainingRequestInFlight = true;
  setTrainingActionPending("stop");
  try {
    const res = await fetchJSON("/api/training/stop", { method: "POST" });
    setTrainingActionPending(null);
    await refreshOverview();
    const sym = res.training?.job?.symbol || selectedSymbol;
    await applyBestStrategyForBacktest(sym, res.strategy_file);
  } catch (e) {
    setTrainingActionPending(null);
    $("debugView").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } finally {
    trainingRequestInFlight = false;
    if (trainingStartPending) setTrainingActionPending(null);
  }
}

// ═══════════════════════════════════════════════════════════════════
// 分页切换
// ═══════════════════════════════════════════════════════════════════
function switchPage(page) {
  if (page !== "train" && page !== "backtest" && page !== "realtime") return;
  currentPage = page;
  document.querySelectorAll(".stepper .step").forEach((s) => {
    s.classList.toggle("active", s.dataset.page === page);
  });
  document.querySelectorAll(".page").forEach((p) => {
    p.classList.toggle("active", p.id === `page-${page}`);
  });
  if (page === "backtest") {
    loadBacktestStrategyContext();
    refreshBacktest();
  } else if (page === "realtime") {
    initRealtimeOnce();
    refreshRealtime();
  }
}

// ═══════════════════════════════════════════════════════════════════
// 回测：格式化辅助
// ═══════════════════════════════════════════════════════════════════
function fmtPct(v, digits = 2) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits) + "%";
}
function fmtSigned(v, digits = 3) {
  if (v == null || Number.isNaN(v)) return "—";
  return (v >= 0 ? "+" : "") + Number(v).toFixed(digits);
}

// ═══════════════════════════════════════════════════════════════════
// 回测：状态轮询 + UI 更新
// ═══════════════════════════════════════════════════════════════════
async function refreshBacktest() {
  let st;
  try {
    st = await fetchJSON("/api/backtest/status", { silent: true });
  } catch (_) {
    return;
  }
  btActive = !!st.active;
  const job = st.job;
  const state = job?.state || "idle";

  // 按钮
  const stopBtn = $("btStopBtn");
  updateBtStartBtn();
  if (stopBtn) stopBtn.disabled = !btActive;

  // 缓存刷新键：用最近一次任务的结束/开始时间
  btBuster = job?.finished_at || job?.started_at || btBuster;

  // 日志
  const logView = $("btLogView");
  const logText = (st.log_tail || []).join("\n") || "等待任务…";
  if (logView) {
    const atBottom = isViewAtBottom(logView);
    logView.textContent = logText;
    if (atBottom) logView.scrollTop = logView.scrollHeight;
  }
  if ($("btLogHint")) $("btLogHint").textContent = job?.log_path || "—";

  // 阶段进度条
  updateBacktestPhase(st, state);

  if (state === "failed") {
    const alertKey = `${job?.log_path || ""}|${job?.finished_at || ""}|${job?.exit_code ?? ""}`;
    if (alertKey && alertKey !== btLastAlertKey) {
      btLastAlertKey = alertKey;
      const errLine = job?.error ? `\n错误: ${job.error}` : "";
      showErrorPopup(
        "回测失败",
        `退出码: ${job?.exit_code ?? "?"}${errLine}\n日志: ${job?.log_path || "—"}\n\n${logText}`
      );
    }
  }

  // 结果报告（非运行态时刷新，运行态保留上次结果）
  if (!btActive) {
    await refreshBacktestReport();
  }
}

const BT_STATE_LABEL = {
  running: "回测中",
  completed: "已完成",
  failed: "失败",
  stopped: "已停止",
  idle: "待机",
};

function updateBacktestPhase(st, state) {
  const fill = $("btPhaseFill");
  const label = $("btPhaseLabel");
  if (!fill || !label) return;

  const total = st.phase_total || 7;
  const idx = st.phase_index || 0;

  let pct;
  if (btActive) {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = `${st.phase_label || "回测中"}…`;
    fill.classList.add("animate");
  } else if (state === "completed") {
    pct = 100;
    label.textContent = "完成";
    fill.classList.remove("animate");
  } else if (state === "failed" || state === "stopped") {
    pct = Math.min(96, Math.round(((idx + 1) / total) * 100));
    label.textContent = BT_STATE_LABEL[state];
    fill.classList.remove("animate");
  } else {
    pct = 0;
    label.textContent = "待机";
    fill.classList.remove("animate");
  }
  fill.style.width = pct + "%";
}

async function refreshBacktestReport() {
  let data;
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/report?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/report";
  try {
    data = await fetchJSON(url, { silent: true });
  } catch (_) {
    return;
  }
  if (!data.available || !data.report) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = "尚未运行回测";
    lastEquityData = null;
    btPortfolioSig = "";
    renderEquity(null);
    return;
  }
  // 先取资金曲线（写入 lastEquityData），再渲染绩效卡，让 sparkline 用上真实数据
  await refreshEquityCurve();
  renderPortfolio(data.report);
  renderBacktestTable(data.report.symbols || {});
}

async function refreshEquityCurve() {
  const sym = selectedStrategySymbol || selectedSymbol;
  const url = sym
    ? `/api/backtest/equity?symbol=${encodeURIComponent(sym)}`
    : "/api/backtest/equity";
  try {
    const data = await fetchJSON(url, { silent: true });
    lastEquityData = data?.available ? data.data : null;
    renderEquity(data);
  } catch (_) {
    lastEquityData = null;
    renderEquity(null);
  }
}

// ═══════════════════════════════════════════════════════════════════
// 迷你 sparkline + 数字滚动动画（终端仪表盘质感）
// ═══════════════════════════════════════════════════════════════════
const METRIC_FMT = {
  pct: (v) => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%",
  signed: (v) => (v >= 0 ? "+" : "") + v.toFixed(3),
  ratio: (v) => v.toFixed(3),
  int: (v) => Math.round(v).toLocaleString(),
  winrate: (v) => (v * 100).toFixed(1) + "%",
  strength: (v) => Math.round(v * 100) + "%",
};

function prefersReducedMotion() {
  return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
}

// 短促 count-up（≈420ms, easeOutCubic），克制不浮夸
function animateCount(el, to, fmt) {
  const fn = METRIC_FMT[fmt] || ((v) => String(v));
  if (!Number.isFinite(to)) {
    el.textContent = "—";
    return;
  }
  if (prefersReducedMotion()) {
    el.textContent = fn(to);
    return;
  }
  const dur = 420;
  const t0 = performance.now();
  function frame(now) {
    const p = Math.min(1, (now - t0) / dur);
    const e = 1 - Math.pow(1 - p, 3); // easeOutCubic
    el.textContent = fn(to * e);
    if (p < 1) requestAnimationFrame(frame);
    else el.textContent = fn(to);
  }
  requestAnimationFrame(frame);
}

function runCountUp(root) {
  if (!root) return;
  root.querySelectorAll("[data-count]").forEach((el) => {
    animateCount(el, parseFloat(el.dataset.count), el.dataset.fmt || "");
  });
}

// 均匀降采样为 <= target 个有限点
function downsampleSeries(arr, target) {
  const clean = (arr || [])
    .map(Number)
    .filter((v) => Number.isFinite(v));
  if (clean.length <= target) return clean;
  const out = [];
  const step = (clean.length - 1) / (target - 1);
  for (let i = 0; i < target; i++) out.push(clean[Math.round(i * step)]);
  return out;
}

// 生成极小趋势微线（内联 SVG，轻量、清晰）
function sparklineSVG(values, { color = "#5eead4", fillRGB = null, w = 74, h = 22 } = {}) {
  const v = downsampleSeries(values, 56);
  if (v.length < 2) return "";
  const min = Math.min(...v);
  const max = Math.max(...v);
  const range = max - min || 1;
  const n = v.length;
  const x = (i) => (i / (n - 1)) * w;
  const y = (val) => h - 2 - ((val - min) / range) * (h - 4);
  const line = "M" + v.map((val, i) => `${x(i).toFixed(1)} ${y(val).toFixed(1)}`).join(" L ");
  const area = fillRGB
    ? `<path d="${line} L ${w} ${h} L 0 ${h} Z" fill="rgba(${fillRGB},0.14)" stroke="none"/>`
    : "";
  const dot = `<circle cx="${x(n - 1).toFixed(1)}" cy="${y(v[n - 1]).toFixed(1)}" r="1.6" fill="${color}"/>`;
  return `<svg class="spark-svg" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">${area}<path d="${line}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linejoin="round" stroke-linecap="round"/>${dot}</svg>`;
}

// 取当前主资金曲线序列（组合优先，否则第一个品种）
function mainEquitySeries() {
  const d = lastEquityData;
  if (!d) return null;
  if (d.portfolio) return d.portfolio;
  const syms = d.symbols || {};
  const names = Object.keys(syms);
  return names.length ? syms[names[0]] : null;
}

function renderPortfolio(report) {
  const grid = $("btPortfolioGrid");
  if (!grid) return;
  const p = report.portfolio || {};
  const focus = report.focus_symbol || Object.keys(report.symbols || {})[0] || "";
  const symData = focus ? (report.symbols || {})[focus] : null;

  if (!Object.keys(p).length) {
    grid.innerHTML = '<div class="metric-empty">回测结果无绩效数据</div>';
    btPortfolioSig = "";
    return;
  }

  const plNum = Number(symData?.profit_loss_ratio ?? p.profit_loss_ratio);
  const nTrades = symData?.n_trades ?? p.n_trades;
  const winRate = symData?.win_rate;

  // sparkline 数据源：主资金曲线 + 滚动夏普
  const eq = mainEquitySeries();
  const posColor = p.total_return >= 0 ? "#4ade80" : "#f87171";
  const posRGB = p.total_return >= 0 ? "74, 222, 128" : "248, 113, 113";
  const equitySpark = eq ? sparklineSVG(eq.equity, { color: posColor, fillRGB: posRGB }) : "";
  const rollSpark = eq ? sparklineSVG(eq.rolling_sharpe, { color: "#5eead4", fillRGB: "94, 234, 212" }) : "";

  const cards = [
    { label: "总收益", raw: p.total_return, fmt: "pct", cls: p.total_return >= 0 ? "pos" : "neg", spark: equitySpark },
    { label: "Sharpe", raw: p.sharpe, fmt: "signed", cls: "accent", spark: rollSpark },
    { label: "Sortino", raw: p.sortino, fmt: "signed", cls: "accent", spark: rollSpark },
    { label: "盈亏比", raw: Number.isFinite(plNum) ? plNum : null, fmt: "ratio", cls: Number.isFinite(plNum) ? "accent" : "" },
    { label: "交易数", raw: Number.isFinite(Number(nTrades)) ? Number(nTrades) : null, fmt: "int", cls: "" },
    { label: "胜率", raw: winRate != null ? Number(winRate) : null, fmt: "winrate", cls: "" },
  ];

  // 签名守卫：数值/焦点/资金曲线未变则不重建，避免每次轮询重播动画
  const sig = [focus, btEquitySig, ...cards.map((c) => c.raw)].join("|");
  if (sig === btPortfolioSig) {
    if ($("btPortfolioHint")) $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
    return;
  }
  btPortfolioSig = sig;

  grid.innerHTML = cards
    .map((c) => {
      const cardCls = c.cls === "pos" || c.cls === "neg" ? c.cls : "";
      const finite = c.raw != null && Number.isFinite(c.raw);
      const finalText = finite ? METRIC_FMT[c.fmt](c.raw) : "—";
      const countAttr = finite ? ` data-count="${c.raw}" data-fmt="${c.fmt}"` : "";
      const spark = c.spark ? `<div class="metric-spark">${c.spark}</div>` : "";
      return `
    <div class="metric-card ${cardCls}">
      <div class="metric-label">${c.label}</div>
      <div class="metric-value ${c.cls}"${countAttr}>${finalText}</div>
      ${spark}
    </div>`;
    })
    .join("");

  runCountUp(grid);

  if ($("btPortfolioHint")) {
    $("btPortfolioHint").textContent = focus ? `${focus} 回测绩效` : "回测绩效";
  }
}

function renderBacktestTable(symbols) {
  const tbody = $("btTableBody");
  if (!tbody) return;
  const rows = Object.entries(symbols);
  if (!rows.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="7">暂无回测结果</td></tr>';
    if ($("btTableHint")) $("btTableHint").textContent = "—";
    return;
  }
  if ($("btTableHint")) $("btTableHint").textContent = rows.length === 1 ? rows[0][0] : `${rows.length} 个品种`;
  tbody.innerHTML = rows
    .map(([sym, d]) => {
      const retCls = (d.total_return || 0) >= 0 ? "pos" : "neg";
      const shCls = (d.sharpe || 0) >= 0 ? "pos" : "neg";
      return `
      <tr>
        <td class="sym-cell">${sym}</td>
        <td class="${retCls}">${fmtPct(d.total_return)}</td>
        <td class="${shCls}">${fmtSigned(d.sharpe)}</td>
        <td>${fmtSigned(d.sortino)}</td>
        <td>${Number.isFinite(Number(d.profit_loss_ratio)) ? Number(d.profit_loss_ratio).toFixed(3) : "—"}</td>
        <td>${d.n_trades ?? "—"}</td>
        <td>${d.win_rate != null ? (d.win_rate * 100).toFixed(1) + "%" : "—"}</td>
      </tr>`;
    })
    .join("");
}

// ═══════════════════════════════════════════════════════════════════
// 交互式资金曲线（HTML / Chart.js）
// ═══════════════════════════════════════════════════════════════════
let equityChart = null;
let rollingChart = null;
let btEquitySig = "";

const EQUITY_COLORS = [
  { hex: "#5eead4", rgb: "94, 234, 212" },
  { hex: "#38bdf8", rgb: "56, 189, 248" },
  { hex: "#818cf8", rgb: "129, 140, 248" },
  { hex: "#fbbf24", rgb: "251, 191, 36" },
  { hex: "#f472b6", rgb: "244, 114, 182" },
  { hex: "#a3e635", rgb: "163, 230, 53" },
];

function verticalGradient(chart, rgb, topAlpha, bottomAlpha) {
  const { ctx, chartArea } = chart;
  if (!chartArea) return `rgba(${rgb}, ${topAlpha})`;
  const g = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
  g.addColorStop(0, `rgba(${rgb}, ${topAlpha})`);
  g.addColorStop(0.62, `rgba(${rgb}, ${(topAlpha + bottomAlpha) / 4})`);
  g.addColorStop(1, `rgba(${rgb}, ${bottomAlpha})`);
  return g;
}

const EQUITY_TOOLTIP = {
  backgroundColor: "rgba(8, 12, 20, 0.94)",
  borderColor: "rgba(94, 234, 212, 0.35)",
  borderWidth: 1,
  titleColor: "#e8edf4",
  bodyColor: "#a9bccf",
  titleFont: { family: "'JetBrains Mono'", size: 11 },
  bodyFont: { family: "'JetBrains Mono'", size: 11 },
  padding: 10,
  cornerRadius: 8,
  usePointStyle: true,
};

const EQUITY_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 500, easing: "easeOutQuart" },
  plugins: {
    legend: {
      display: true,
      labels: {
        color: "#a9bccf",
        usePointStyle: true,
        pointStyle: "circle",
        boxWidth: 8,
        boxHeight: 8,
        padding: 14,
        font: { family: "'DM Sans'", size: 12, weight: "600" },
      },
    },
    tooltip: {
      ...EQUITY_TOOLTIP,
      callbacks: {
        label: (c) => ` ${c.dataset.label}: ${Number(c.parsed.y).toFixed(4)}`,
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
  },
};

const ROLLING_OPTIONS = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: { mode: "index", intersect: false },
  animation: { duration: 500, easing: "easeOutQuart" },
  spanGaps: false,
  plugins: {
    legend: { display: false },
    tooltip: {
      ...EQUITY_TOOLTIP,
      borderColor: "rgba(251, 191, 36, 0.4)",
      callbacks: {
        label: (c) => {
          const v = c.parsed.y;
          if (v == null || Number.isNaN(v)) return " 滚动夏普: —";
          return ` 滚动夏普: ${Number(v).toFixed(3)}`;
        },
      },
    },
  },
  scales: {
    x: {
      ticks: { color: "#6b7d92", maxTicksLimit: 8, maxRotation: 0, font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(120,190,235,0.05)" },
      border: { color: "rgba(120,190,235,0.12)" },
    },
    y: {
      ticks: { color: "#6b7d92", font: { family: "'JetBrains Mono'", size: 10 } },
      grid: { color: "rgba(251,191,36,0.06)" },
      border: { color: "rgba(251,191,36,0.18)" },
    },
  },
};

function destroyEquityCharts() {
  if (equityChart) { equityChart.destroy(); equityChart = null; }
  if (rollingChart) { rollingChart.destroy(); rollingChart = null; }
}

function renderEquityStats(name, series) {
  const el = $("btEquityStats");
  if (!el) return;
  const pl = series.profit_loss_ratio;
  const plText = Number.isFinite(Number(pl)) ? Number(pl).toFixed(3) : "—";
  const roll = series.rolling_sharpe || [];
  let lastRoll = null;
  for (let i = roll.length - 1; i >= 0; i--) {
    const v = Number(roll[i]);
    if (Number.isFinite(v)) {
      lastRoll = v;
      break;
    }
  }
  const plNum = Number(pl);
  const cards = [
    { label: "总收益", raw: series.total_return, fmt: "pct", cls: series.total_return >= 0 ? "pos" : "neg" },
    { label: "夏普", raw: series.sharpe, fmt: "signed", cls: "accent" },
    { label: "索提诺", raw: series.sortino, fmt: "signed", cls: "accent" },
    { label: "盈亏比", raw: Number.isFinite(plNum) ? plNum : null, fmt: "ratio", cls: "accent" },
    {
      label: "最新滚动夏普",
      raw: lastRoll,
      fmt: "signed",
      cls: lastRoll == null ? "" : lastRoll >= 0 ? "accent" : "neg",
    },
  ];
  el.innerHTML =
    `<span class="equity-stat-name">${name}</span>` +
    cards
      .map((c) => {
        const finite = c.raw != null && Number.isFinite(c.raw);
        const finalText = finite ? METRIC_FMT[c.fmt](c.raw) : "—";
        const countAttr = finite ? ` data-count="${c.raw}" data-fmt="${c.fmt}"` : "";
        return `
      <div class="equity-stat">
        <span class="equity-stat-label">${c.label}</span>
        <span class="equity-stat-value ${c.cls}"${countAttr}>${finalText}</span>
      </div>`;
      })
      .join("");
  runCountUp(el);
}

function buildEquityChart(labels, symbols, portfolio) {
  const canvas = $("btEquityChart");
  if (!canvas) return;
  const symNames = Object.keys(symbols);
  const multi = symNames.length > 1;
  const datasets = symNames.map((s, i) => {
    const col = EQUITY_COLORS[i % EQUITY_COLORS.length];
    return {
      label: s,
      data: symbols[s].equity,
      borderColor: col.hex,
      borderWidth: multi ? 1.5 : 2.2,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: col.hex,
      pointHoverBorderColor: "#05070d",
      fill: !multi,
      backgroundColor: (ctx) => verticalGradient(ctx.chart, col.rgb, 0.3, 0),
    };
  });
  if (portfolio) {
    datasets.push({
      label: "等权组合",
      data: portfolio.equity,
      borderColor: "#e8edf4",
      borderWidth: 2.4,
      tension: 0.25,
      pointRadius: 0,
      pointHoverRadius: 4,
      pointHoverBackgroundColor: "#e8edf4",
      pointHoverBorderColor: "#05070d",
      fill: true,
      backgroundColor: (ctx) => verticalGradient(ctx.chart, "232, 237, 244", 0.16, 0),
    });
  }
  if (equityChart) equityChart.destroy();
  equityChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: EQUITY_OPTIONS,
  });
}

function buildRollingChart(labels, series, windowBars) {
  const canvas = $("btRollingChart");
  if (!canvas) return;
  if (rollingChart) rollingChart.destroy();
  const data = series.rolling_sharpe || [];
  const labelEl = $("btRollingLabel");
  if (labelEl) {
    labelEl.textContent = windowBars
      ? `滚动夏普 · ${windowBars} bars`
      : "滚动夏普 · Rolling Sharpe";
  }
  rollingChart = new Chart(canvas.getContext("2d"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "滚动夏普",
          data,
          borderColor: "#fbbf24",
          borderWidth: 1.5,
          tension: 0.2,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: "#fbbf24",
          pointHoverBorderColor: "#05070d",
          spanGaps: false,
          fill: {
            target: "origin",
            above: "rgba(251, 191, 36, 0.16)",
            below: "rgba(248, 113, 113, 0.16)",
          },
        },
      ],
    },
    options: ROLLING_OPTIONS,
  });
}

function renderEquity(resp) {
  const live = $("btEquityLive");
  const empty = $("btEquityEmpty");
  const data = resp?.data;
  const symbols = data?.symbols || {};
  const symNames = Object.keys(symbols);

  if (!resp?.available || !symNames.length) {
    if (live) live.hidden = true;
    if (empty) empty.hidden = false;
    destroyEquityCharts();
    btEquitySig = "";
    return;
  }

  const focus = resp.focus_symbol;
  const sig = [focus, data.total_bars, data.n_points, data.rolling_window, symNames.join(",")].join("|") + "|" + btBuster;
  if (sig === btEquitySig && equityChart) return; // 无变化，避免重建闪烁
  btEquitySig = sig;

  if (live) live.hidden = false;
  if (empty) empty.hidden = true;

  const portfolio = data.portfolio || null;
  let mainName, mainSeries;
  if (portfolio) {
    mainName = "等权组合";
    mainSeries = portfolio;
  } else {
    const key = focus && symbols[focus] ? focus : symNames[0];
    mainName = key;
    mainSeries = symbols[key];
  }

  renderEquityStats(mainName, mainSeries);
  buildEquityChart(data.labels, symbols, portfolio);
  buildRollingChart(data.labels, mainSeries, data.rolling_window);

  if ($("btChartsHint")) {
    $("btChartsHint").textContent = `${mainName} · 交互式资金曲线 · 悬停查看数值`;
  }
}

async function startBacktest() {
  if (!selectedStrategyFile) {
    await logClientError("请先选择策略文件");
    return;
  }
  const startBtn = $("btStartBtn");
  if (startBtn) startBtn.disabled = true;
  try {
    const costs = readBacktestCosts();
    const res = await fetchJSON("/api/backtest/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        strategy_file: selectedStrategyFile,
        data_file: selectedBacktestDataFile,
        commission_pct: costs.commission_pct,
        slippage_pct: costs.slippage_pct,
      }),
    });
    if (res.strategy_file) renderStrategyFileCard(res.strategy_file);
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
    updateBtStartBtn();
  }
}

async function stopBacktest() {
  try {
    await fetchJSON("/api/backtest/stop", { method: "POST" });
    await refreshBacktest();
  } catch (e) {
    if ($("btLogHint")) $("btLogHint").textContent = e.message;
  }
}

// ═══════════════════════════════════════════════════════════════════
// 实时行情分析（信号雷达）
// ═══════════════════════════════════════════════════════════════════
let rtInited = false;
let rtEngineRunning = false;
let rtSources = [];
let rtSourceById = {};
let rtImportedStrategy = null; // {path, name}
let rtGridSig = "";
let rtServerSkew = 0; // server_time - local_now（秒）
let rtCountdownTimer = null;
let rtTvBlockedShownAt = 0;
let rtTvWikiUrl = "https://my.feishu.cn/wiki/FuqnwkPwdiCLhQkPloKc7r1lntg";
const RT_TV_BLOCKED_MSG =
  "当前设备无法连接 TradingView 数据服务，将无法获取以下 K 线数据：\n" +
  "  · A 股（上证 SSE、深证 SZSE）\n" +
  "  · 港股（HKEX）\n" +
  "  · 美股及指数（NYSE、NASDAQ、SP）\n" +
  "  · 外汇、贵金属、商品期货\n\n" +
  "解决方案：\n" +
  "  · 把你的VPN工具设成全局，并开启TUN(虚拟网卡)模式，如果还不行：\n" +
  "  · 使用云服务器部署本程序（推荐）—— 云服务器可正常连接 TradingView\n" +
  "  · 或切换回 MT5 数据源，仅使用 MT5 提供的品种数据";
const RT_TV_BLOCKED_CODE = "TV_CONNECTIVITY_BLOCKED";

const RT_DIR = {
  LONG: { label: "↑ 预期上涨", cls: "rt-long", color: "#4ade80" },
  SHORT: { label: "↓ 预期下跌", cls: "rt-short", color: "#f87171" },
  FLAT: { label: "— 先观望", cls: "rt-flat", color: "#7a8a9e" },
};
const RT_STATE_LABEL = {
  pending: "等待首次计算",
  ok: "运行中",
  insufficient: "历史不足",
  error: "错误",
};

/** 把 0~1 强度翻成「把握」白话 */
function rtSizePlain(strength, direction) {
  if (direction === "FLAT" || direction == null) {
    return { size: "没把握" };
  }
  const s = Math.max(0, Math.min(1, Number(strength) || 0));
  let size;
  if (s < 0.2) size = "一点把握";
  else if (s < 0.4) size = "把握不大";
  else if (s < 0.6) size = "一半把握";
  else if (s < 0.8) size = "比较有把握";
  else size = "很有把握";
  return { size };
}

function escHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

function rtClock(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

function rtNowSec() {
  return Date.now() / 1000 + rtServerSkew;
}

function rtFmtCountdown(sec) {
  const s = Math.max(0, Math.floor(sec));
  if (s < 60) return `${s}秒`;
  const m = Math.floor(s / 60);
  const rs = s % 60;
  if (m < 60) return `${m}分${String(rs).padStart(2, "0")}秒`;
  const h = Math.floor(m / 60);
  const rm = m % 60;
  if (h < 48) return `${h}小时${rm}分`;
  const d = Math.floor(h / 24);
  return `${d}天${h % 24}小时`;
}

function ensureRtCountdownTimer() {
  if (rtCountdownTimer) return;
  rtCountdownTimer = setInterval(tickRtCountdowns, 1000);
}

function tickRtCountdowns() {
  document.querySelectorAll(".rt-countdown").forEach((el) => {
    if (el.dataset.session === "closed") {
      el.textContent = "休市中";
      return;
    }
    const nxt = Number(el.dataset.nextClose);
    if (!Number.isFinite(nxt) || nxt <= 0) {
      el.textContent = "距离下次判断 —";
      return;
    }
    const left = nxt - rtNowSec();
    el.textContent = left <= 0 ? "即将重新判断…" : `距离下次判断 ${rtFmtCountdown(left)}`;
  });
  const hintCd = $("rtNextHint");
  if (hintCd) {
    if (hintCd.dataset.session === "closed") {
      hintCd.textContent = "休市中";
      return;
    }
    if (hintCd.dataset.nextClose) {
      const nxt = Number(hintCd.dataset.nextClose);
      if (Number.isFinite(nxt) && nxt > 0) {
        const left = nxt - rtNowSec();
        hintCd.textContent =
          left <= 0 ? "即将重新判断" : `距离下次判断 ${rtFmtCountdown(left)}`;
      }
    }
  }
}

async function initRealtimeOnce() {
  if (rtInited) return;
  rtInited = true;
  try {
    const data = await fetchJSON("/api/realtime/sources");
    rtSources = data.sources || [];
    rtSourceById = {};
    rtSources.forEach((s) => (rtSourceById[s.id] = s));
    const sel = $("rtSourceSelect");
    if (sel) {
      sel.innerHTML = rtSources
        .map((s) => `<option value="${s.id}">${escHtml(s.label)}${s.available ? "" : " · 未就绪"}</option>`)
        .join("");
      const preferred = data.selected_source || "mt5";
      const hasPreferred = rtSources.some((s) => s.id === preferred);
      if (hasPreferred) sel.value = preferred;
      else if (rtSources.length) sel.value = rtSources[0].id;
    }
    if (data.min_exposure != null && $("rtThresholdHint")) {
      $("rtThresholdHint").textContent = `无信号阈值 |tanh(因子)| < ${data.min_exposure}`;
    }
    onRtSourceChange({ persist: false });
  } catch (e) {
    await logClientError("加载数据源失败: " + e.message);
  }
  await loadRtStrategies();
  await loadRtFeishuSettings();
}

async function loadRtFeishuSettings() {
  try {
    const data = await fetchJSON("/api/realtime/feishu");
    const en = $("rtFeishuEnabled");
    const wh = $("rtFeishuWebhook");
    const sec = $("rtFeishuSecret");
    if (en) en.checked = !!data.enabled;
    if (wh) wh.value = data.webhook_url || "";
    if (sec) sec.value = data.secret || "";
  } catch (e) {
    const hint = $("rtFeishuHint");
    if (hint) {
      hint.textContent = "加载飞书设置失败: " + e.message;
      hint.classList.add("bad");
    }
  }
}

async function saveRtFeishuSettings() {
  const hint = $("rtFeishuHint");
  const btn = $("rtFeishuSaveBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/feishu", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        enabled: !!$("rtFeishuEnabled")?.checked,
        webhook_url: $("rtFeishuWebhook")?.value || "",
        secret: $("rtFeishuSecret")?.value || "",
      }),
    });
    if (hint) {
      hint.textContent = "✓ 已保存，方向转折时会推送到飞书群。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
    if (btn) {
      const old = btn.textContent;
      btn.textContent = "已保存";
      setTimeout(() => {
        if (btn.textContent === "已保存") btn.textContent = old || "保存";
      }, 1600);
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "保存失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function testRtFeishu() {
  const hint = $("rtFeishuHint");
  const btn = $("rtFeishuTestBtn");
  if (btn) btn.disabled = true;
  try {
    await fetchJSON("/api/realtime/feishu/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        webhook_url: $("rtFeishuWebhook")?.value || "",
        secret: $("rtFeishuSecret")?.value || "",
      }),
    });
    if (hint) {
      hint.textContent = "✓ 测试消息已发送，请到飞书群查收。";
      hint.classList.remove("bad", "invalid");
      hint.classList.add("valid");
    }
  } catch (e) {
    if (hint) {
      hint.textContent = "测试失败: " + e.message;
      hint.classList.remove("valid");
      hint.classList.add("bad", "invalid");
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

function openRtFeishuHelpModal() {
  const modal = $("rtFeishuHelpModal");
  if (modal) modal.hidden = false;
}

function closeRtFeishuHelpModal() {
  const modal = $("rtFeishuHelpModal");
  if (modal) modal.hidden = true;
}

async function loadRtStrategies() {
  const sel = $("rtStrategySelect");
  if (!sel) return;
  let rows = [];
  try {
    const data = await fetchJSON("/api/realtime/strategies");
    rows = data.strategies || [];
  } catch (_) {}
  const opts = ['<option value="">— 选择已保存策略 —</option>'];
  if (rtImportedStrategy) {
    const isym = escHtml(rtImportedStrategy.symbol || "");
    opts.push(
      `<option value="${escHtml(rtImportedStrategy.path)}" data-symbol="${isym}">导入: ${escHtml(rtImportedStrategy.name)}</option>`
    );
  }
  rows.forEach((r) => {
    const score = r.best_score != null ? Number(r.best_score).toFixed(3) : "—";
    const tf = r.display_timeframe || r.timeframe || "未知周期";
    const sourceNote =
      r.timeframe_source === "filename"
        ? "文件名推断"
        : r.timeframe_source === "unknown"
          ? "未标注"
          : "JSON标注";
    const stateNote = r.is_legacy ? "旧命名" : "规范";
    const filename = r.filename ? ` · ${r.filename}` : "";
    opts.push(
      `<option value="${escHtml(r.strategy_file)}" data-symbol="${escHtml(r.symbol || "")}">${escHtml(r.symbol)} ${escHtml(tf)} · 分数 ${score} · ${escHtml(sourceNote)} · ${escHtml(stateNote)}${escHtml(filename)}</option>`
    );
  });
  const prev = sel.value;
  sel.innerHTML = opts.join("");
  if (rtImportedStrategy) sel.value = rtImportedStrategy.path;
  else if (prev) sel.value = prev;
  onRtStrategyChange();
}

async function saveRtSourcePreference(source) {
  if (!source) return;
  try {
    await fetchJSON("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ realtime_source: source }),
      silent: true,
    });
  } catch (_) {
    /* preference only; keep UI responsive */
  }
}

function onRtSourceChange(options = {}) {
  const src = rtSourceById[$("rtSourceSelect")?.value];
  const tfSel = $("rtTimeframeSelect");
  const presets = $("rtSymbolPresets");
  const hint = $("rtSourceHint");
  if (!src) return;
  if (options.persist !== false) saveRtSourcePreference(src.id);
  if (tfSel) {
    const cur = tfSel.value;
    tfSel.innerHTML = (src.timeframes || []).map((t) => `<option value="${t}">${t}</option>`).join("");
    if (src.timeframes && src.timeframes.includes(cur)) tfSel.value = cur;
    else if (src.timeframes && src.timeframes.includes("1h")) tfSel.value = "1h";
  }
  // 品种输入/下拉切换：presets 较多时（如国内期货 60 个品种）用下拉框
  const symbolInput = $("rtSymbolInput");
  const symbolSelect = $("rtSymbolSelect");
  const useSelect = src.id === "domestic_futures" || (src.presets && src.presets.length > 20);
  if (symbolInput && symbolSelect) {
    if (useSelect) {
      symbolInput.hidden = true;
      symbolSelect.hidden = false;
      symbolSelect.innerHTML = (src.presets || [])
        .map((s) => `<option value="${escHtml(s)}">${escHtml(s)}</option>`)
        .join("");
      symbolSelect.onchange = () => { symbolInput.value = symbolSelect.value; };
      if (symbolSelect.value) symbolInput.value = symbolSelect.value;
    } else {
      symbolInput.hidden = false;
      symbolSelect.hidden = true;
      if (presets) {
        presets.innerHTML = (src.presets || [])
          .map((s) => `<option value="${escHtml(s)}"></option>`)
          .join("");
      }
    }
  } else if (presets) {
    presets.innerHTML = (src.presets || [])
      .map((s) => `<option value="${escHtml(s)}"></option>`)
      .join("");
  }
  if (hint) {
    hint.textContent = `${src.label}：${src.hint || ""}`;
    hint.classList.toggle("bad", !src.available);
  }
}

function rtParseSymbolFromFilename(pathOrName) {
  const name = String(pathOrName || "").split(/[/\\]/).pop() || "";
  let m = name.match(/^best_(.+)\.json$/i);
  if (m) return m[1];
  m = name.match(/^strategy_(.+)_step\d+/i);
  if (m) return m[1];
  return "";
}

function rtApplySymbolFromStrategy(sym) {
  const s = String(sym || "").trim();
  if (!s) return;
  const input = $("rtSymbolInput");
  if (input) input.value = s;
  const select = $("rtSymbolSelect");
  if (select && !select.hidden) select.value = s;
}

function onRtStrategyChange() {
  const sel = $("rtStrategySelect");
  const picked = $("rtStrategyPicked");
  if (!sel || !picked) return;
  const opt = sel.options[sel.selectedIndex];
  picked.textContent = sel.value
    ? `因子来源：${opt ? opt.textContent : sel.value}。信号取最后已收盘 bar。`
    : "因子来源：从已保存策略下拉选择，或「导入策略」选本地 JSON。信号取最后已收盘 bar。";
  if (!sel.value) return;
  const fromOpt = (opt && opt.dataset.symbol) || "";
  const fromImport =
    rtImportedStrategy && sel.value === rtImportedStrategy.path
      ? rtImportedStrategy.symbol || ""
      : "";
  const sym = fromOpt || fromImport || rtParseSymbolFromFilename(sel.value);
  rtApplySymbolFromStrategy(sym);
}

async function rtBrowseStrategy() {
  try {
    const res = await fetchJSON("/api/strategy-file/browse", { method: "POST" });
    if (res.cancelled) return;
    const name = res.filename || res.strategy_file;
    rtImportedStrategy = {
      path: res.strategy_file,
      name,
      symbol: (res.symbol || "").trim() || rtParseSymbolFromFilename(name),
    };
    await loadRtStrategies();
    rtApplySymbolFromStrategy(rtImportedStrategy.symbol);
  } catch (e) {
    await logClientError("导入策略失败: " + e.message);
  }
}

async function rtAddWatch() {
  const source = $("rtSourceSelect")?.value;
  const symbol = ($("rtSymbolInput")?.value || "").trim();
  const timeframe = $("rtTimeframeSelect")?.value;
  const strategy_file = $("rtStrategySelect")?.value;
  const picked = $("rtStrategyPicked");
  if (!symbol) {
    if (picked) { picked.textContent = "请填写品种"; picked.classList.add("bad"); }
    return;
  }
  if (!strategy_file) {
    if (picked) { picked.textContent = "请选择或导入策略因子"; picked.classList.add("bad"); }
    return;
  }
  const btn = $("rtAddBtn");
  if (btn) btn.disabled = true;
  try {
    if (source === "tradingview") {
      const reachable = await rtEnsureTradingViewReachable();
      if (!reachable) return;
    }
    await fetchJSON("/api/realtime/watch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, symbol, timeframe, strategy_file }),
    });
    if (picked) picked.classList.remove("bad");
    rtEngineRunning = true;
    rtGridSig = "";
    await refreshRealtime();
  } catch (e) {
    if (picked) { picked.textContent = "添加失败: " + e.message; picked.classList.add("bad"); }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function rtEnsureTradingViewReachable() {
  const picked = $("rtStrategyPicked");
  if (picked) {
    picked.textContent = "正在检测 TradingView 连通性…";
    picked.classList.remove("bad");
  }
  try {
    const res = await fetchJSON("/api/realtime/tradingview/probe", {
      method: "POST",
      silent: true,
    });
    if (res && res.ok) {
      if (picked) picked.textContent = "";
      return true;
    }
    if (res?.wiki_url) rtTvWikiUrl = res.wiki_url;
    await showTvBlockedDialog(res?.message || RT_TV_BLOCKED_MSG);
    if (picked) {
      picked.textContent = "TradingView 不可用，请开 VPN 或换云服务器";
      picked.classList.add("bad");
    }
    return false;
  } catch (e) {
    await showTvBlockedDialog(RT_TV_BLOCKED_MSG);
    if (picked) {
      picked.textContent = "TradingView 检测失败: " + (e.message || e);
      picked.classList.add("bad");
    }
    return false;
  }
}

function showTvBlockedDialog(message) {
  const now = Date.now();
  // 避免轮询反复弹出
  if (now - rtTvBlockedShownAt < 60_000) {
    return Promise.resolve("dedupe");
  }
  rtTvBlockedShownAt = now;
  const modal = $("tvBlockedModal");
  const body = $("tvBlockedBody");
  if (!modal || !body) {
    window.alert(message || RT_TV_BLOCKED_MSG);
    return Promise.resolve("alert");
  }
  body.textContent = message || RT_TV_BLOCKED_MSG;
  modal.hidden = false;
  return new Promise((resolve) => {
    modal._tvResolve = resolve;
  });
}

function closeTvBlockedDialog(choice) {
  const modal = $("tvBlockedModal");
  if (modal) modal.hidden = true;
  const resolve = modal && modal._tvResolve;
  if (resolve) {
    modal._tvResolve = null;
    resolve(choice || "cancel");
  }
}

async function onTvBlockedSwitchMt5() {
  const sel = $("rtSourceSelect");
  if (sel) {
    const hasMt5 = [...sel.options].some((o) => o.value === "mt5");
    if (hasMt5) {
      sel.value = "mt5";
      onRtSourceChange();
    }
  }
  closeTvBlockedDialog("mt5");
}

function onTvBlockedOpenCloud() {
  try {
    window.open(rtTvWikiUrl, "_blank", "noopener,noreferrer");
  } catch (_) {}
  closeTvBlockedDialog("cloud");
}

function maybeShowTvBlockedFromWatches(watches) {
  const hit = (watches || []).find(
    (w) =>
      w.source === "tradingview" &&
      (w.tv_blocked || w.message === RT_TV_BLOCKED_CODE)
  );
  if (!hit) return;
  showTvBlockedDialog(RT_TV_BLOCKED_MSG);
}

async function rtRemoveWatch(id) {
  try {
    await fetchJSON("/api/realtime/unwatch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    });
    rtGridSig = "";
    await refreshRealtime();
  } catch (e) {
    await logClientError("移除监控失败: " + e.message);
  }
}

async function refreshRealtime() {
  let st;
  try {
    st = await fetchJSON("/api/realtime/status", { silent: true });
  } catch (_) {
    return;
  }
  // 有监控项却未在跑时自动拉起（不再提供手动开关）
  if (st.count > 0 && !st.running) {
    try {
      st = await fetchJSON("/api/realtime/start", { method: "POST", silent: true });
    } catch (_) {}
  }
  rtEngineRunning = !!st.running;
  if (typeof st.server_time === "number") {
    rtServerSkew = st.server_time - Date.now() / 1000;
  }

  const nearest = st.nearest_seconds_to_next;
  let nearestClose = null;
  let anyLive = false;
  let anyOk = false;
  for (const w of st.watches || []) {
    if (w.state === "ok") anyOk = true;
    if (w.session_live && w.next_bar_close_at != null) {
      anyLive = true;
      if (nearestClose == null || w.next_bar_close_at < nearestClose) {
        nearestClose = w.next_bar_close_at;
      }
    }
  }

  const hint = $("rtStatusHint");
  if (hint) {
    const base = st.count
      ? `${rtEngineRunning ? "运行中" : "已暂停"} · ${st.count} 项`
      : "暂无监控项";
    if (nearestClose) {
      hint.innerHTML = `${base} · <span id="rtNextHint" data-next-close="${nearestClose}"></span>`;
    } else if (anyOk && !anyLive) {
      hint.innerHTML = `${base} · <span id="rtNextHint" data-session="closed">休市中</span>`;
    } else {
      hint.textContent = base;
    }
  }
  renderRealtimeGrid(st.watches || []);
  maybeShowTvBlockedFromWatches(st.watches || []);
  ensureRtCountdownTimer();
  tickRtCountdowns();
}

// 半环表盘（180° 上半环，值弧按强度填充）
const RT_ARC_LEN = 150.8; // π * 48
function halfRingGauge(strength, colorHex) {
  const s = Math.max(0, Math.min(1, strength || 0));
  const off = RT_ARC_LEN * (1 - s);
  return `<svg class="rt-gauge-svg" viewBox="0 0 120 74" aria-hidden="true">
    <path class="rt-gauge-track" d="M12 62 A 48 48 0 0 1 108 62" />
    <path class="rt-gauge-val" d="M12 62 A 48 48 0 0 1 108 62"
      style="stroke:${colorHex};stroke-dasharray:${RT_ARC_LEN};stroke-dashoffset:${off.toFixed(1)};" />
  </svg>`;
}

function renderRealtimeGrid(watches) {
  const grid = $("rtGrid");
  if (!grid) return;
  if (!watches.length) {
    grid.innerHTML =
      '<div class="metric-empty">尚无监控项。添加「数据源 + 品种 + 周期 + 因子」后开始实时分析。</div>';
    rtGridSig = "";
    return;
  }

  // 签名：只在信号相关字段变化时重建（避免每次轮询重播动画）
  const sig = watches
    .map((w) =>
      [
        w.id,
        w.state,
        w.direction,
        w.strength,
        w.warn,
        w.message,
        w.last_bar_ts,
        w.updated_at,
        w.session_live ? 1 : 0,
        w.next_bar_close_at || "",
      ].join("~")
    )
    .join("|");
  // 签名未变时仍同步休市/倒计时锚点
  if (sig === rtGridSig) {
    watches.forEach((w) => {
      const el = grid.querySelector(`.rt-card[data-id="${CSS.escape(w.id)}"] .rt-countdown`);
      if (!el) return;
      if (w.session_live && w.next_bar_close_at) {
        el.dataset.session = "";
        el.dataset.nextClose = String(w.next_bar_close_at);
      } else if (w.state === "ok") {
        el.dataset.nextClose = "";
        el.dataset.session = "closed";
      } else {
        el.dataset.nextClose = "";
        el.dataset.session = "";
      }
    });
    return;
  }
  rtGridSig = sig;

  grid.innerHTML = watches
    .map((w) => {
      const dir = w.state === "ok" ? RT_DIR[w.direction] || RT_DIR.FLAT : null;
      const color = dir ? dir.color : "#7a8a9e";
      const strength = w.state === "ok" ? w.strength || 0 : 0;
      const dirKey = w.state === "ok" ? w.direction : null;
      const plain = w.state === "ok" ? rtSizePlain(strength, dirKey) : null;
      const dirLabel = dir ? dir.label : RT_STATE_LABEL[w.state] || w.state;
      const dirCls = dir ? dir.cls : "rt-flat";
      const srcLabel = (rtSourceById[w.source] || {}).label || w.source;
      const factorText = w.factor_value != null ? Number(w.factor_value).toFixed(4) : "—";
      const warn = w.warn ? `<div class="rt-warn" title="${escHtml(w.warn)}">⚠ ${escHtml(w.warn)}</div>` : "";
      const displayMsg =
        w.message === RT_TV_BLOCKED_CODE || w.tv_blocked
          ? "无法连接 TradingView：请开启全局 VPN（TUN）或使用云服务器"
          : w.message;
      const msg =
        w.state !== "ok" && displayMsg
          ? `<div class="rt-msg">${escHtml(displayMsg)}</div>`
          : "";
      const sizeText = plain ? plain.size : "—";
      return `
    <div class="rt-card ${dirCls}" data-id="${escHtml(w.id)}">
      <button class="rt-remove" data-remove="${escHtml(w.id)}" title="移除监控">×</button>
      <div class="rt-card-head">
        <span class="rt-sym">${escHtml(w.symbol)}</span>
        <span class="rt-tf">${escHtml(w.timeframe)}</span>
        <span class="rt-src">${escHtml(srcLabel)}</span>
      </div>
      <div class="rt-gauge">
        ${halfRingGauge(strength, color)}
        <div class="rt-gauge-center">
          <div class="rt-strength">${escHtml(sizeText)}</div>
          <div class="rt-dir ${dirCls}">${dirLabel}</div>
        </div>
      </div>
      <div class="rt-meta">
        <span class="rt-meta-item">因子 <b>${factorText}</b></span>
        <span class="rt-meta-item">${escHtml(w.strategy_name)}</span>
      </div>
      <div class="rt-foot">
        <span class="rt-state ${w.state}">${RT_STATE_LABEL[w.state] || w.state}</span>
        <span class="rt-time">更新 ${rtClock(w.updated_at)}</span>
        <span class="rt-countdown"${
          w.session_live && w.next_bar_close_at
            ? ` data-next-close="${w.next_bar_close_at}"`
            : w.state === "ok"
              ? ` data-session="closed"`
              : ""
        }>${
          w.session_live && w.next_bar_close_at
            ? "距离下次判断 …"
            : w.state === "ok"
              ? "休市中"
              : "距离下次判断 —"
        }</span>
      </div>
      ${warn}
      ${msg}
    </div>`;
    })
    .join("");

  runCountUp(grid);
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    refreshOverview();
    if (currentPage === "backtest" || btActive) refreshBacktest();
    if (currentPage === "realtime" || rtEngineRunning) refreshRealtime();
  }, 4000);
}

async function init() {
  try {
    await loadConfig();
    await refreshOverview();
  } catch (e) {
    await logClientError("初始化失败: " + e.message);
  }
  initEvalModeSelect();
  $("browseBtn").addEventListener("click", browseDataFile);
  if ($("dataRootBrowseBtn")) $("dataRootBrowseBtn").addEventListener("click", browseDataRootDir);
  if ($("dataRootSaveBtn")) $("dataRootSaveBtn").addEventListener("click", saveDataRootDir);
  if ($("dataRootInput")) $("dataRootInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") saveDataRootDir();
  });
  $("startBtn").addEventListener("click", startTraining);
  if ($("retrainBtn")) $("retrainBtn").addEventListener("click", retrainFromScratch);
  $("stopBtn").addEventListener("click", stopTraining);
  $("exportBtn").addEventListener("click", exportStrategy);
  $("exportTrainingBtn").addEventListener("click", exportTraining);
  $("importTrainingBtn").addEventListener("click", triggerImportTraining);
  $("importTrainingFile").addEventListener("change", handleImportTrainingFile);
  $("debugModeCheck").addEventListener("change", (e) => setDebugMode(e.target.checked));
  if ($("aiApiKeyInput")) {
    $("aiApiKeyInput").addEventListener("input", updateAiChannelHint);
    $("aiApiKeyInput").addEventListener("change", updateAiChannelHint);
  }
  if ($("aiAnalyzeBtn")) $("aiAnalyzeBtn").addEventListener("click", runAiAnalyze);
  if ($("aiUnlimitedBtn")) $("aiUnlimitedBtn").addEventListener("click", openUnlimitedModal);
  document.querySelectorAll("[data-close-unlimited]").forEach((el) => {
    el.addEventListener("click", closeUnlimitedModal);
  });
  document.querySelectorAll("[data-close-error]").forEach((el) => {
    el.addEventListener("click", closeErrorPopup);
  });
  if ($("errorModalCopyBtn")) {
    $("errorModalCopyBtn").addEventListener("click", copyErrorPopupDetail);
  }

  // 步骤导航
  document.querySelectorAll(".stepper .step").forEach((btn) => {
    btn.addEventListener("click", () => switchPage(btn.dataset.page));
  });

  // 回测控制
  if ($("btBrowseStrategyBtn")) $("btBrowseStrategyBtn").addEventListener("click", browseStrategyFile);
  if ($("btStrategySelect")) $("btStrategySelect").addEventListener("change", onBacktestStrategySelect);
  if ($("btDataFileSelect")) $("btDataFileSelect").addEventListener("change", onBacktestDataFileSelect);
  if ($("btStartBtn")) $("btStartBtn").addEventListener("click", startBacktest);
  if ($("btStopBtn")) $("btStopBtn").addEventListener("click", stopBacktest);
  ["btCommissionInput", "btSlippageInput"].forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener("input", updateBtCostHint);
    el.addEventListener("change", updateBtCostHint);
  });

  // 实时分析控制
  if ($("rtSourceSelect")) $("rtSourceSelect").addEventListener("change", onRtSourceChange);
  if ($("rtStrategySelect")) $("rtStrategySelect").addEventListener("change", onRtStrategyChange);
  if ($("rtBrowseStrategyBtn")) $("rtBrowseStrategyBtn").addEventListener("click", rtBrowseStrategy);
  if ($("rtAddBtn")) $("rtAddBtn").addEventListener("click", rtAddWatch);
  if ($("tvBlockedMt5Btn")) $("tvBlockedMt5Btn").addEventListener("click", onTvBlockedSwitchMt5);
  if ($("tvBlockedCloudBtn")) $("tvBlockedCloudBtn").addEventListener("click", onTvBlockedOpenCloud);
  document.querySelectorAll("[data-close-tv-blocked]").forEach((el) => {
    el.addEventListener("click", () => closeTvBlockedDialog("cancel"));
  });
  if ($("rtFeishuSaveBtn")) $("rtFeishuSaveBtn").addEventListener("click", saveRtFeishuSettings);
  if ($("rtFeishuTestBtn")) $("rtFeishuTestBtn").addEventListener("click", testRtFeishu);
  if ($("rtFeishuHelpBtn")) $("rtFeishuHelpBtn").addEventListener("click", openRtFeishuHelpModal);
  document.querySelectorAll("[data-close-feishu-help]").forEach((el) => {
    el.addEventListener("click", closeRtFeishuHelpModal);
  });
  if ($("rtGrid")) {
    $("rtGrid").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-remove]");
      if (btn) rtRemoveWatch(btn.dataset.remove);
    });
  }

  startPolling();
}

init();
