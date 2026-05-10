<script lang="ts">
  import { get } from 'svelte/store';

  import {
    buildBenchmarkCombos,
    defaultBenchmarkRows,
    downloadBenchmarkJson,
    estimateBenchmarkDuration,
    formatDuration,
    getBestBenchmarkResult,
    getNumericFieldOptions,
    resolveBenchmarkParam,
    type BenchmarkParamDraft,
    type BenchmarkResultEntry
  } from '$lib/benchmark';
  import Button from '$lib/components/Button.svelte';
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { pipelineValues } from '$lib/store';
  import type { DebugInfo, FieldProps, Fields } from '$lib/types';
  import { InputSource } from '$lib/types';
  import { videoControllerActions, videoControllerState } from '$lib/videoController';

  export let pipelineParams: Fields;
  export let inputSource: InputSource = InputSource.CAMERA;
  export let isImageMode: boolean = false;
  export let running: boolean = false;

  const CANCELLED = 'BENCHMARK_CANCELLED';

  let waitSeconds = 60;
  let rows: BenchmarkParamDraft[] = [];
  let seededDefaults = false;
  let phaseLabel = '待命';
  let phaseRemainingSeconds: number | null = null;
  let currentIndex = 0;
  let currentParams: Record<string, number> | null = null;
  let currentScore: number | null = null;
  let errorMessage = '';
  let lastDownloadName = '';
  let results: BenchmarkResultEntry[] = [];
  let abortRequested = false;

  $: numericFields = getNumericFieldOptions(pipelineParams ?? {});
  $: fieldMap = new Map<string, FieldProps>(numericFields.map((field) => [field.id, field]));
  $: if (!seededDefaults && numericFields.length > 0) {
    rows = defaultBenchmarkRows(numericFields);
    seededDefaults = true;
  }
  $: resolvedRows = rows.map((row) => {
    if (!row.enabled) {
      return { row, spec: null, error: '' };
    }
    try {
      return { row, spec: resolveBenchmarkParam(row, fieldMap), error: '' };
    } catch (err) {
      return {
        row,
        spec: null,
        error: err instanceof Error ? err.message : '参数配置有误'
      };
    }
  });
  $: activeSpecs = resolvedRows
    .filter((item) => item.row.enabled && item.spec)
    .map((item) => item.spec as NonNullable<(typeof item)['spec']>);
  $: rowErrors = resolvedRows
    .filter((item) => item.row.enabled && item.error)
    .map((item) => `${fieldMap.get(item.row.paramId)?.title ?? item.row.paramId}: ${item.error}`);
  $: comboPreview = rowErrors.length === 0 ? buildBenchmarkCombos(activeSpecs) : [];
  $: totalCombos = comboPreview.length;
  $: estimatedDuration = formatDuration(estimateBenchmarkDuration(totalCombos, Number(waitSeconds), 0));
  $: bestResult = getBestBenchmarkResult(results);
  $: sessionReady = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED && Boolean($streamId);
  $: canRun =
    !running &&
    isImageMode &&
    inputSource === InputSource.VIDEO &&
    $videoControllerState.hasVideoLoaded &&
    sessionReady &&
    rowErrors.length === 0 &&
    totalCombos > 0;
  $: selectedSoftedge = activeSpecs.some((spec) => spec.paramId === 'softedge_scale');
  $: softedgeHint =
    selectedSoftedge && get(pipelineValues).use_softedge !== true
      ? '当前 `use_softedge` 未开启，softedge_scale 的搜索结果可能没有意义。'
      : '';

  function createRow(fieldId: string): BenchmarkParamDraft {
    const field = fieldMap.get(fieldId) ?? numericFields[0];
    return {
      key: `${field?.id ?? 'param'}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      enabled: true,
      paramId: field?.id ?? '',
      mode: 'list',
      valuesText: '0, 0.2, 0.4, 0.6, 0.8, 1',
      minText: String(field?.min ?? 0),
      maxText: String(field?.max ?? 1),
      stepText: String(field?.step ?? 0.2)
    };
  }

  function addRow() {
    if (numericFields.length === 0) return;
    rows = [...rows, createRow(numericFields[0].id)];
  }

  function removeRow(key: string) {
    rows = rows.filter((row) => row.key !== key);
  }

  function duplicateRow(row: BenchmarkParamDraft) {
    rows = [...rows, createRow(row.paramId)];
    rows = rows.map((item, index, list) =>
      index === list.length - 1
        ? {
            ...item,
            enabled: row.enabled,
            mode: row.mode,
            valuesText: row.valuesText,
            minText: row.minText,
            maxText: row.maxText,
            stepText: row.stepText
          }
        : item
    );
  }

  function markCancelled() {
    abortRequested = true;
    phaseLabel = '正在中止';
    phaseRemainingSeconds = null;
  }

  function assertNotCancelled() {
    if (abortRequested) {
      throw new Error(CANCELLED);
    }
  }

  function getActiveStreamId(): string {
    const sid = $streamId;
    if (!sid || $lcmLiveStatus === LCMLiveStatus.DISCONNECTED) {
      throw new Error('请先手动点击上方 Start 启动推理，再启动稳定性测试');
    }
    return sid;
  }

  async function fetchDebugInfo(sid: string): Promise<DebugInfo | null> {
    const response = await fetch(`/api/debug/${sid}`, { cache: 'no-store' });
    if (!response.ok) return null;
    const data = (await response.json()) as DebugInfo;
    return Object.keys(data).length > 0 ? data : null;
  }

  async function sleep(ms: number) {
    await new Promise<void>((resolve) => {
      window.setTimeout(resolve, ms);
    });
  }

  async function waitWindow(seconds: number, sid: string) {
    phaseLabel = '等待指标稳定';
    const deadline = Date.now() + seconds * 1000;
    while (Date.now() < deadline) {
      assertNotCancelled();
      if ($lcmLiveStatus === LCMLiveStatus.DISCONNECTED || $streamId !== sid) {
        throw new Error('推理会话已中断，测试终止');
      }
      phaseRemainingSeconds = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
      await sleep(200);
    }
    phaseRemainingSeconds = 0;
  }

  async function readCurrentScore(sid: string): Promise<number | null> {
    phaseLabel = '读取指标';
    const debugInfo = await fetchDebugInfo(sid);
    const score = debugInfo?.temporal_stability_delta_avg_30s;
    currentScore = typeof score === 'number' && Number.isFinite(score) ? score : null;
    return currentScore;
  }

  async function prepareCombo(sid: string, params: Record<string, number>) {
    if ($lcmLiveStatus === LCMLiveStatus.DISCONNECTED || $streamId !== sid) {
      throw new Error('推理会话已中断，无法继续测试');
    }
    await videoControllerActions.jumpToStart();
    pipelineValues.update((value) => ({ ...value, ...params }));
    await sleep(300);
  }

  function restoreState(baseValues: Record<string, any>) {
    pipelineValues.set(baseValues);
  }

  async function startBenchmark() {
    errorMessage = '';
    lastDownloadName = '';
    results = [];
    currentIndex = 0;
    currentParams = null;
    currentScore = null;
    abortRequested = false;

    if (!canRun) {
      errorMessage = '请先加载本地视频，并手动点击上方 Start 启动推理。';
      return;
    }

    const specs = [...activeSpecs];
    const combos = buildBenchmarkCombos(specs);
    const baseValues = { ...get(pipelineValues) };
    const initialVideoState = get(videoControllerState);
    let sid = '';

    running = true;
    phaseLabel = '等待开始';
    phaseRemainingSeconds = null;

    try {
      sid = getActiveStreamId();
      for (const [index, params] of combos.entries()) {
        assertNotCancelled();
        currentIndex = index + 1;
        currentParams = params;
        currentScore = null;
        await prepareCombo(sid, params);
        if (waitSeconds > 0) {
          await waitWindow(waitSeconds, sid);
        }
        const startedAt = new Date().toISOString();
        const score = await readCurrentScore(sid);
        const completedAt = new Date().toISOString();
        results = [
          ...results,
          {
            index: index + 1,
            params,
            score,
            sample_points: [],
            sample_count: 0,
            started_at: startedAt,
            completed_at: completedAt
          }
        ];
      }

      const finalResults = [...results];
      const payload = {
        meta: {
          created_at: new Date().toISOString(),
          source: 'video',
          video_file: initialVideoState.selectedVideoFileName || 'video',
          combo_count: combos.length,
          wait_seconds: Number(waitSeconds),
          score_field: 'temporal_stability_delta_avg_30s'
        },
        search_space: specs.map((spec) => ({
          param_id: spec.paramId,
          title: spec.title,
          mode: spec.mode,
          values: spec.values,
          min: spec.min,
          max: spec.max,
          step: spec.step
        })),
        base_values: baseValues,
        results: finalResults,
        best_result: getBestBenchmarkResult(finalResults)
      };
      const stem = `${initialVideoState.selectedVideoFileName || 'video'}_stability_benchmark_${Date.now()}`;
      lastDownloadName = downloadBenchmarkJson(stem, payload);
      phaseLabel = '已完成';
      phaseRemainingSeconds = null;
    } catch (err) {
      if (err instanceof Error && err.message === CANCELLED) {
        errorMessage = '测试已中止。';
        phaseLabel = '已中止';
      } else {
        errorMessage = err instanceof Error ? err.message : '测试失败';
        phaseLabel = '失败';
      }
    } finally {
      restoreState(baseValues);
      running = false;
      abortRequested = false;
      phaseRemainingSeconds = null;
      currentParams = null;
      currentScore = null;
    }
  }
</script>

<div class="flex flex-col gap-4">
  <div class="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
    <div>
      <h3 class="text-lg font-semibold text-black dark:text-slate-200">稳定性测试</h3>
      <p class="mt-0.5 text-[11px] uppercase tracking-[0.18em] text-black dark:text-slate-400">
        参数遍历 · 自动采样 · 导出 JSON
      </p>
    </div>
    <div class="flex flex-wrap gap-2 text-xs text-black dark:text-slate-300">
      <span class="hud-chip">组合 {totalCombos}</span>
      <span class="hud-chip">预计 {estimatedDuration}</span>
      <span class="hud-chip">最佳 {bestResult?.score?.toFixed(3) ?? '—'}</span>
    </div>
  </div>

  <div class="benchmark-card grid gap-4">
    <div class="grid gap-3 lg:grid-cols-4">
      <label class="setting-card">
        <span>等待秒数</span>
        <input bind:value={waitSeconds} min="0" step="1" type="number" />
      </label>
      <label class="setting-card">
        <span>评分字段</span>
        <input value="temporal_stability_delta_avg_30s" disabled type="text" />
      </label>
      <label class="setting-card">
        <span>输入要求</span>
        <input value="本地视频 + 已手动 Start" disabled type="text" />
      </label>
      <label class="setting-card">
        <span>评分窗口</span>
        <input value="近30秒均值（现有调试指标）" disabled type="text" />
      </label>
    </div>

    <div class="grid gap-3">
      {#each rows as row, index}
        <div class="row-card">
          <label class="inline-flex items-center gap-2 text-sm">
            <input bind:checked={row.enabled} type="checkbox" />
            <span>参数 {index + 1}</span>
          </label>
          <label class="row-field">
            <span>字段</span>
            <select bind:value={row.paramId}>
              {#each numericFields as field}
                <option value={field.id}>{field.title}</option>
              {/each}
            </select>
          </label>
          <label class="row-field">
            <span>模式</span>
            <select bind:value={row.mode}>
              <option value="list">离散列表</option>
              <option value="range">范围步长</option>
            </select>
          </label>
          {#if row.mode === 'list'}
            <label class="row-field row-field--wide">
              <span>候选值</span>
              <input bind:value={row.valuesText} placeholder="0, 0.2, 0.4, 0.6, 0.8, 1" type="text" />
            </label>
          {:else}
            <label class="row-field">
              <span>最小值</span>
              <input bind:value={row.minText} type="number" step="0.01" />
            </label>
            <label class="row-field">
              <span>最大值</span>
              <input bind:value={row.maxText} type="number" step="0.01" />
            </label>
            <label class="row-field">
              <span>步长</span>
              <input bind:value={row.stepText} type="number" step="0.01" />
            </label>
          {/if}
          <div class="row-actions">
            <Button
              on:click={() => duplicateRow(row)}
              disabled={running}
              classList={'px-3 py-2 text-xs font-semibold'}
            >
              复制
            </Button>
            <Button
              on:click={() => removeRow(row.key)}
              disabled={running || rows.length <= 1}
              classList={'px-3 py-2 text-xs font-semibold'}
            >
              删除
            </Button>
          </div>
        </div>
      {/each}
      <div class="flex flex-wrap gap-3">
        <Button on:click={addRow} disabled={running || numericFields.length === 0} classList={'px-4 py-2 text-sm font-semibold'}>
          添加参数
        </Button>
        {#if running}
          <Button on:click={markCancelled} classList={'px-4 py-2 text-sm font-semibold'}>
            中止测试
          </Button>
        {:else}
          <Button on:click={startBenchmark} disabled={!canRun} active={canRun} classList={'px-5 py-2 text-sm font-semibold'}>
            开始批量测试
          </Button>
        {/if}
      </div>
    </div>

    {#if !isImageMode}
      <p class="note">当前模式不是逐帧图像输入，暂不支持稳定性测试。</p>
    {:else if inputSource !== InputSource.VIDEO}
      <p class="note">请先切换到本地视频输入，再启动稳定性测试。</p>
    {:else if !$videoControllerState.hasVideoLoaded}
      <p class="note">请先加载本地视频。</p>
    {:else if !sessionReady}
      <p class="note">请先点击上方 `Start` 启动推理，再启动稳定性测试。</p>
    {/if}

    {#if softedgeHint}
      <p class="warn">{softedgeHint}</p>
    {/if}

    {#if rowErrors.length > 0}
      <div class="warn flex flex-col gap-1">
        {#each rowErrors as item}
          <p>{item}</p>
        {/each}
      </div>
    {/if}

    {#if running || results.length > 0 || errorMessage}
      <div class="status-grid">
        <div class="metric-card py-3">
          <p class="text-[10px] uppercase tracking-[0.18em] text-black dark:text-slate-400">状态</p>
          <p class="mt-1 text-sm font-semibold text-black dark:text-slate-200">{phaseLabel}</p>
        </div>
        <div class="metric-card py-3">
          <p class="text-[10px] uppercase tracking-[0.18em] text-black dark:text-slate-400">进度</p>
          <p class="mt-1 text-sm font-semibold text-black dark:text-slate-200">{currentIndex}/{totalCombos}</p>
        </div>
        <div class="metric-card py-3">
          <p class="text-[10px] uppercase tracking-[0.18em] text-black dark:text-slate-400">倒计时</p>
          <p class="mt-1 text-sm font-semibold text-black dark:text-slate-200">{phaseRemainingSeconds ?? '—'}</p>
        </div>
        <div class="metric-card py-3">
          <p class="text-[10px] uppercase tracking-[0.18em] text-black dark:text-slate-400">当前均分</p>
          <p class="mt-1 text-sm font-semibold text-black dark:text-slate-200">{currentScore != null ? currentScore.toFixed(4) : '—'}</p>
        </div>
      </div>
    {/if}

    {#if currentParams}
      <div class="note">
        当前参数：
        {#each Object.entries(currentParams) as [key, value], index}
          <span>{key}={value}{index < Object.entries(currentParams).length - 1 ? ' · ' : ''}</span>
        {/each}
      </div>
    {/if}

    {#if bestResult}
      <div class="note">
        最佳组合：{#each Object.entries(bestResult.params) as [key, value], index}
          <span>{key}={value}{index < Object.entries(bestResult.params).length - 1 ? ' · ' : ''}</span>
        {/each}
        ，得分 {bestResult.score?.toFixed(4)}
      </div>
    {/if}

    {#if lastDownloadName}
      <p class="note">结果已导出：{lastDownloadName}</p>
    {/if}

    {#if errorMessage}
      <p class="warn">{errorMessage}</p>
    {/if}
  </div>
</div>

<style lang="postcss">
  .benchmark-card {
    @apply rounded-[24px] border p-4 sm:p-5;
    border-color: rgba(34, 211, 238, 0.22);
    background:
      linear-gradient(180deg, rgba(255, 255, 255, 0.88), rgba(240, 249, 255, 0.68)),
      rgba(255, 255, 255, 0.8);
  }

  .setting-card,
  .row-field {
    @apply flex flex-col gap-2 rounded-2xl border px-3 py-3 text-sm;
    border-color: rgba(148, 163, 184, 0.24);
    background: rgba(255, 255, 255, 0.72);
  }

  .setting-card span,
  .row-field span {
    @apply text-xs font-semibold uppercase tracking-[0.14em] text-black dark:text-slate-400;
  }

  .setting-card input,
  .row-field input,
  .row-field select {
    @apply rounded-xl border px-3 py-2 text-sm text-black outline-none transition dark:text-slate-100;
    border-color: rgba(148, 163, 184, 0.32);
    background: rgba(255, 255, 255, 0.92);
  }

  .row-card {
    @apply grid gap-3 rounded-[22px] border p-3 lg:grid-cols-[auto_1.2fr_0.8fr_1.6fr_auto];
    border-color: rgba(148, 163, 184, 0.22);
    background: rgba(255, 255, 255, 0.54);
  }

  .row-field--wide {
    @apply lg:col-span-2;
  }

  .row-actions {
    @apply flex flex-wrap items-end gap-2;
  }

  .status-grid {
    @apply grid gap-2 sm:grid-cols-2 xl:grid-cols-4;
  }

  .note,
  .warn {
    @apply rounded-2xl px-4 py-3 text-sm;
  }

  .note {
    @apply text-black dark:text-slate-200;
    background: rgba(148, 163, 184, 0.12);
  }

  .warn {
    @apply text-amber-950 dark:text-amber-100;
    background: rgba(251, 191, 36, 0.18);
  }

  :global(.dark) .benchmark-card {
    border-color: rgba(34, 211, 238, 0.16);
    background:
      linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(8, 47, 73, 0.26)),
      rgba(15, 23, 42, 0.7);
  }

  :global(.dark) .setting-card,
  :global(.dark) .row-field,
  :global(.dark) .row-card {
    border-color: rgba(71, 85, 105, 0.8);
    background: rgba(15, 23, 42, 0.62);
  }

  :global(.dark) .setting-card input,
  :global(.dark) .row-field input,
  :global(.dark) .row-field select {
    border-color: rgba(71, 85, 105, 0.92);
    background: rgba(15, 23, 42, 0.88);
  }
</style>
