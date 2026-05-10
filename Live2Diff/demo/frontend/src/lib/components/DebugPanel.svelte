<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import type { DebugInfo } from '$lib/types';
  import { onDestroy } from 'svelte';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;

  const placeholderSrc = '/debug-preview-placeholder.png';
  const previewCol = 'minmax(0,260px)';

  function fmtStability(n: number | null | undefined): string {
    return n != null && !Number.isNaN(n) ? n.toFixed(2) : '—';
  }

  let debugInfo: DebugInfo | null = null;
  let depthSrc = '';
  let softedgeSrc = '';
  let subjectMaskSrc = '';
  let inputSrc = '';
  let debugTimer: ReturnType<typeof setInterval> | null = null;
  $: showSoftedgeMetrics =
    debugInfo &&
    (debugInfo.use_softedge ||
      debugInfo.has_softedge_preview ||
      (debugInfo.softedge_time ?? 0) > 0);
  $: metrics = debugInfo
    ? [
        { label: 'Status', value: debugInfo.phase },
        {
          label: '帧率',
          value:
            debugInfo.inference_time > 0 ? `${(1 / debugInfo.inference_time).toFixed(1)} FPS` : '—'
        },
        { label: 'Inference', value: `${(debugInfo.inference_time * 1000).toFixed(1)} ms` },
        { label: 'Depth', value: `${(debugInfo.depth_time * 1000).toFixed(1)} ms` },
        {
          label: '软描边',
          value: showSoftedgeMetrics
            ? `${((debugInfo.softedge_time ?? 0) * 1000).toFixed(1)} ms`
            : '—'
        },
        {
          label: 'SoftEdge 模式',
          value: debugInfo.softedge_mode ?? '—'
        },
        {
          label: 'SoftEdge 实际后端',
          value: debugInfo.softedge_backend ?? '—'
        },
        {
          label: '主体 Mask',
          value: debugInfo.has_subject_mask_preview
            ? `${((debugInfo.subject_mask_time ?? 0) * 1000).toFixed(1)} ms`
            : '—'
        },
        {
          label: 'Mask 后端',
          value: debugInfo.subject_mask_backend ?? '—'
        },
        ...(debugInfo.softedge_pidinet_error
          ? [{ label: 'PiDiNet 错误', value: debugInfo.softedge_pidinet_error }]
          : []),
        ...(debugInfo.subject_mask_sam2_error
          ? [{ label: 'SAM2 错误', value: debugInfo.subject_mask_sam2_error }]
          : []),
        { label: 'Steps', value: String(debugInfo.inference_steps) },
        { label: 'Size', value: `${debugInfo.width} x ${debugInfo.height}` },
        { label: '原画稳定度', value: fmtStability(debugInfo.temporal_stability_src) },
        { label: '重绘稳定度', value: fmtStability(debugInfo.temporal_stability_out) },
        { label: '稳定度差值', value: fmtStability(debugInfo.temporal_stability_delta) },
        { label: 'Δ 近30s 均值', value: fmtStability(debugInfo.temporal_stability_delta_avg_30s) },
        { label: '关键步同步', value: debugInfo.sync_enabled ? 'on' : 'off' },
        { label: '同步步', value: String(debugInfo.sync_key_step_index ?? 0) },
        { label: '同步权重', value: fmtStability(debugInfo.sync_weight) },
        { label: '同步记忆', value: debugInfo.sync_memory_valid ? 'ready' : 'empty' },
        { label: '光流后端', value: debugInfo.flow_backend ?? 'farneback' }
      ]
    : [];

  async function refreshDebug() {
    try {
      if (!$streamId) {
        return;
      }
      const response = await fetch(`/api/debug/${$streamId}`, {
        cache: 'no-store'
      });
      if (!response.ok) {
        debugInfo = null;
        depthSrc = '';
        softedgeSrc = '';
        subjectMaskSrc = '';
        inputSrc = '';
        return;
      }
      const data = await response.json();
      debugInfo = Object.keys(data).length > 0 ? data : null;
      const t = Date.now();
      depthSrc = debugInfo?.has_depth ? `/api/depth/${$streamId}?t=${t}` : '';
      softedgeSrc = debugInfo?.has_softedge_preview ? `/api/softedge/${$streamId}?t=${t}` : '';
      subjectMaskSrc = debugInfo?.has_subject_mask_preview
        ? `/api/subject-mask/${$streamId}?t=${t}`
        : '';
      inputSrc = debugInfo?.has_stylized_preview
        ? `/api/stylized/${$streamId}?t=${t}`
        : debugInfo?.has_source_preview
          ? `/api/preview/input/${$streamId}?t=${t}`
          : '';
    } catch {
      debugInfo = null;
      depthSrc = '';
      softedgeSrc = '';
      subjectMaskSrc = '';
      inputSrc = '';
    }
  }

  function stopDebugPolling() {
    if (debugTimer !== null) {
      clearInterval(debugTimer);
      debugTimer = null;
    }
    debugInfo = null;
    depthSrc = '';
    softedgeSrc = '';
    subjectMaskSrc = '';
    inputSrc = '';
  }

  function startDebugPolling() {
    if (debugTimer !== null) {
      return;
    }
    void refreshDebug();
    debugTimer = setInterval(() => {
      void refreshDebug();
    }, 1000);
  }

  $: if (isLCMRunning && $streamId) {
    startDebugPolling();
  } else {
    stopDebugPolling();
  }

  onDestroy(() => {
    stopDebugPolling();
  });
</script>

<div class="flex flex-col gap-3">
  <div class="flex flex-wrap items-end justify-between gap-2">
    <h3 class="text-lg font-semibold text-black dark:text-slate-200">调试信息</h3>
    <p class="text-[10px] uppercase tracking-[0.16em] text-black dark:text-slate-500">
      Depth · 软描边 · 主体 Mask · 进 latent · 指标
    </p>
  </div>
  <div
    class="debug-preview-grid grid gap-3 lg:grid-cols-[var(--dbg-c1)_var(--dbg-c2)_var(--dbg-c3)_var(--dbg-c4)_1fr]"
    style="--dbg-c1: {previewCol}; --dbg-c2: {previewCol}; --dbg-c3: {previewCol}; --dbg-c4: {previewCol};"
  >
    <div class="data-panel">
      <div class="mb-1.5 flex items-center justify-between gap-2">
        <span class="text-xs font-semibold text-black dark:text-slate-200">Depth</span>
        <span class="hud-chip text-[10px]">preview</span>
      </div>
      <div class="media-shell media-shell--quiet relative overflow-hidden rounded-[16px]">
        {#if depthSrc}
          <img
            class="aspect-square w-full object-cover"
            src={depthSrc}
            alt="Depth map"
            decoding="async"
          />
        {:else if !isLCMRunning}
          <img
            class="opacity-35 aspect-square w-full object-cover"
            src={placeholderSrc}
            alt=""
            decoding="async"
          />
          <span
            class="pointer-events-none absolute inset-0 flex items-center justify-center px-2 text-center text-[11px] text-black dark:text-slate-300"
            >离线示意</span
          >
        {:else}
          <div
            class="flex aspect-square max-h-[260px] min-h-[140px] items-center justify-center px-3 text-center text-xs text-black dark:text-slate-400"
          >
            等待 depth…
          </div>
        {/if}
      </div>
    </div>
    <div class="data-panel">
      <div class="mb-1.5 flex items-center justify-between gap-2">
        <span class="text-xs font-semibold text-black dark:text-slate-200">软描边</span>
        <span class="hud-chip text-[10px]">preview</span>
      </div>
      <div class="media-shell media-shell--quiet relative overflow-hidden rounded-[16px]">
        {#if softedgeSrc}
          <img
            class="aspect-square w-full object-cover"
            src={softedgeSrc}
            alt="Soft edge map"
            decoding="async"
          />
        {:else if !isLCMRunning}
          <img
            class="opacity-35 aspect-square w-full object-cover"
            src={placeholderSrc}
            alt=""
            decoding="async"
          />
          <span
            class="pointer-events-none absolute inset-0 flex items-center justify-center px-2 text-center text-[11px] text-black dark:text-slate-300"
            >离线示意</span
          >
        {:else}
          <div
            class="flex aspect-square max-h-[260px] min-h-[140px] items-center justify-center px-3 text-center text-xs text-black dark:text-slate-400"
          >
            未启用软描边或未就绪…
          </div>
        {/if}
      </div>
    </div>
    <div class="data-panel">
      <div class="mb-1.5 flex items-center justify-between gap-2">
        <span class="text-xs font-semibold text-black dark:text-slate-200">主体 Mask</span>
        <span class="hud-chip text-[10px]">{debugInfo?.subject_mask_backend ?? 'preview'}</span>
      </div>
      <div class="media-shell media-shell--quiet relative overflow-hidden rounded-[16px]">
        {#if subjectMaskSrc}
          <img
            class="aspect-square w-full object-cover"
            src={subjectMaskSrc}
            alt="Subject mask"
            decoding="async"
          />
        {:else if !isLCMRunning}
          <img
            class="opacity-35 aspect-square w-full object-cover"
            src={placeholderSrc}
            alt=""
            decoding="async"
          />
          <span
            class="pointer-events-none absolute inset-0 flex items-center justify-center px-2 text-center text-[11px] text-black dark:text-slate-300"
            >离线示意</span
          >
        {:else}
          <div
            class="flex aspect-square max-h-[260px] min-h-[140px] items-center justify-center px-3 text-center text-xs text-black dark:text-slate-400"
          >
            等待主体 mask…
          </div>
        {/if}
      </div>
    </div>
    <div class="data-panel">
      <div class="mb-1.5 flex items-center justify-between gap-2">
        <span class="text-xs font-semibold text-black dark:text-slate-200">Input（进 latent）</span>
        <span class="hud-chip text-[10px]">preview</span>
      </div>
      <div class="media-shell media-shell--quiet relative overflow-hidden rounded-[16px]">
        {#if inputSrc}
          <img
            class="aspect-square w-full object-cover"
            src={inputSrc}
            alt="风格化后送入模型的输入"
            decoding="async"
          />
        {:else if !isLCMRunning}
          <img
            class="opacity-35 aspect-square w-full object-cover"
            src={placeholderSrc}
            alt=""
            decoding="async"
          />
          <span
            class="pointer-events-none absolute inset-0 flex items-center justify-center px-2 text-center text-[11px] text-black dark:text-slate-300"
            >离线示意</span
          >
        {:else}
          <div
            class="flex aspect-square max-h-[260px] min-h-[140px] items-center justify-center px-3 text-center text-xs text-black dark:text-slate-400"
          >
            等待输入帧…
          </div>
        {/if}
      </div>
    </div>
    <div class="data-panel text-sm">
      {#if debugInfo}
        <div class="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {#each metrics as metric}
            <div class="metric-card py-3">
              <p class="text-[10px] uppercase tracking-[0.18em] text-black dark:text-slate-400">
                {metric.label}
              </p>
              <p class="mt-1 text-sm font-semibold text-black dark:text-slate-200">
                {metric.value}
              </p>
            </div>
          {/each}
        </div>
      {:else}
        <div
          class="flex min-h-[120px] items-center justify-center text-center text-sm text-black dark:text-slate-400"
        >
          {#if !isLCMRunning}
            启动推理后显示指标
          {:else}
            等待调试数据…
          {/if}
        </div>
      {/if}
    </div>
  </div>
</div>
