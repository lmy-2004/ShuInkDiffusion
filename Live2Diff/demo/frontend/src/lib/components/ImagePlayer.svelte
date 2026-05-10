<script lang="ts">
  import { fade } from 'svelte/transition';
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  let imageEl: HTMLImageElement;

  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }
</script>

<div class="media-shell {isLCMRunning && $streamId ? 'media-shell--active' : ''} aspect-square w-full">
  <div class="section-wire"></div>
  <div class="absolute left-4 top-4 z-20 flex items-center gap-2">
    <span class="hud-chip">
      <span class="status-dot {isLCMRunning && $streamId ? 'status-dot-live' : ''}"></span>
      {#if isLCMRunning && $streamId}
        live stream
      {:else if isLCMRunning}
        warming up
      {:else}
        standby
      {/if}
    </span>
  </div>
  <div class="absolute right-4 top-4 z-20">
    <span class="hud-chip">{getPipelineValues()?.width ?? 512} px</span>
  </div>
  <div class="relative z-20 aspect-square w-full object-cover">
    <!-- svelte-ignore a11y-missing-attribute -->
    {#if isLCMRunning && $streamId}
      <img
        bind:this={imageEl}
        in:fade={{ duration: 260 }}
        class="aspect-square h-full w-full rounded-[26px] object-cover"
        src={'/api/stream/' + $streamId}
      />
      <div class="absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-slate-950/45 to-transparent"></div>
      <div class="absolute bottom-4 right-4 z-20">
        <Button
          on:click={takeSnapshot}
          disabled={!isLCMRunning}
          active={isLCMRunning}
          title={'Take Snapshot'}
          classList={'ml-auto rounded-2xl px-3 py-2 text-sm shadow-lg shadow-slate-900/20'}
        >
          <Floppy classList={''} />
        </Button>
      </div>
    {:else}
      <div
        in:fade={{ duration: 220 }}
        class="light-surface absolute inset-0 flex flex-col items-center justify-center gap-4 bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.14),transparent_35%),linear-gradient(180deg,rgba(255,255,255,0.36),rgba(226,232,240,0.2))] px-8 text-center dark:bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.12),transparent_35%),linear-gradient(180deg,rgba(15,23,42,0.72),rgba(15,23,42,0.4))]"
      >
        <div class="h-24 w-24 rounded-full border border-cyan-300/30 bg-cyan-400/10 shadow-[0_0_44px_rgba(34,211,238,0.18)]"></div>
        <div class="space-y-2">
          <p class="text-lg font-semibold text-black dark:text-slate-200">等待输出流启动</p>
          <p class="text-sm leading-6 text-black dark:text-slate-400">
            推理开始后，结果会以实时流形式投射到这里。
          </p>
        </div>
      </div>
      <img
        class="aspect-square h-full w-full rounded-[26px] bg-slate-200 object-cover opacity-0 dark:bg-slate-800"
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
      />
    {/if}
  </div>
  {#if isLCMRunning && !$streamId}
    <div class="media-overlay z-20" in:fade={{ duration: 180 }}>
      <span class="hud-chip">warming up pipeline</span>
      <p class="media-overlay-title mt-5">接收第一帧输出中</p>
      <p class="media-overlay-subtitle">模型正在建立实时输出通道。</p>
    </div>
  {/if}
  <div class="pointer-events-none absolute inset-x-0 bottom-0 z-[1] h-24 bg-gradient-to-t from-slate-950/18 to-transparent"></div>
  <div class="pointer-events-none absolute inset-0 z-[1] opacity-25 [background-image:linear-gradient(to_right,rgba(255,255,255,0.06)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px]"></div>
</div>
