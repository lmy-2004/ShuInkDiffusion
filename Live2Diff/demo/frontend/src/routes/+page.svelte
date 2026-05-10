<script lang="ts">
  import gsap from 'gsap';
  import BackgroundFX from '$lib/components/BackgroundFX.svelte';
  import BenchmarkPanel from '$lib/components/BenchmarkPanel.svelte';
  import Button from '$lib/components/Button.svelte';
  import DebugPanel from '$lib/components/DebugPanel.svelte';
  import StylizationPanel from '$lib/components/StylizationPanel.svelte';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import { lcmLiveActions, lcmLiveStatus, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, mediaStreamStatus, MediaStreamStatusEnum, onFrameChangeStore } from '$lib/mediaStream';
  import { deboucedPipelineValues, getPipelineValues } from '$lib/store';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { InputSource, PipelineMode } from '$lib/types';
  import { onMount, tick } from 'svelte';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  let inputSource: InputSource = InputSource.CAMERA;
  let hasVideoLoaded: boolean = false;
  let heroSectionEl: HTMLElement;
  let queuePanelEl: HTMLElement;
  let inputPanelEl: HTMLElement;
  let outputPanelEl: HTMLElement;
  let controlsPanelEl: HTMLElement;
  let queueSizeBadgeEl: HTMLElement;
  let workspaceAnimated = false;
  let lastQueueSize = -1;

  onMount(() => {
    getSettings();
    requestAnimationFrame(() => {
      const heroTargets = [heroSectionEl, queuePanelEl].filter(Boolean);
      if (heroTargets.length) {
        gsap.fromTo(
          heroTargets,
          { autoAlpha: 0, y: 28 },
          { autoAlpha: 1, y: 0, duration: 0.82, stagger: 0.12, ease: 'power3.out' }
        );
      }
    });
  });

  async function getSettings() {
    const settings = await fetch('/api/settings').then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    toggleQueueChecker(true);
  }
  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch('/api/queue').then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }

  function getSreamdata() {
    if (isImageMode) {
      return [getPipelineValues(), $onFrameChangeStore?.blob];
    } else {
      return [$deboucedPipelineValues];
    }
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
  }
  $: inputModeLabel = inputSource === InputSource.CAMERA ? 'camera' : 'video';
  $: sessionLabel = isLCMRunning ? 'running' : disabled ? 'arming' : 'standby';
  $: sourceReady = inputSource === InputSource.CAMERA ? $mediaStreamStatus === MediaStreamStatusEnum.CONNECTED : hasVideoLoaded;
  $: if (pipelineParams && !workspaceAnimated) {
    void revealWorkspace();
  }
  $: if (queueSizeBadgeEl && currentQueueSize !== lastQueueSize) {
    lastQueueSize = currentQueueSize;
    gsap.fromTo(queueSizeBadgeEl, { scale: 1.14 }, { scale: 1, duration: 0.42, ease: 'power2.out' });
  }
  let disabled = false;
  let benchmarkRunning = false;

  async function revealWorkspace() {
    await tick();
    const panels = [inputPanelEl, outputPanelEl, controlsPanelEl].filter(Boolean);
    if (!panels.length || workspaceAnimated) {
      return;
    }
    gsap.set(panels, { autoAlpha: 0, y: 26, scale: 0.985 });
    gsap.to(panels, {
      autoAlpha: 1,
      y: 0,
      scale: 1,
      duration: 0.72,
      stagger: 0.08,
      ease: 'power3.out'
    });
    workspaceAnimated = true;
  }

  async function toggleLcmLive() {
    try {
      if (!isLCMRunning) {
        if (isImageMode) {
          if (
            inputSource === InputSource.CAMERA &&
            $mediaStreamStatus !== MediaStreamStatusEnum.CONNECTED
          ) {
            warningMessage = 'Please allow camera access.';
            return;
          }
          if (inputSource === InputSource.VIDEO && !hasVideoLoaded) {
            warningMessage = 'Please load a video first.';
            return;
          }
        }
        disabled = true;
        await lcmLiveActions.start(getSreamdata);
        disabled = false;
        toggleQueueChecker(false);
      } else {
        // if (isImageMode) {
        //   mediaStreamActions.stop();
        // }
        lcmLiveActions.stop();
        mediaStreamActions.stop()
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      console.error(warningMessage)
      disabled = false;
      toggleQueueChecker(true);
    }
  }
</script>

<svelte:head>
  <title>real-time-ink-diffusion</title>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<div class="aurora-backdrop relative isolate min-h-screen overflow-hidden">
  <BackgroundFX />
  <main class="relative z-10 mx-auto flex min-h-screen w-full max-w-[1800px] flex-col gap-6 px-5 py-8 lg:px-8 lg:py-10 2xl:px-10">
    <Warning bind:message={warningMessage}></Warning>
    <section
      bind:this={heroSectionEl}
      class="glass-panel glass-panel-strong surface-noise panel-glow rounded-[32px] px-8 py-8 lg:px-10 lg:py-10"
    >
      <div
        class="light-surface relative overflow-hidden rounded-[28px] border border-white/30 bg-white/10 px-6 py-7 dark:border-white/10 dark:bg-slate-950/10"
      >
        <div class="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-400/80 to-transparent"></div>
        <div class="absolute left-[-5rem] top-[-7rem] h-48 w-48 rounded-full bg-cyan-400/20 blur-3xl"></div>
        <div class="absolute right-[-4rem] top-10 h-44 w-44 rounded-full bg-fuchsia-500/15 blur-3xl"></div>
        <div class="relative flex flex-col gap-5 text-center">
          <div class="flex justify-center">
            <p class="hero-kicker text-black dark:text-slate-200">
              <span class="inline-block h-2 w-2 rounded-full bg-cyan-400 shadow-[0_0_18px_rgba(34,211,238,0.95)]"></span>
              Live stylization
            </p>
          </div>
          <div class="space-y-4">
            <h1 class="hero-title text-4xl font-semibold tracking-tight lg:text-6xl">
              real-time-ink-diffusion
            </h1>
            <p class="mx-auto max-w-3xl text-base leading-7 text-black dark:text-slate-300 lg:text-lg">
              实时接入摄像头或本地视频，将当前画面持续送入 AI 风格化重绘管线。
            </p>
          </div>
          <div class="flex flex-wrap items-center justify-center gap-3">
            <span class="hud-chip">
              <span class="status-dot {isLCMRunning ? 'status-dot-live' : ''}"></span>
              Session {sessionLabel}
            </span>
            <span class="hud-chip">
              <span class="status-dot {sourceReady ? 'status-dot-live' : 'status-dot-warn'}"></span>
              Source {inputModeLabel}
            </span>
            <span class="hud-chip">
              <span class="status-dot {maxQueueSize > 0 ? 'status-dot-live' : ''}"></span>
              Queue cap {maxQueueSize || 'off'}
            </span>
          </div>
          <div class="flex flex-wrap items-center justify-center gap-3 text-sm text-black dark:text-slate-300">
            <span class="cyber-pill">
              <span class="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_14px_rgba(74,222,128,0.9)]"></span>
              Realtime pipeline
            </span>
            <span class="cyber-pill">
              <span class="h-2 w-2 rounded-full bg-violet-400 shadow-[0_0_14px_rgba(167,139,250,0.9)]"></span>
              Camera or local video
            </span>
            <span class="cyber-pill">
              <span class="h-2 w-2 rounded-full bg-sky-400 shadow-[0_0_14px_rgba(56,189,248,0.9)]"></span>
              Live AI stylization
            </span>
          </div>
        </div>
      </div>
      {#if maxQueueSize > 0}
        <div
          bind:this={queuePanelEl}
          class="glass-panel mt-6 rounded-[24px] px-5 py-4 text-sm leading-6 text-amber-950 dark:text-amber-100"
        >
          <p class="pr-6">
            There are
            <span
              id="queue_size"
              bind:this={queueSizeBadgeEl}
              class="mx-1 inline-flex min-w-[2.25rem] items-center justify-center rounded-full border border-amber-400/30 bg-amber-100/80 px-2 py-0.5 font-bold text-amber-950 shadow-[0_0_20px_rgba(251,191,36,0.18)] dark:bg-amber-400/10 dark:text-amber-50"
            >
              {currentQueueSize}
            </span>
            user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
            <a
              href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
              target="_blank"
              class="font-semibold underline decoration-amber-400/60 underline-offset-4 transition hover:text-amber-700 hover:decoration-amber-600 dark:hover:text-amber-200">Duplicate</a
            >
            and run it on your own GPU.
          </p>
        </div>
      {/if}
    </section>
    {#if pipelineParams}
      <article class="grid grid-cols-1 gap-6">
        <div class="grid gap-6">
          <div class="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <section
              bind:this={inputPanelEl}
              class="glass-panel surface-noise panel-glow hover-lift flex h-full flex-col gap-5 rounded-[30px] p-6 lg:p-7"
            >
            <div class="flex items-center justify-between gap-4">
              <h2 class="text-2xl font-semibold text-black dark:text-slate-200">Input</h2>
              {#if isImageMode && !isLCMRunning}
                <div class="data-panel flex items-center gap-3 px-4 py-3">
                  <label class="text-sm font-medium text-black dark:text-slate-300" for="input-source">
                    Source
                  </label>
                  <select
                    id="input-source"
                    bind:value={inputSource}
                    disabled={benchmarkRunning}
                    class="input-shell px-4 py-2 font-medium"
                  >
                    <option value={InputSource.CAMERA}>Camera</option>
                    <option value={InputSource.VIDEO}>Video</option>
                  </select>
                </div>
              {/if}
            </div>
            {#if isImageMode}
              <VideoInput
                width={Number(pipelineParams.width.default)}
                height={Number(pipelineParams.height.default)}
                bind:source={inputSource}
                bind:hasVideoLoaded
                disabled={isLCMRunning}
                locked={benchmarkRunning}
              ></VideoInput>
            {/if}
            </section>

            <section
              bind:this={outputPanelEl}
              class="glass-panel surface-noise panel-glow hover-lift flex h-full flex-col gap-5 rounded-[30px] p-6 lg:p-7"
            >
            <div class="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <h2 class="text-2xl font-semibold text-black dark:text-slate-200">Output</h2>
              <Button
                on:click={toggleLcmLive}
                disabled={disabled || benchmarkRunning}
                active={isLCMRunning}
                classList={'min-w-[180px] px-6 py-3 text-lg font-semibold shadow-lg shadow-slate-900/10'}
              >
                {#if isLCMRunning}
                  Stop
                {:else}
                  Start
                {/if}
              </Button>
            </div>
            <ImagePlayer />
            </section>
          </div>

          <section
            bind:this={controlsPanelEl}
            class="glass-panel surface-noise panel-glow hover-lift flex flex-col gap-6 rounded-[30px] p-6 lg:p-7"
          >
            <PipelineOptions {pipelineParams} disabled={benchmarkRunning}></PipelineOptions>
            <div class="h-px w-full bg-gradient-to-r from-transparent via-slate-300/80 to-transparent dark:via-slate-600/60"></div>
            <DebugPanel />
            <div class="h-px w-full bg-gradient-to-r from-transparent via-slate-300/80 to-transparent dark:via-slate-600/60"></div>
            <StylizationPanel {pipelineParams} disabled={benchmarkRunning} />
            <div class="h-px w-full bg-gradient-to-r from-transparent via-slate-300/80 to-transparent dark:via-slate-600/60"></div>
            <BenchmarkPanel
              {pipelineParams}
              {inputSource}
              {isImageMode}
              bind:running={benchmarkRunning}
            />
          </section>
        </div>
      </article>
    {:else}
      <section class="glass-panel glass-panel-strong surface-noise rounded-[32px] p-8 lg:p-10">
        <div class="grid gap-6">
          <div class="grid gap-6 lg:grid-cols-2">
            <div class="loading-shimmer h-[420px] rounded-[28px] border border-white/20"></div>
            <div class="loading-shimmer h-[420px] rounded-[28px] border border-white/20"></div>
          </div>
          <div class="loading-shimmer h-[320px] rounded-[28px] border border-white/20"></div>
        </div>
        <div class="mt-8 flex items-center justify-center gap-4 text-lg text-black dark:text-slate-300">
          <Spinner classList={'animate-spin text-black opacity-80 dark:text-slate-300'}></Spinner>
          <div>
            <p class="font-semibold text-black dark:text-slate-200">Preparing realtime workspace</p>
            <p class="text-sm text-black dark:text-slate-300">Loading pipeline metadata and interface controls...</p>
          </div>
        </div>
      </section>
    {/if}
  </main>
</div>

<style lang="postcss">
  :global(html) {
    color: #000000;
  }

  :global(html.dark) {
    color: #e2e8f0;
  }

  :global(body) {
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    background:
      radial-gradient(circle at top, rgba(59, 130, 246, 0.16), transparent 30%),
      radial-gradient(circle at 20% 20%, rgba(34, 211, 238, 0.12), transparent 28%),
      linear-gradient(180deg, #f8fafc 0%, #eef2ff 48%, #f8fafc 100%);
  }

  :global(html.dark body) {
    background:
      radial-gradient(circle at top, rgba(59, 130, 246, 0.2), transparent 28%),
      radial-gradient(circle at 78% 8%, rgba(168, 85, 247, 0.14), transparent 24%),
      linear-gradient(180deg, #020617 0%, #0f172a 44%, #020617 100%);
  }
</style>
