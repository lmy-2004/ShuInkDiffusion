<script lang="ts">
  import 'rvfc-polyfill';

  import { fade } from 'svelte/transition';
  import { onDestroy, onMount } from 'svelte';
  import {
    MediaStreamStatusEnum,
    mediaDevices,
    mediaStream,
    mediaStreamActions,
    mediaStreamStatus,
    onFrameChangeStore
  } from '$lib/mediaStream';
  import { streamId } from '$lib/lcmLive';
  import { InputSource } from '$lib/types';
  import { downloadBlob, sanitizeFileStem } from '$lib/utils';
  import { videoControllerActions } from '$lib/videoController';
  import Button from './Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import MediaListSwitcher from './MediaListSwitcher.svelte';
  export let width = 512;
  export let height = 512;
  export let source: InputSource = InputSource.CAMERA;
  export let disabled: boolean = false;
  export let locked: boolean = false;
  export let hasVideoLoaded: boolean = false;
  const size = { width, height };

  let videoEl: HTMLVideoElement;
  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let fileInputEl: HTMLInputElement;
  let videoFrameCallbackId: number | null = null;
  let isWebCamActive = false;
  let isFrameLoopRunning = false;
  let videoIsReady = false;
  let objectUrl: string | null = null;
  let dragActive = false;
  let videoDuration = 0;
  let videoProgress = 0;
  let lastSource: InputSource = source;
  let lastMillis = 0;
  let selectedVideoFileName = '';

  const THROTTLE = 1000 / 120;
  $: interactionDisabled = disabled || locked;
  $: previewReady =
    videoIsReady && ((source === InputSource.CAMERA && isWebCamActive) || (source === InputSource.VIDEO && hasVideoLoaded));
  $: currentLabel = source === InputSource.CAMERA ? 'camera feed' : 'local video';
  $: liveLabel = source === InputSource.CAMERA ? (isWebCamActive ? 'capturing' : 'camera idle') : hasVideoLoaded ? 'clip loaded' : 'awaiting clip';
  $: progressPercent = Math.max(0, Math.min(100, videoProgress));
  $: videoControllerActions.update({
    source,
    hasVideoLoaded,
    videoIsReady,
    currentTime: videoEl?.currentTime ?? 0,
    duration: videoDuration,
    selectedVideoFileName
  });

  onMount(() => {
    ctx = canvasEl.getContext('2d') as CanvasRenderingContext2D;
    canvasEl.width = size.width;
    canvasEl.height = size.height;
    videoControllerActions.register({
      waitForReady: waitForVideoReady,
      seek: seekVideo,
      jumpToStart,
      play: playVideo,
      pause: pauseVideo
    });
  });

  onDestroy(() => {
    stopFrameLoop();
    clearObjectUrl();
    videoControllerActions.unregister();
  });

  $: if (videoEl) {
    if (source === InputSource.CAMERA) {
      videoEl.pause();
      videoEl.removeAttribute('src');
      videoEl.load();
      videoEl.srcObject = $mediaStream;
    } else {
      videoEl.srcObject = null;
      if (objectUrl && videoEl.src !== objectUrl) {
        videoEl.src = objectUrl;
        void videoEl.play().catch(() => undefined);
      }
    }
  }

  $: if (source !== lastSource) {
    videoIsReady = false;
    stopFrameLoop();
    onFrameChangeStore.set({ blob: new Blob() });
    if (source === InputSource.VIDEO && $mediaStreamStatus === MediaStreamStatusEnum.CONNECTED) {
      void mediaStreamActions.stop();
    }
    if (source === InputSource.VIDEO && objectUrl && videoEl) {
      videoEl.srcObject = null;
      videoEl.src = objectUrl;
      void videoEl.play().catch(() => undefined);
    }
    lastSource = source;
  }

  $: isWebCamActive = source === InputSource.CAMERA && $mediaStreamStatus === MediaStreamStatusEnum.CONNECTED;

  $: if (videoEl && videoIsReady && ((source === InputSource.CAMERA && isWebCamActive) || (source === InputSource.VIDEO && hasVideoLoaded))) {
    startFrameLoop();
  } else {
    stopFrameLoop();
  }

  function startFrameLoop() {
    if (isFrameLoopRunning || !videoEl) {
      return;
    }
    isFrameLoopRunning = true;
    lastMillis = 0;
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  function stopFrameLoop() {
    if (!videoEl || videoFrameCallbackId === null) {
      isFrameLoopRunning = false;
      videoFrameCallbackId = null;
      return;
    }
    videoEl.cancelVideoFrameCallback(videoFrameCallbackId);
    isFrameLoopRunning = false;
    videoFrameCallbackId = null;
  }

  function clearObjectUrl() {
    if (objectUrl) {
      URL.revokeObjectURL(objectUrl);
      objectUrl = null;
    }
  }

  async function onFrameChange(now: DOMHighResTimeStamp) {
    if (!isFrameLoopRunning) {
      return;
    }
    if (now - lastMillis < THROTTLE) {
      videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
      return;
    }
    const videoWidth = videoEl.videoWidth;
    const videoHeight = videoEl.videoHeight;
    let height0 = videoHeight;
    let width0 = videoWidth;
    let x0 = 0;
    let y0 = 0;
    if (videoWidth > videoHeight) {
      width0 = videoHeight;
      x0 = (videoWidth - videoHeight) / 2;
    } else {
      height0 = videoWidth;
      y0 = (videoHeight - videoWidth) / 2;
    }
    ctx.drawImage(videoEl, x0, y0, width0, height0, 0, 0, size.width, size.height);
    const blob = await new Promise<Blob>((resolve) => {
      canvasEl.toBlob(
        (blob) => {
          resolve(blob as Blob);
        },
        'image/jpeg',
        1
      );
    });
    onFrameChangeStore.set({ blob });
    videoFrameCallbackId = videoEl.requestVideoFrameCallback(onFrameChange);
  }

  async function waitForVideoReady(timeoutMs: number = 15000) {
    if (source !== InputSource.VIDEO || !hasVideoLoaded || !videoEl) {
      throw new Error('请先加载本地视频');
    }
    if (videoIsReady && videoEl.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => {
        cleanup();
        reject(new Error('视频未就绪'));
      }, timeoutMs);
      const done = () => {
        cleanup();
        resolve();
      };
      const cleanup = () => {
        window.clearTimeout(timer);
        videoEl?.removeEventListener('loadeddata', done);
        videoEl?.removeEventListener('canplay', done);
      };
      videoEl.addEventListener('loadeddata', done, { once: true });
      videoEl.addEventListener('canplay', done, { once: true });
    });
  }

  async function playVideo() {
    await waitForVideoReady();
    await videoEl.play().catch((err) => {
      throw err instanceof Error ? err : new Error('视频播放失败');
    });
  }

  function pauseVideo() {
    videoEl?.pause();
  }

  async function seekVideo(seconds: number) {
    await waitForVideoReady();
    const target = Math.max(0, Math.min(seconds, videoDuration || seconds));
    if (Math.abs((videoEl?.currentTime ?? 0) - target) < 0.02) {
      videoProgress = videoDuration > 0 ? (target / videoDuration) * 100 : 0;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      if (!videoEl) {
        reject(new Error('视频未就绪'));
        return;
      }
      const timer = window.setTimeout(() => {
        cleanup();
        reject(new Error('视频 seek 超时'));
      }, 8000);
      const done = () => {
        cleanup();
        resolve();
      };
      const cleanup = () => {
        window.clearTimeout(timer);
        videoEl.removeEventListener('seeked', done);
      };
      videoEl.addEventListener('seeked', done, { once: true });
      videoEl.currentTime = target;
    });
    videoProgress = videoDuration > 0 ? (target / videoDuration) * 100 : 0;
  }

  async function jumpToStart() {
    await waitForVideoReady();
    if (!videoEl) {
      throw new Error('视频未就绪');
    }
    videoEl.currentTime = 0;
    videoProgress = 0;
    await new Promise<void>((resolve) => {
      requestAnimationFrame(() => resolve());
    });
  }

  async function startWebCam() {
    await mediaStreamActions.enumerateDevices();
    await mediaStreamActions.start();
  }

  function stopWebCam() {
    mediaStreamActions.stop();
  }

  async function toggleWebCam() {
    if (interactionDisabled) {
      return;
    }
    if (isWebCamActive) {
      stopWebCam();
    } else {
      await startWebCam();
    }
  }

  function extFromBlobType(type: string): string {
    if (type === 'image/png') return 'png';
    if (type === 'image/webp') return 'webp';
    return 'jpg';
  }

  async function saveInputSnapshots() {
    if (!previewReady || !canvasEl || interactionDisabled) {
      return;
    }
    const stem =
      source === InputSource.VIDEO
        ? sanitizeFileStem(selectedVideoFileName || 'video')
        : 'camera';
    const progressPart =
      source === InputSource.VIDEO
        ? `_${Math.floor(videoEl?.currentTime ?? 0)}s`
        : `_${Date.now()}`;
    const base = `${stem}${progressPart}`;
    const videoBlob = await new Promise<Blob | null>((resolve) => {
      canvasEl.toBlob((b) => resolve(b), 'image/jpeg', 0.95);
    });
    if (videoBlob) {
      downloadBlob(videoBlob, `${base}_video.jpg`);
    }
    const sid = $streamId;
    if (!sid) {
      return;
    }
    try {
      const debugRes = await fetch(`/api/debug/${sid}`, { cache: 'no-store' });
      if (!debugRes.ok) {
        return;
      }
      const data = (await debugRes.json()) as {
        has_depth?: boolean;
        has_softedge_preview?: boolean;
      };
      if (data.has_depth) {
        const depthRes = await fetch(`/api/depth/${sid}?t=${Date.now()}`);
        if (depthRes.ok) {
          const blob = await depthRes.blob();
          downloadBlob(blob, `${base}_depth.${extFromBlobType(blob.type)}`);
        }
      }
      if (data.has_softedge_preview) {
        const edgeRes = await fetch(`/api/softedge/${sid}?t=${Date.now()}`);
        if (edgeRes.ok) {
          const blob = await edgeRes.blob();
          downloadBlob(blob, `${base}_softedge.${extFromBlobType(blob.type)}`);
        }
      }
    } catch {
      /* ignore */
    }
  }

  async function loadVideo(file: File) {
    if (interactionDisabled || !file.type.startsWith('video/')) {
      return;
    }
    clearObjectUrl();
    selectedVideoFileName = file.name;
    objectUrl = URL.createObjectURL(file);
    hasVideoLoaded = true;
    videoIsReady = false;
    videoDuration = 0;
    videoProgress = 0;
    if ($mediaStreamStatus === MediaStreamStatusEnum.CONNECTED) {
      await mediaStreamActions.stop();
    }
    if (videoEl) {
      videoEl.srcObject = null;
      videoEl.src = objectUrl;
      videoEl.currentTime = 0;
      await videoEl.play().catch(() => undefined);
    }
  }

  function onFileChange(event: Event) {
    const file = (event.currentTarget as HTMLInputElement).files?.[0];
    if (file) {
      void loadVideo(file);
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    dragActive = false;
    if (interactionDisabled) {
      return;
    }
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      void loadVideo(file);
    }
  }

  function onSeek(event: Event) {
    if (interactionDisabled) {
      return;
    }
    const progress = Number((event.currentTarget as HTMLInputElement).value);
    videoProgress = progress;
    if (videoEl && videoDuration > 0) {
      videoEl.currentTime = (videoDuration * progress) / 100;
    }
  }

  function formatTime(value: number) {
    if (!Number.isFinite(value)) {
      return '00:00';
    }
    const totalSeconds = Math.floor(value);
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    return `${minutes}:${seconds}`;
  }
</script>

<div
  class="media-shell {previewReady ? 'media-shell--active' : ''} {dragActive ? 'media-shell--drag' : ''} w-full"
  role="group"
  aria-label="video input"
  on:dragenter|preventDefault={() => {
    if (source === InputSource.VIDEO && !interactionDisabled) dragActive = true;
  }}
  on:dragover|preventDefault={() => {
    if (source === InputSource.VIDEO && !interactionDisabled) dragActive = true;
  }}
  on:dragleave|preventDefault={() => {
    dragActive = false;
  }}
  on:drop={onDrop}
>
  <div class="section-wire"></div>
  <div class="absolute left-4 top-4 z-20 flex flex-wrap gap-2">
    <span class="hud-chip">
      <span class="status-dot {previewReady ? 'status-dot-live' : ''}"></span>
      {liveLabel}
    </span>
    <span class="hud-chip">{currentLabel}</span>
  </div>
  <div class="relative z-20 aspect-square w-full object-cover">
    {#if source === InputSource.CAMERA && isWebCamActive && $mediaDevices.length > 0}
      <div class="absolute right-4 top-16 z-20">
        <MediaListSwitcher />
      </div>
    {/if}
    <video
      class="pointer-events-none aspect-square h-full w-full object-cover transition-opacity duration-300 {previewReady ? 'opacity-100' : 'opacity-0'}"
      bind:this={videoEl}
      on:loadedmetadata={() => {
        videoDuration = Number.isFinite(videoEl.duration) ? videoEl.duration : 0;
      }}
      on:loadeddata={() => {
        videoIsReady = true;
        if (source === InputSource.VIDEO) {
          void videoEl.play().catch(() => undefined);
        }
      }}
      on:timeupdate={() => {
        if (videoDuration > 0) {
          videoProgress = (videoEl.currentTime / videoDuration) * 100;
        }
      }}
      playsinline
      autoplay
      muted
      loop
    ></video>
    <canvas
      bind:this={canvasEl}
      class="absolute left-0 top-0 aspect-square h-full w-full object-cover transition-opacity duration-300 {previewReady ? 'opacity-100' : 'opacity-0'}"
    ></canvas>
    <div class="pointer-events-none absolute inset-0 z-[1] opacity-20 [background-image:linear-gradient(to_right,rgba(255,255,255,0.06)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.06)_1px,transparent_1px)] [background-size:24px_24px]"></div>
    <div class="pointer-events-none absolute inset-x-0 bottom-0 z-[1] h-28 bg-gradient-to-t from-slate-950/40 to-transparent"></div>
    {#if previewReady}
      <div class="absolute bottom-4 left-4 z-20">
        <span class="hud-chip">
          <span class="status-dot status-dot-live"></span>
          live signal locked
        </span>
      </div>
      <div class="absolute bottom-4 right-4 z-20">
        <Button
          on:click={() => void saveInputSnapshots()}
          disabled={!previewReady || interactionDisabled}
          active={previewReady}
          title={'Save input snapshots'}
          classList={'rounded-2xl px-3 py-2 text-sm shadow-lg shadow-slate-900/20'}
        >
          <Floppy classList={''} />
        </Button>
      </div>
    {/if}
  </div>
  {#if source === InputSource.CAMERA && !isWebCamActive}
    <div class="media-overlay" in:fade={{ duration: 180 }}>
      <span class="hud-chip">camera required</span>
      <p class="media-overlay-title mt-5">启动摄像头采集实时输入</p>
      <p class="media-overlay-subtitle">摄像头连接后，当前画面会持续采样送入推理管线。</p>
    </div>
  {/if}
  {#if source === InputSource.VIDEO && !hasVideoLoaded}
    <button
      type="button"
      class="media-overlay gap-3"
      disabled={interactionDisabled}
      on:click={() => {
        if (!interactionDisabled) fileInputEl.click();
      }}
    >
      <span class="hud-chip">
        {dragActive ? 'drop to import' : 'video ingest'}
      </span>
      <span class="media-overlay-title">{dragActive ? '松开导入视频' : '拖入视频'}</span>
      <span class="media-overlay-subtitle">或点击选择本地视频，导入后自动循环播放。</span>
    </button>
  {/if}
</div>

<input
  bind:this={fileInputEl}
  class="hidden"
  type="file"
  accept="video/*"
  disabled={interactionDisabled}
  on:change={onFileChange}
/>

{#if source === InputSource.CAMERA}
  <Button
    on:click={toggleWebCam}
    disabled={interactionDisabled}
    active={isWebCamActive}
    classList={'mt-5 w-full px-5 py-3 text-base font-semibold'}
  >
    {#if isWebCamActive}
      <span>Stop WebCam</span>
    {:else}
      <span>Start WebCam</span>
    {/if}
  </Button>
{:else}
  <div class="mt-5 flex flex-col gap-3">
    <Button
      on:click={() => {
        if (!interactionDisabled) fileInputEl.click();
      }}
      disabled={interactionDisabled}
      active={hasVideoLoaded}
      classList={'w-full px-5 py-3 text-base font-semibold'}
    >
      {#if hasVideoLoaded}
        <span>Replace Video</span>
      {:else}
        <span>Select Video</span>
      {/if}
    </Button>
    {#if hasVideoLoaded}
      <div class="data-panel flex items-center gap-3 text-sm">
        <span class="w-12 text-right tabular-nums text-black dark:text-slate-400">
          {formatTime(videoEl?.currentTime ?? 0)}
        </span>
        <input
          class="range-shell h-2.5 w-full cursor-pointer appearance-none rounded-full"
          style={`--range-progress: ${progressPercent}%`}
          type="range"
          min="0"
          max="100"
          step="0.1"
          disabled={interactionDisabled}
          value={videoProgress}
          on:input={onSeek}
        />
        <span class="w-12 tabular-nums text-black dark:text-slate-400">{formatTime(videoDuration)}</span>
      </div>
    {/if}
  </div>
{/if}
