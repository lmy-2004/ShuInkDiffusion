<script lang="ts">
  import { mediaDevices, mediaStreamActions } from '$lib/mediaStream';
  import Screen from '$lib/icons/screen.svelte';
  import { onMount } from 'svelte';

  let deviceId: string = '';
  onMount(() => {
    deviceId = $mediaDevices[0]?.deviceId ?? '';
  });
</script>

<div class="flex items-center justify-center gap-2 text-xs">
  <button
    title="Share your screen"
    class="hud-chip my-1 cursor-pointer border-white/30 text-black transition hover:-translate-y-0.5 hover:border-cyan-300/40 hover:text-black dark:text-slate-200"
    on:click={() => mediaStreamActions.startScreenCapture()}
  >
    <span>Share</span>

    <Screen classList={''} />
  </button>
  {#if $mediaDevices}
    <select
      bind:value={deviceId}
      on:change={() => mediaStreamActions.switchCamera(deviceId)}
      id="devices-list"
      class="input-shell block cursor-pointer px-3 py-2 font-medium"
    >
      {#each $mediaDevices as device, i}
        <option value={device.deviceId}>{device.label}</option>
      {/each}
    </select>
  {/if}
</div>
