<script lang="ts">
  import { fly, scale } from 'svelte/transition';

  export let message: string = '';

  let timeout = 0;
  $: if (message !== '') {
    clearTimeout(timeout);
    timeout = setTimeout(() => {
      message = '';
    }, 5000);
  }
</script>

{#if message}
  <button
    class="fixed right-5 top-5 z-50 text-left"
    type="button"
    on:click={() => (message = '')}
    in:fly={{ x: 24, y: -8, duration: 220 }}
    out:scale={{ start: 0.92, duration: 160 }}
  >
    <div
      class="glass-panel glass-panel-strong relative min-w-[300px] max-w-md overflow-hidden rounded-[24px] px-5 py-4 text-sm font-medium text-red-900 shadow-[0_24px_70px_rgba(239,68,68,0.16)] dark:text-red-100"
    >
      <div class="warning-progress"></div>
      <div class="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-red-400/80 to-transparent"></div>
      <div class="absolute -right-10 top-0 h-24 w-24 rounded-full bg-red-400/20 blur-3xl"></div>
      <div class="relative flex items-start gap-3">
        <span class="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-red-500/10 text-red-500 shadow-[0_0_24px_rgba(248,113,113,0.22)]">
          !
        </span>
        <div class="flex-1">
          <p class="text-xs font-semibold uppercase tracking-[0.28em] text-red-800 dark:text-red-200">
            Notice
          </p>
          <p class="mt-1 leading-6">{message}</p>
        </div>
      </div>
    </div>
  </button>
{/if}

<style lang="postcss">
  .warning-progress {
    position: absolute;
    inset-inline: 0;
    bottom: 0;
    height: 3px;
    transform-origin: left;
    background: linear-gradient(90deg, rgba(248, 113, 113, 0.82), rgba(251, 191, 36, 0.72));
    animation: shrink-x 5s linear forwards;
  }
</style>
