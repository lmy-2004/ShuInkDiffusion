<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = 8.0;
  export let params: FieldProps;
  export let disabled: boolean = false;
  /** 单行、低高度，用于风格化等密集参数区 */
  export let compact: boolean = false;

  onMount(() => {
    value = Number(params?.default) ?? 8.0;
  });

  $: min = Number(params?.min ?? 0);
  $: max = Number(params?.max ?? 100);
  $: progressPercent = max > min ? ((Number(value) - min) / (max - min)) * 100 : 0;
</script>

{#if compact}
  <div
    class="field-shell grid min-h-0 grid-cols-1 items-center gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,7.5rem)_1fr_4.25rem]"
  >
    <div class="min-w-0">
      <label class="block truncate text-xs font-semibold text-black dark:text-slate-200" for={params.id}>
        {params?.title}
      </label>
      <p class="truncate text-[10px] tabular-nums tracking-wide text-black/55 dark:text-slate-500">
        {min}–{max} · step {params?.step ?? 1}
      </p>
    </div>
    <input
      class="range-shell h-2 w-full min-w-0 cursor-pointer appearance-none rounded-full"
      style={`--range-progress: ${Math.max(0, Math.min(100, progressPercent))}%`}
      bind:value
      type="range"
      id={params.id}
      name={params.id}
      min={params?.min}
      max={params?.max}
      step={params?.step ?? 1}
      {disabled}
    />
    <input
      type="number"
      step={params?.step ?? 1}
      bind:value
      disabled={disabled}
      class="input-shell w-full max-w-[4.25rem] justify-self-end py-1.5 text-center text-xs font-semibold tabular-nums sm:justify-self-auto"
    />
  </div>
{:else}
  <div
    class="field-shell grid min-h-[104px] grid-cols-1 gap-3 sm:grid-cols-[minmax(0,180px)_1fr_108px] sm:items-center"
  >
    <div>
      <label class="text-sm font-semibold text-black dark:text-slate-200" for={params.id}>
        {params?.title}
      </label>
      <p class="mt-1 text-xs uppercase tracking-[0.22em] text-black dark:text-slate-400">
        {min} to {max}
      </p>
    </div>
    <div class="space-y-3">
      <input
        class="range-shell h-2.5 w-full cursor-pointer appearance-none rounded-full"
        style={`--range-progress: ${Math.max(0, Math.min(100, progressPercent))}%`}
        bind:value
        type="range"
        id={params.id}
        name={params.id}
        min={params?.min}
        max={params?.max}
        step={params?.step ?? 1}
        {disabled}
      />
      <div class="flex items-center justify-between text-[11px] font-medium uppercase tracking-[0.2em] text-black dark:text-slate-400">
        <span>{params?.step ?? 1} step</span>
        <span class="tabular-nums">{Math.round(progressPercent)}%</span>
      </div>
    </div>
    <input
      type="number"
      step={params?.step ?? 1}
      bind:value
      disabled={disabled}
      class="input-shell text-center font-semibold tabular-nums"
    />
  </div>
{/if}
