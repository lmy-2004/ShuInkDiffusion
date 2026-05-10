<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = '';
  export let params: FieldProps;
  export let disabled: boolean = false;
  onMount(() => {
    value = String(params?.default);
  });
</script>

<div
  class="field-shell grid min-h-[82px] grid-cols-1 gap-3 sm:grid-cols-[minmax(0,180px)_1fr] sm:items-center"
>
  <div>
    <label for="model-list" class="text-sm font-semibold text-black dark:text-slate-200">
      {params?.title}
    </label>
    <p class="mt-1 text-xs uppercase tracking-[0.22em] text-black dark:text-slate-400">pipeline source</p>
  </div>
  {#if params?.values}
    <select
      bind:value
      disabled={disabled}
      id="model-list"
      class="input-shell w-full cursor-pointer font-medium"
    >
      {#each params.values as model, i}
        <option value={model} selected={i === 0}>{model}</option>
      {/each}
    </select>
  {/if}
</div>
