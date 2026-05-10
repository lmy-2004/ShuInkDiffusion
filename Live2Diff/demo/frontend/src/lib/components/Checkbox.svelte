<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = false;
  export let params: FieldProps;
  export let disabled: boolean = false;
  export let compact: boolean = false;
  onMount(() => {
    value = Boolean(params?.default ?? false);
  });
</script>

<div
  class="field-shell flex items-center justify-between gap-3"
  class:min-h-[82px]={!compact}
  class:min-h-0={compact}
  class:py-1={compact}
>
  <div class="min-w-0">
    <label
      class="font-semibold text-black dark:text-slate-200 {compact ? 'text-xs' : 'text-sm'}"
      for={params.id}
    >
      {params?.title}
    </label>
    {#if !compact}
      <p class="mt-1 text-xs uppercase tracking-[0.22em] text-black dark:text-slate-400">toggle module</p>
    {/if}
  </div>
  <label class="relative inline-flex cursor-pointer items-center shrink-0">
    <input
      bind:checked={value}
      disabled={disabled}
      type="checkbox"
      id={params.id}
      class="peer sr-only"
    />
    <span class="h-7 w-12 rounded-full bg-slate-300 transition peer-checked:bg-cyan-400/90 peer-disabled:opacity-50 dark:bg-slate-700"></span>
    <span class="absolute left-1 h-5 w-5 rounded-full bg-white shadow transition peer-checked:translate-x-5"></span>
  </label>
</div>
