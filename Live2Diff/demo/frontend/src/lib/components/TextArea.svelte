<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value: string;
  export let params: FieldProps;
  export let disabled: boolean = false;
  onMount(() => {
    value = String(params?.default ?? '');
  });
</script>

<div class="field-shell">
  <label class="text-sm font-semibold text-black dark:text-slate-200" for={params?.title}>
    {params?.title}
  </label>
  <p class="mt-1 text-xs uppercase tracking-[0.22em] text-black dark:text-slate-400">text conditioning</p>
  <div
    class="light-surface mt-3 flex items-center rounded-2xl border border-slate-300/70 bg-white/80 dark:border-slate-600 dark:bg-slate-950/70"
  >
    <textarea
      class="min-h-[120px] w-full resize-y bg-transparent px-4 py-3 text-sm text-black outline-none dark:text-slate-200"
      title={params?.title}
      placeholder="Add your prompt here..."
      bind:value
      disabled={disabled}
    ></textarea>
  </div>
</div>
