<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  import Button from './Button.svelte';
  export let value = 299792458;
  export let params: FieldProps;
  export let disabled: boolean = false;

  onMount(() => {
    value = Number(params?.default ?? '');
  });
  function randomize() {
    value = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
  }
</script>

<div
  class="field-shell grid min-h-[82px] grid-cols-1 gap-3 sm:grid-cols-[minmax(0,120px)_1fr_auto] sm:items-center"
>
  <div>
    <label class="text-sm font-semibold text-black dark:text-slate-200" for="seed">Seed</label>
    <p class="mt-1 text-xs uppercase tracking-[0.22em] text-black dark:text-slate-400">reproducible run</p>
  </div>
  <input
    bind:value
    type="number"
    id="seed"
    name="seed"
    class="input-shell text-right font-medium tabular-nums"
    disabled={disabled}
  />
  <Button on:click={randomize} classList={'px-5 py-3'}>Rand</Button>
</div>
