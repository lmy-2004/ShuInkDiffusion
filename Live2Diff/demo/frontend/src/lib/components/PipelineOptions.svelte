<script lang="ts">
  import type { FieldProps, Fields } from '$lib/types';
  import { FieldType } from '$lib/types';
  import InputRange from './InputRange.svelte';
  import SeedInput from './SeedInput.svelte';
  import TextArea from './TextArea.svelte';
  import Checkbox from './Checkbox.svelte';
  import Selectlist from './Selectlist.svelte';
  import { pipelineValues } from '$lib/store';
  import { isHiddenFromMainPipeline } from '$lib/stylizationParams';
  import { jsonPropertyToFieldProps } from '$lib/schemaFields';

  export let pipelineParams: Fields;
  export let disabled: boolean = false;

  function mergeFieldProps(key: string, raw: Record<string, unknown>): FieldProps {
    const inferred = jsonPropertyToFieldProps(raw, key);
    if (raw.field) {
      return { ...inferred, ...raw, field: raw.field as FieldProps['field'], id: String(raw.id ?? key) };
    }
    return inferred;
  }

  $: advanceOptions = Object.entries(pipelineParams ?? {})
    .filter(([key, e]) => {
      const raw = e as unknown as Record<string, unknown>;
      return (
        e &&
        raw.hide == true &&
        raw.disabled !== true &&
        !isHiddenFromMainPipeline(key, raw.id as string | undefined)
      );
    })
    .map(([key, e]) => mergeFieldProps(key, e as unknown as Record<string, unknown>));
  $: featuredOptions = Object.entries(pipelineParams ?? {})
    .filter(([key, e]) => {
      const raw = e as unknown as Record<string, unknown>;
      return e && raw.hide !== true && !isHiddenFromMainPipeline(key, raw.id as string | undefined);
    })
    .map(([key, e]) => mergeFieldProps(key, e as unknown as Record<string, unknown>));
</script>

<div class="flex flex-col gap-5">
  <div class="grid grid-cols-1 gap-4 xl:grid-cols-2">
    {#if featuredOptions}
      {#each featuredOptions as params}
        {#if params.field === FieldType.RANGE}
          <InputRange {params} bind:value={$pipelineValues[params.id]} {disabled}></InputRange>
        {:else if params.field === FieldType.SEED}
          <SeedInput {params} bind:value={$pipelineValues[params.id]} {disabled}></SeedInput>
        {:else if params.field === FieldType.TEXTAREA}
          <TextArea {params} bind:value={$pipelineValues[params.id]} {disabled}></TextArea>
        {:else if params.field === FieldType.CHECKBOX}
          <Checkbox {params} bind:value={$pipelineValues[params.id]} {disabled}></Checkbox>
        {:else if params.field === FieldType.SELECT}
          <Selectlist {params} bind:value={$pipelineValues[params.id]} {disabled}></Selectlist>
        {/if}
      {/each}
    {/if}
  </div>
  {#if advanceOptions && advanceOptions.length > 0}
    <details class="advanced-panel">
      <summary class="advanced-panel__summary">
        <div>
          <p class="text-base font-semibold text-black dark:text-slate-200">Advanced Options</p>
          <p class="mt-1 text-xs uppercase tracking-[0.24em] text-black dark:text-slate-400">
            extended control surface
          </p>
        </div>
        <span class="advanced-panel__icon">+</span>
      </summary>
      <div class="advanced-panel__content">
        <div
          class="advanced-panel__inner mt-4 grid grid-cols-1 gap-4 {Object.values(pipelineParams).length > 5
            ? 'sm:grid-cols-2'
            : ''}"
        >
          {#each advanceOptions as params}
            {#if params.field === FieldType.RANGE}
              <InputRange {params} bind:value={$pipelineValues[params.id]} {disabled}></InputRange>
            {:else if params.field === FieldType.SEED}
              <SeedInput {params} bind:value={$pipelineValues[params.id]} {disabled}></SeedInput>
            {:else if params.field === FieldType.TEXTAREA}
              <TextArea {params} bind:value={$pipelineValues[params.id]} {disabled}></TextArea>
            {:else if params.field === FieldType.CHECKBOX}
              <Checkbox {params} bind:value={$pipelineValues[params.id]} {disabled}></Checkbox>
            {:else if params.field === FieldType.SELECT}
              <Selectlist {params} bind:value={$pipelineValues[params.id]} {disabled}></Selectlist>
            {/if}
          {/each}
        </div>
      </div>
    </details>
  {/if}
</div>

<style lang="postcss">
  .advanced-panel {
    @apply rounded-[24px] border p-5;
    border-color: rgba(148, 163, 184, 0.22);
    background:
      linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(255, 255, 255, 0.58)),
      rgba(255, 255, 255, 0.72);
  }

  .advanced-panel :global(summary::-webkit-details-marker) {
    display: none;
  }

  .advanced-panel__summary {
    @apply flex cursor-pointer list-none items-center justify-between gap-4;
  }

  .advanced-panel__icon {
    @apply inline-flex h-10 w-10 items-center justify-center rounded-2xl text-lg font-semibold text-cyan-500;
    background: rgba(56, 189, 248, 0.1);
    transition:
      transform 220ms ease,
      background-color 220ms ease;
  }

  .advanced-panel__content {
    display: grid;
    grid-template-rows: 0fr;
    transition:
      grid-template-rows 280ms ease,
      opacity 220ms ease;
    opacity: 0.64;
  }

  .advanced-panel__inner {
    overflow: hidden;
  }

  .advanced-panel[open] .advanced-panel__content {
    grid-template-rows: 1fr;
    opacity: 1;
  }

  .advanced-panel[open] .advanced-panel__icon {
    transform: rotate(45deg);
    background: rgba(56, 189, 248, 0.16);
  }

  :global(.dark) .advanced-panel {
    border-color: rgba(71, 85, 105, 0.78);
    background:
      linear-gradient(180deg, rgba(30, 41, 59, 0.56), rgba(15, 23, 42, 0.48)),
      rgba(15, 23, 42, 0.56);
  }
</style>
