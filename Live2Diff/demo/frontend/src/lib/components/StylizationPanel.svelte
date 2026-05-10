<script lang="ts">
  import type { Fields, FieldProps } from '$lib/types';
  import { FieldType } from '$lib/types';
  import InputRange from './InputRange.svelte';
  import Checkbox from './Checkbox.svelte';
  import { pipelineValues } from '$lib/store';
  import { STYLIZATION_GROUPS } from '$lib/stylizationParams';
  import { jsonPropertyToFieldProps } from '$lib/schemaFields';

  export let pipelineParams: Fields;
  export let disabled: boolean = false;

  let stylizationSeeded = false;

  function fieldFor(id: string): FieldProps | null {
    const raw = pipelineParams?.[id] as unknown as Record<string, unknown> | undefined;
    if (!raw) {
      return null;
    }
    return jsonPropertyToFieldProps(raw, id);
  }

  $: if (pipelineParams && !stylizationSeeded && Object.keys(pipelineParams).length > 0) {
    stylizationSeeded = true;
    pipelineValues.update((v) => {
      const n = { ...v };
      for (const g of STYLIZATION_GROUPS) {
        for (const id of g.ids) {
          const raw = pipelineParams[id] as unknown as Record<string, unknown> | undefined;
          if (!raw || n[id] !== undefined) {
            continue;
          }
          n[id] = jsonPropertyToFieldProps(raw, id).default;
        }
      }
      return n;
    });
  }
</script>

<div class="flex flex-col gap-4">
  <div>
    <h3 class="text-lg font-semibold text-black dark:text-slate-200">风格化预处理</h3>
    <p class="mt-0.5 text-[11px] uppercase tracking-[0.18em] text-black dark:text-slate-400">
      推理过程中可实时调节，下一帧即生效
    </p>
  </div>

  {#each STYLIZATION_GROUPS as group}
    <div class="stylize-group">
      <h4 class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-cyan-700 dark:text-cyan-300">
        {group.title}
      </h4>
      <div class="stylize-param-grid">
        {#each group.ids as paramId}
          {@const params = fieldFor(paramId)}
          {#if params}
            {#if params.field === FieldType.CHECKBOX}
              <div class="col-span-full">
                <Checkbox compact {params} bind:value={$pipelineValues[params.id]} {disabled} />
              </div>
            {:else if params.field === FieldType.RANGE}
              <InputRange compact {params} bind:value={$pipelineValues[params.id]} {disabled} />
            {/if}
          {/if}
        {/each}
      </div>
    </div>
  {/each}
</div>

<style lang="postcss">
  .stylize-group {
    @apply rounded-2xl border px-3 py-3 sm:px-4 sm:py-3;
    border-color: rgba(34, 211, 238, 0.22);
    background:
      linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(240, 253, 250, 0.5)),
      rgba(255, 255, 255, 0.72);
  }

  :global(.dark) .stylize-group {
    border-color: rgba(34, 211, 238, 0.2);
    background:
      linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(15, 118, 110, 0.12)),
      rgba(15, 23, 42, 0.56);
  }

  .stylize-param-grid {
    @apply grid grid-cols-1 gap-x-3 gap-y-2 sm:grid-cols-2 xl:grid-cols-3;
  }
</style>
