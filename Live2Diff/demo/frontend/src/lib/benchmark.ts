import { jsonPropertyToFieldProps } from '$lib/schemaFields';
import type { FieldProps, Fields } from '$lib/types';
import { FieldType } from '$lib/types';
import { downloadBlob, sanitizeFileStem } from '$lib/utils';

export type BenchmarkParamMode = 'list' | 'range';

export interface BenchmarkParamDraft {
  key: string;
  enabled: boolean;
  paramId: string;
  mode: BenchmarkParamMode;
  valuesText: string;
  minText: string;
  maxText: string;
  stepText: string;
}

export interface BenchmarkParamSpec {
  paramId: string;
  title: string;
  values: number[];
  mode: BenchmarkParamMode;
  min?: number;
  max?: number;
  step?: number;
}

export interface BenchmarkSamplePoint {
  elapsed_s: number;
  score: number;
}

export interface BenchmarkResultEntry {
  index: number;
  params: Record<string, number>;
  score: number | null;
  sample_points: BenchmarkSamplePoint[];
  sample_count: number;
  started_at: string;
  completed_at: string;
  error?: string;
}

export function getNumericFieldOptions(pipelineParams: Fields): FieldProps[] {
  return Object.entries(pipelineParams ?? {})
    .map(([key, raw]) => jsonPropertyToFieldProps(raw as unknown as Record<string, unknown>, key))
    .filter((field) => field.field === FieldType.RANGE && field.disabled !== true);
}

function parseNumberList(valuesText: string): number[] {
  const values = valuesText
    .split(/[\s,]+/u)
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => Number(value));
  if (values.length === 0 || values.some((value) => !Number.isFinite(value))) {
    throw new Error('候选值列表中包含非法数字');
  }
  return Array.from(new Set(values.map((value) => roundNumber(value, 6))));
}

function getPrecision(value: number): number {
  const text = String(value);
  if (!text.includes('.')) {
    return 0;
  }
  return text.split('.')[1]?.length ?? 0;
}

function expandRange(min: number, max: number, step: number): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(step) || step <= 0) {
    throw new Error('范围参数不合法');
  }
  if (max < min) {
    throw new Error('范围上限不能小于下限');
  }
  const precision = Math.max(getPrecision(min), getPrecision(max), getPrecision(step), 4);
  const factor = 10 ** precision;
  const start = Math.round(min * factor);
  const end = Math.round(max * factor);
  const stepInt = Math.round(step * factor);
  if (stepInt <= 0) {
    throw new Error('步长必须大于 0');
  }
  const values: number[] = [];
  for (let cursor = start; cursor <= end; cursor += stepInt) {
    values.push(Number((cursor / factor).toFixed(precision)));
    if (values.length > 1000) {
      throw new Error('参数组合过多，请缩小范围');
    }
  }
  const last = values[values.length - 1];
  if (last == null || Math.abs(last - max) > 10 ** -precision) {
    values.push(Number(max.toFixed(precision)));
  }
  return Array.from(new Set(values.map((value) => roundNumber(value, 6))));
}

export function resolveBenchmarkParam(
  draft: BenchmarkParamDraft,
  fieldMap: Map<string, FieldProps>
): BenchmarkParamSpec {
  const field = fieldMap.get(draft.paramId);
  if (!field) {
    throw new Error('请选择有效参数');
  }
  if (draft.mode === 'list') {
    return {
      paramId: draft.paramId,
      title: field.title,
      mode: 'list',
      values: parseNumberList(draft.valuesText)
    };
  }
  const min = Number(draft.minText);
  const max = Number(draft.maxText);
  const step = Number(draft.stepText);
  return {
    paramId: draft.paramId,
    title: field.title,
    mode: 'range',
    min,
    max,
    step,
    values: expandRange(min, max, step)
  };
}

export function buildBenchmarkCombos(specs: BenchmarkParamSpec[]): Array<Record<string, number>> {
  if (specs.length === 0) {
    return [];
  }
  const combos: Array<Record<string, number>> = [];
  const walk = (index: number, current: Record<string, number>) => {
    if (index >= specs.length) {
      combos.push({ ...current });
      return;
    }
    const spec = specs[index];
    for (const value of spec.values) {
      current[spec.paramId] = value;
      walk(index + 1, current);
    }
  };
  walk(0, {});
  return combos;
}

export function averageScore(values: Array<number | null | undefined>): number | null {
  const valid = values.filter((value): value is number => typeof value === 'number' && Number.isFinite(value));
  if (valid.length === 0) {
    return null;
  }
  return roundNumber(valid.reduce((sum, value) => sum + value, 0) / valid.length, 6);
}

export function roundNumber(value: number, digits = 4): number {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

export function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.ceil(totalSeconds));
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remain = seconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m ${remain}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remain}s`;
  }
  return `${remain}s`;
}

export function estimateBenchmarkDuration(
  comboCount: number,
  warmupSeconds: number,
  sampleSeconds: number
): number {
  return comboCount * Math.max(0, warmupSeconds + sampleSeconds);
}

export function getBestBenchmarkResult(
  results: BenchmarkResultEntry[]
): BenchmarkResultEntry | null {
  const valid = results.filter(
    (result): result is BenchmarkResultEntry & { score: number } =>
      typeof result.score === 'number' && Number.isFinite(result.score)
  );
  if (valid.length === 0) {
    return null;
  }
  return valid.reduce((best, current) => (current.score > best.score ? current : best));
}

export function downloadBenchmarkJson(filenameStem: string, payload: unknown): string {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const filename = `${sanitizeFileStem(filenameStem)}.json`;
  downloadBlob(blob, filename);
  return filename;
}

export function defaultBenchmarkRows(fields: FieldProps[]): BenchmarkParamDraft[] {
  const defaults = ['softedge_scale', 'depth_scale'];
  const selected = defaults
    .map((id) => fields.find((field) => field.id === id))
    .filter((field): field is FieldProps => Boolean(field));
  if (selected.length === 0) {
    selected.push(...fields.slice(0, Math.min(2, fields.length)));
  }
  return selected.map((field, index) => ({
    key: `${field.id}-${index}`,
    enabled: true,
    paramId: field.id,
    mode: 'list',
    valuesText: '0, 0.2, 0.4, 0.6, 0.8, 1',
    minText: String(field.min ?? 0),
    maxText: String(field.max ?? 1),
    stepText: String(field.step ?? 0.2)
  }));
}
