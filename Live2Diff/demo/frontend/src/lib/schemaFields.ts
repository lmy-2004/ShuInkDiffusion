import { FieldType } from '$lib/types';
import type { FieldProps } from '$lib/types';

export function jsonPropertyToFieldProps(raw: Record<string, unknown>, key: string): FieldProps {
  const id = String(raw.id ?? key);
  const title = String(raw.title ?? key);
  const def = raw.default as FieldProps['default'];
  const min = raw.minimum as number | undefined;
  const max = raw.maximum as number | undefined;
  const step = raw.step as number | undefined;
  const disabled = Boolean(raw.disabled);
  const hide = Boolean(raw.hide);
  const values = raw.enum as string[] | undefined;

  if (raw.field) {
    return {
      id,
      title,
      default: def,
      min,
      max,
      step,
      disabled,
      hide,
      values,
      field: raw.field as FieldProps['field']
    };
  }

  const t = raw.type as string | undefined;
  if (t === 'boolean') {
    return {
      id,
      title,
      default: Boolean(def),
      disabled,
      hide,
      field: FieldType.CHECKBOX
    };
  }
  if (t === 'number' || t === 'integer') {
    return {
      id,
      title,
      default: Number(def ?? 0),
      min: min ?? 0,
      max: max ?? 100,
      step: step ?? (t === 'integer' ? 1 : 0.01),
      disabled,
      hide,
      field: FieldType.RANGE
    };
  }
  if (t === 'string' && Array.isArray(values) && values.length > 0) {
    return {
      id,
      title,
      default: String(def ?? values[0]),
      values,
      disabled,
      hide,
      field: FieldType.SELECT
    };
  }
  if (t === 'string') {
    return {
      id,
      title,
      default: String(def ?? ''),
      disabled,
      hide,
      field: FieldType.TEXTAREA
    };
  }
  return {
    id,
    title,
    default: Number(def ?? 0),
    min: 0,
    max: 1,
    step: 0.01,
    field: FieldType.RANGE
  };
}
