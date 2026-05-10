export const enum FieldType {
  RANGE = 'range',
  SEED = 'seed',
  TEXTAREA = 'textarea',
  CHECKBOX = 'checkbox',
  SELECT = 'select'
}
export const enum PipelineMode {
  IMAGE = 'image',
  VIDEO = 'video',
  TEXT = 'text'
}

export const enum InputSource {
  CAMERA = 'camera',
  VIDEO = 'video'
}

export interface Fields {
  [key: string]: FieldProps;
}

export interface FieldProps {
  default: boolean | number | string;
  max?: number;
  min?: number;
  title: string;
  field: FieldType;
  step?: number;
  disabled?: boolean;
  hide?: boolean;
  id: string;
  values?: string[];
}
export interface PipelineInfo {
  title: {
    default: string;
  };
  name: string;
  description: string;
  input_mode: {
    default: PipelineMode;
  };
}

export interface DebugInfo {
  phase: string;
  prompt: string;
  width: number;
  height: number;
  inference_time: number;
  depth_time: number;
  softedge_time: number;
  subject_mask_time?: number;
  inference_steps: number;
  has_depth: boolean;
  has_softedge_preview?: boolean;
  has_subject_mask_preview?: boolean;
  has_source_preview?: boolean;
  has_stylized_preview?: boolean;
  use_softedge?: boolean;
  softedge_mode?: string;
  softedge_backend?: string;
  softedge_pidinet_error?: string | null;
  subject_mask_debug?: boolean;
  subject_mask_backend?: string;
  subject_mask_requested_backend?: string;
  subject_mask_sam2_error?: string | null;
  fog_depth_mask_blend?: number;
  subject_mask_fog_blur_sigma?: number;
  temporal_stability_src?: number | null;
  temporal_stability_out?: number | null;
  temporal_stability_delta?: number | null;
  temporal_stability_delta_avg_30s?: number | null;
  temporal_stability_e_src?: number | null;
  temporal_stability_e_out?: number | null;
  sync_enabled?: boolean;
  sync_key_step_index?: number;
  sync_strength?: number;
  sync_weight?: number;
  sync_memory_valid?: boolean;
  flow_backend?: string;
}
