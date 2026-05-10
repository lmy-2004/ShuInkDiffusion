export const STYLIZATION_PARAM_IDS = new Set([
  'enable_stylize_preprocess',
  'depth_blur_sigma',
  'depth_power',
  'depth_smoothstep_min',
  'depth_smoothstep_max',
  'saturation_scale',
  'image_blur_sigma',
  'fog_white_mix',
  'outline_blur_sigma',
  'outline_power',
  'outline_smoothstep_min',
  'outline_smoothstep_max'
]);

/** 主「管线选项」中不展示：由 demo 配置 conditioning_kwargs 决定 */
export const HIDDEN_CONDITIONING_UI_IDS = new Set([
  'use_softedge',
  'softedge_mode',
  'softedge_debug'
]);

export function isHiddenFromMainPipeline(key: string, rawId: string | undefined): boolean {
  const id = String(rawId ?? key);
  return STYLIZATION_PARAM_IDS.has(id) || HIDDEN_CONDITIONING_UI_IDS.has(id);
}

export const STYLIZATION_GROUPS: { title: string; ids: readonly string[] }[] = [
  {
    title: '总开关与深度',
    ids: [
      'enable_stylize_preprocess',
      'depth_blur_sigma',
      'depth_power',
      'depth_smoothstep_min',
      'depth_smoothstep_max'
    ]
  },
  {
    title: '原图与雾化',
    ids: ['saturation_scale', 'image_blur_sigma', 'fog_white_mix']
  },
  {
    title: '软描边合成',
    ids: [
      'outline_blur_sigma',
      'outline_power',
      'outline_smoothstep_min',
      'outline_smoothstep_max'
    ]
  }
];
