import { get, writable } from 'svelte/store';

import { InputSource } from '$lib/types';

export interface VideoControllerState {
  source: InputSource;
  hasVideoLoaded: boolean;
  videoIsReady: boolean;
  currentTime: number;
  duration: number;
  selectedVideoFileName: string;
  canControl: boolean;
}

interface VideoController {
  waitForReady(timeoutMs?: number): Promise<void>;
  seek(seconds: number): Promise<void>;
  jumpToStart(): Promise<void>;
  play(): Promise<void>;
  pause(): void;
}

const initialState: VideoControllerState = {
  source: InputSource.CAMERA,
  hasVideoLoaded: false,
  videoIsReady: false,
  currentTime: 0,
  duration: 0,
  selectedVideoFileName: '',
  canControl: false
};

let controller: VideoController | null = null;

export const videoControllerState = writable<VideoControllerState>(initialState);

function getController(): VideoController {
  if (!controller) {
    throw new Error('视频控制器未就绪');
  }
  return controller;
}

export const videoControllerActions = {
  register(next: VideoController) {
    controller = next;
    videoControllerState.update((state) => ({ ...state, canControl: true }));
  },
  unregister() {
    controller = null;
    videoControllerState.set(initialState);
  },
  update(patch: Partial<VideoControllerState>) {
    videoControllerState.update((state) => ({ ...state, ...patch }));
  },
  snapshot(): VideoControllerState {
    return get(videoControllerState);
  },
  async waitForReady(timeoutMs = 15000) {
    await getController().waitForReady(timeoutMs);
  },
  async seek(seconds: number) {
    await getController().seek(seconds);
  },
  async jumpToStart() {
    await getController().jumpToStart();
  },
  async play() {
    await getController().play();
  },
  pause() {
    getController().pause();
  }
};
