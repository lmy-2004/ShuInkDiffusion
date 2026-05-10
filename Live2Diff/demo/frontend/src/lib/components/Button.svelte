<script lang="ts">
  export let classList: string = 'p-2';
  export let disabled: boolean = false;
  export let title: string = '';
  export let active: boolean = false;
</script>

<button class={`button group ${active ? 'button--active' : ''} ${classList}`} on:click {disabled} {title}>
  <span class="button__halo"></span>
  <span class="button__beam"></span>
  <span class="button__grid"></span>
  <span class="button__content">
    <slot />
  </span>
</button>

<style lang="postcss" scoped>
  .button {
    @apply relative isolate inline-flex items-center justify-center overflow-hidden rounded-2xl px-4 py-2.5 text-sm font-medium text-black transition;
    background:
      linear-gradient(135deg, rgba(248, 250, 252, 0.96), rgba(226, 232, 240, 0.88)),
      linear-gradient(120deg, rgba(56, 189, 248, 0.18), rgba(168, 85, 247, 0.16));
    box-shadow:
      0 18px 38px rgba(15, 23, 42, 0.12),
      inset 0 1px 0 rgba(255, 255, 255, 0.45);
    transform: translateY(0);
  }

  .button::before {
    content: '';
    position: absolute;
    inset: 1px;
    border-radius: inherit;
    border: 1px solid rgba(15, 23, 42, 0.08);
    pointer-events: none;
  }

  .button:hover {
    transform: translateY(-2px);
    box-shadow:
      0 24px 48px rgba(15, 23, 42, 0.16),
      0 0 0 1px rgba(56, 189, 248, 0.16),
      inset 0 1px 0 rgba(255, 255, 255, 0.52);
  }

  .button:active {
    transform: translateY(0);
  }

  .button:focus-visible {
    outline: none;
    box-shadow:
      0 0 0 4px rgba(56, 189, 248, 0.16),
      0 24px 48px rgba(15, 23, 42, 0.16),
      inset 0 1px 0 rgba(255, 255, 255, 0.52);
  }

  .button:disabled {
    @apply text-slate-600;
    background:
      linear-gradient(135deg, rgba(148, 163, 184, 0.66), rgba(148, 163, 184, 0.54)),
      rgba(148, 163, 184, 0.46);
    box-shadow: none;
  }

  .button__halo {
    position: absolute;
    inset: -30%;
    opacity: 0;
    pointer-events: none;
    background:
      radial-gradient(circle, rgba(34, 211, 238, 0.32), transparent 45%),
      radial-gradient(circle at 70% 30%, rgba(168, 85, 247, 0.22), transparent 42%);
    transform: scale(0.85);
    transition:
      opacity 280ms ease,
      transform 280ms ease;
  }

  .button__beam {
    @apply absolute inset-0 opacity-0 transition-opacity duration-300;
    background:
      linear-gradient(120deg, transparent 15%, rgba(255, 255, 255, 0.24) 50%, transparent 85%),
      radial-gradient(circle at top, rgba(34, 211, 238, 0.22), transparent 54%);
    transform: translateX(-36%);
  }

  .button__grid {
    position: absolute;
    inset: 0;
    opacity: 0.15;
    pointer-events: none;
    background-image:
      linear-gradient(to right, rgba(255, 255, 255, 0.16) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255, 255, 255, 0.12) 1px, transparent 1px);
    background-size: 16px 16px;
    mask-image: linear-gradient(180deg, rgba(0, 0, 0, 0.5), transparent 100%);
  }

  .button:hover .button__halo,
  .button:focus-visible .button__halo,
  .button--active .button__halo {
    opacity: 1;
    transform: scale(1);
  }

  .button:hover .button__beam,
  .button:focus-visible .button__beam {
    @apply opacity-100;
    transform: translateX(0);
  }

  .button--active {
    box-shadow:
      0 24px 56px rgba(15, 23, 42, 0.16),
      0 0 0 1px rgba(34, 211, 238, 0.2),
      0 0 28px rgba(34, 211, 238, 0.16),
      inset 0 1px 0 rgba(255, 255, 255, 0.52);
  }

  .button__content {
    @apply relative z-10 inline-flex items-center justify-center gap-2;
  }

  :global(.dark) .button {
    box-shadow:
      0 18px 38px rgba(2, 6, 23, 0.3),
      inset 0 1px 0 rgba(255, 255, 255, 0.45);
  }

  :global(.dark) .button:hover {
    box-shadow:
      0 24px 48px rgba(2, 6, 23, 0.36),
      0 0 0 1px rgba(56, 189, 248, 0.16),
      inset 0 1px 0 rgba(255, 255, 255, 0.52);
  }

  :global(.dark) .button--active {
    box-shadow:
      0 24px 56px rgba(2, 6, 23, 0.4),
      0 0 0 1px rgba(34, 211, 238, 0.16),
      0 0 24px rgba(34, 211, 238, 0.18),
      inset 0 1px 0 rgba(255, 255, 255, 0.52);
  }

  :global(.dark) .button:disabled {
    @apply bg-slate-700 text-slate-400;
  }
</style>
