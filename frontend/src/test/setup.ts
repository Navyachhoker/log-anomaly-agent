import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver, which the charts library needs.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;