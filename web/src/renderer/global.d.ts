export {};

declare global {
  interface Window {
    api: {
      getConfig: () => Promise<{ apiBaseUrl: string }>;
    };
  }
}