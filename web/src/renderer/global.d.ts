export {};

declare global {
  interface Window {
    api: {
      getConfig: () => Promise<{ apiBaseUrl: string }>;
      setConfig: (partial: { apiBaseUrl?: string }) => Promise<{ apiBaseUrl: string }>;
    };
  }
}