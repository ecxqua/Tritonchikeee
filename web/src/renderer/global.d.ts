export {};

declare global {
  interface Window {
    api: {
      getConfig: () => Promise<{ apiBaseUrl: string; recognizeTopK: number }>;
      setConfig: (partial: {
        apiBaseUrl?: string;
        recognizeTopK?: number;
      }) => Promise<{ apiBaseUrl: string; recognizeTopK: number }>;
    };
  }
}