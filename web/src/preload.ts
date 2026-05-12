// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts
// preload.ts
import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("api", {
  getConfig: () => ipcRenderer.invoke("config:get"),
  setConfig: (partial: { apiBaseUrl?: string; recognizeTopK?: number }) =>
    ipcRenderer.invoke("config:set", partial),
});

contextBridge.exposeInMainWorld("boot", {
  init: () => {
    console.log("preload done");
  }
});