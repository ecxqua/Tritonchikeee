import { app, BrowserWindow, ipcMain } from 'electron';
import fs from "fs";
import path from 'path';
import started from 'electron-squirrel-startup';

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (started) {
  app.quit();
}

const OVERRIDE_FILE = "app-cfg.override.json";

type AppConfig = {
  apiBaseUrl: string;
  recognizeTopK: number;
};

const DEFAULT_CONFIG: AppConfig = {
  apiBaseUrl: "http://localhost:8080",
  recognizeTopK: 5,
};

let config: AppConfig = { ...DEFAULT_CONFIG };

function stripTrailingSlashes(url: string): string {
  return url.replace(/\/+$/, "");
}

function clampRecognizeTopK(value: unknown): number {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return DEFAULT_CONFIG.recognizeTopK;
  const k = Math.floor(n);
  if (k < 1) return 1;
  if (k > 100) return 100;
  return k;
}

function normalizeConfig(raw: Partial<AppConfig> & Record<string, unknown>): AppConfig {
  const url = stripTrailingSlashes(String(raw.apiBaseUrl ?? "").trim()) || DEFAULT_CONFIG.apiBaseUrl;
  return {
    apiBaseUrl: url,
    recognizeTopK: clampRecognizeTopK(raw.recognizeTopK),
  };
}

function loadConfig() {
  let merged: Record<string, unknown> = { ...DEFAULT_CONFIG };

  try {
    const configPath = app.isPackaged
      ? path.join(process.resourcesPath, "app-cfg.json")
      : path.join(__dirname, "../../app-cfg.json");

    const raw = fs.readFileSync(configPath, "utf-8");
    merged = { ...merged, ...JSON.parse(raw) };
  } catch {
    console.warn("Using default config");
  }

  try {
    const overridePath = path.join(app.getPath("userData"), OVERRIDE_FILE);
    if (fs.existsSync(overridePath)) {
      const over = JSON.parse(fs.readFileSync(overridePath, "utf-8"));
      merged = { ...merged, ...over };
    }
  } catch (e) {
    console.warn("Could not merge config override:", e);
  }

  config = normalizeConfig(merged);
}

function registerConfigIpc() {
  ipcMain.removeHandler("config:get");
  ipcMain.removeHandler("config:set");

  ipcMain.handle("config:get", () => ({ ...config }));

  ipcMain.handle(
    "config:set",
    (_evt, partial: { apiBaseUrl?: string; recognizeTopK?: number }) => {
      let next: AppConfig = { ...config };

      if (partial && typeof partial.apiBaseUrl === "string") {
        const t = stripTrailingSlashes(partial.apiBaseUrl.trim());
        if (t) {
          next.apiBaseUrl = t;
        }
      }
      if (partial && partial.recognizeTopK !== undefined) {
        next.recognizeTopK = clampRecognizeTopK(partial.recognizeTopK);
      }

      config = normalizeConfig(next);

      const userData = app.getPath("userData");
      fs.mkdirSync(userData, { recursive: true });
      const overridePath = path.join(userData, OVERRIDE_FILE);
      fs.writeFileSync(
        overridePath,
        JSON.stringify(
          { apiBaseUrl: config.apiBaseUrl, recognizeTopK: config.recognizeTopK },
          null,
          2,
        ),
        "utf-8",
      );
      return { ...config };
    },
  );
}

const createWindow = () => {
  loadConfig();
  registerConfigIpc();
  
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    title: "NewtTracker",
    icon: path.join(__dirname, "assets/logo.png"),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      webSecurity: false
    },
  });

  mainWindow.setMenuBarVisibility(false);

  mainWindow.webContents.on('did-fail-load', (_, errorCode, errorDescription) => {
    console.error('Failed to load:', errorCode, errorDescription);
  });

  mainWindow.webContents.on('console-message', (_, level, message) => {
    console.log('Renderer:', message);
  });

  mainWindow.webContents.on('did-fail-load', (_, errorCode, errorDescription, validatedURL) => {
    console.error('❌ did-fail-load:', errorCode, errorDescription, validatedURL);
  });

  mainWindow.webContents.on('did-finish-load', () => {
    console.log('✅ did-finish-load');
  });

  mainWindow.webContents.on("render-process-gone", (_, details) => {
    console.error("RENDER CRASHED:", details);
  });

  const pathToFile = path.join(__dirname, '../renderer/main_window/index.html');

  // and load the index.html of the app.
  if (MAIN_WINDOW_VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(MAIN_WINDOW_VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(pathToFile);
  };

  // Open the DevTools.
};

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow);

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On OS X it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.
