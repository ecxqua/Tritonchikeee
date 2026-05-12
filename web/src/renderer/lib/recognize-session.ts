const REC_KEY = "newttracker_recognize_session_v1";

export type NewtSummaryEntry = {
  isLoading: boolean;
  projectId?: string;
  sex?: string;
  status?: string;
  cardTypes: string[];
};

export type RecognizeSessionSnapshot = {
  result: unknown;
  scope: "all" | "by_species" | "by_territory";
  projectId: string;
  expandedNewtId: string | null;
  newtSummaryById: Record<string, NewtSummaryEntry>;
  /** data: URL persisted so preview survives navigation away from /recognize */
  previewDataUrl?: string | null;
  photoName?: string;
};

export function loadRecognizeSession(): RecognizeSessionSnapshot | null {
  try {
    const raw = sessionStorage.getItem(REC_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as RecognizeSessionSnapshot;
  } catch {
    return null;
  }
}

export function saveRecognizeSession(snapshot: RecognizeSessionSnapshot) {
  try {
    sessionStorage.setItem(REC_KEY, JSON.stringify(snapshot));
  } catch {
    /* quota or private mode */
  }
}

export function clearRecognizeSession() {
  sessionStorage.removeItem(REC_KEY);
}
