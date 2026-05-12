const RETURN_KEY = "newttracker_newt_return";

export type NewtReturnPayload = {
  target: "/recognize";
  newtId: string;
};

export function setNewtOpenedFromRecognize(newtId: string) {
  sessionStorage.setItem(
    RETURN_KEY,
    JSON.stringify({ target: "/recognize", newtId } satisfies NewtReturnPayload),
  );
}

export function clearNewtReturnHint() {
  sessionStorage.removeItem(RETURN_KEY);
}

export function getBackHrefForNewt(newtId: string, defaultHref: string): string {
  try {
    const raw = sessionStorage.getItem(RETURN_KEY);
    if (!raw) return defaultHref;
    const p = JSON.parse(raw) as NewtReturnPayload;
    if (p?.target === "/recognize" && p?.newtId === newtId) return "/recognize";
  } catch {
    /* ignore */
  }
  return defaultHref;
}
