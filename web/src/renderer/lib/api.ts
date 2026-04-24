
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const config = await window.api.getConfig();
  const BASE_URL = config.apiBaseUrl;

  console.log("Running request:", {
    url: `${BASE_URL}${path}`,
    options,
  });
  console.log("BASE_URL raw:", BASE_URL);
  const res = await fetch(`${BASE_URL}${path}`, options);

  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }

  if (res.status === 204) {
    return undefined as T;
  }

  return (await res.json()) as T;
}

export type Match = {
  newtId: string;
  confidence: number; // 0–100
  photoUrl?: string;
};

export type RecognizeResponse =
  | { status: "not_found" }
  | { status: "found"; matches: Match[] };

export async function recognizeNewt(params: {
  photo: File;
  scope: string;
  projectId?: number;
}): Promise<RecognizeResponse> {
  const formData = new FormData();
  formData.append("photo", params.photo);
  formData.append("scope", params.scope);

  if (params.projectId !== undefined) {
    formData.append("projectId", String(params.projectId));
  }

  return apiFetch<RecognizeResponse>("/recognize", {
    method: "POST",
    body: formData,
  });
}

export type CreateCardRequest = {
  cardType: "ИК-1" | "ИК-2" | "КВ-1" | "КВ-2";
  projectId?: number;
  data: {
    idNumber: string;
    [key: string]: any;
  };
};

export type CreateCardResponse = {
  id: string;
};

export async function createCardApi(
  req: CreateCardRequest & {
    files: File[];
    species: string;
  }
): Promise<CreateCardResponse> {
  const config = await window.api.getConfig();
  const BASE_URL = config.apiBaseUrl;
  
  const formData = new FormData();

  req.files.forEach((file) => {
    formData.append("files", file);
  });
  
  formData.append("species", req.species);
  formData.append("template_type", req.cardType);
  formData.append("card_id", req.data.idNumber);

  if (req.projectId !== undefined) {
    formData.append("project_id", String(req.projectId));
  }

  // append dynamic fields
  for (const [key, value] of Object.entries(req.data)) {
    if (key === "idNumber") continue;
    formData.append(key, String(value));
  }

  const res = await fetch(`${BASE_URL}/new`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(`createCardApi failed: ${res.status}`);
  }

  const json = (await res.json()) as { card_id: string };

  return {
    id: json.card_id,
  };
}

export type Project = {
  id: number;
  name: string;
  description?: string;
  species?: string[];
  territories?: string[];
  newtCount?: number;
  createdAt: string;
};

export async function listProjects(): Promise<Project[]> {
  const raw = await apiFetch<
    {
      id: number;
      name: string;
      description?: string;
      species?: string[];
      territories?: string[];
      createdAt: string;
      newtCount?: number;
    }[]
  >("/projects");

  return raw.map((p) => ({
    id: p.id,
    name: p.name,
    description: p.description,
    species: p.species,
    territories: p.territories,
    createdAt: p.createdAt,
    newtCount: p.newtCount,
  }));
}

export async function listSpecies(): Promise<string[]> {
  return apiFetch<string[]>("/species");
}

export async function listTerritories(): Promise<string[]> {
  return apiFetch<string[]>("/territories");
}

export async function createProject(data: {
  name: string;
  description: string;
  species: string[];
  territories: string[];
}): Promise<{ id: number }> {
  return apiFetch<{ id: number }>("/projects", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
}

// --- NEWT DETAIL MOCKS ---

export type Newt = {
  id: string;
  projectId?: string;
  cardType?: "ИК-1" | "ИК-2" | "КВ-1" | "КВ-2";
  status?: string;
  sex?: string;
  createdAt: string;
};

export type NewtCard = {
  cardType: "ИК-1" | "ИК-2" | "КВ-1" | "КВ-2";
  data: Record<string, any>;
  photos: string[];
};

export type HistoryRecord = {
  id: string;
  field: string;
  oldValue?: string;
  newValue?: string;
  changedAt: string;
  changedBy?: string;
};

export async function getNewt(newtId: string): Promise<Newt | null> {
  try {
    const raw = await apiFetch<{
      id: string;
      projectId?: string;
      cardType?: string;
      createdAt: string;
      sex?: string;
      status?: string;
    }>(`/newts/${newtId}`);

    return {
      id: raw.id,
      projectId: raw.projectId,
      cardType: raw.cardType as Newt["cardType"],
      createdAt: raw.createdAt,
      sex: raw.sex,
      status: raw.status,
    };
  } catch (e) {
    // if necessary can be extended
    return null;
  }
}

export async function getNewtCards(newtId: string): Promise<NewtCard[]> {
  const raw = await apiFetch<
    {
      cardType: string;
      data: Record<string, string | null>;
      photos: string[];
    }[]
  >(`/newts/${newtId}/cards`);

  return raw.map((card) => ({
    cardType: card.cardType as NewtCard["cardType"],
    data: card.data,
    photos: card.photos ?? [],
  }));
}

export async function getNewtHistory(newtId: string): Promise<HistoryRecord[]> {
  await new Promise(r => setTimeout(r, 300));

  return [
    {
      id: "1",
      field: "weight",
      oldValue: "4",
      newValue: "5",
      changedAt: new Date().toISOString(),
      changedBy: "Researcher",
    },
  ];
}

export async function updateNewtCardApi(req: {
  newtId: string;
  cardType: string;
  data: Record<string, any>;
}): Promise<void> {
  await apiFetch<void>(`/newts/${req.newtId}/card`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      cardType: req.cardType,
      ...req.data,
    }),
  });
}

// ---------------- PROJECTS ----------------

export async function getProject(projectId: number): Promise<Project | null> {
  try {
    const raw = await apiFetch<{
      id: number;
      name: string;
      description?: string;
      species?: string | string[];
      territories?: string | string[];
      createdAt: string;
      newtCount?: number;
    }>(`/projects/${projectId}`);

    return {
      id: raw.id,
      name: raw.name,
      description: raw.description,
      species: raw.species as string[] | undefined,
      territories: raw.territories as string[] | undefined,
      createdAt: raw.createdAt,
      newtCount: raw.newtCount,
    };
  } catch (e) {
    // assumes apiFetch throws on 404
    return null;
  }
}

export async function listNewts(params: {
  projectId: number;
}): Promise<Newt[]> {
  const raw = await apiFetch<
    {
      id: string;
      projectId: string;
      cardType?: string;
      createdAt: string;
      sex?: string;
      status?: string;
    }[]
  >(`/projects/${params.projectId}/newts`);

  return raw.map((n) => ({
    id: n.id,
    projectId: n.projectId,
    cardType: n.cardType as Newt["cardType"],
    createdAt: n.createdAt,
    sex: n.sex,
    status: n.status,
  }));
}

export async function updateProjectApi(req: {
  projectId: number;
  data: {
    name?: string;
    description?: string;
    species?: string[];
    territories?: string[];
  };
}): Promise<void> {
  await apiFetch<void>(`/projects/${req.projectId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(req.data),
  });
}

export async function deleteProjectApi(projectId: number): Promise<void> {
  await apiFetch<void>(`/projects/${projectId}`, {
    method: "DELETE",
  });
}

// ---------------- STATS / HOME ----------------

export type ProjectStats = {
  totalProjects: number;
  totalNewts: number;
  totalRecognitions: number;

  recentActivity: {
    description: string;
    timestamp: string;
  }[];

  speciesBreakdown: {
    species: string;
    count: number;
  }[];
};

export async function getProjectStats(): Promise<ProjectStats> {
  return apiFetch<ProjectStats>("/stats");
}