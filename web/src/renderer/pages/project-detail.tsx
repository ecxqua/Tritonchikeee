import { useParams, Link } from "wouter";
import {
  getProject,
  listNewts,
  updateProjectApi,
  deleteProjectApi
} from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ArrowLeft, Edit, Trash2, Tag, MapPin, Eye, FilePlus, Search, ArrowUpDown, CalendarDays, AlignLeft, X, ZoomIn, ChevronLeft, ChevronRight } from "lucide-react";
import { format } from "date-fns";
import { ru } from "date-fns/locale";
import { useEffect, useState, useMemo } from "react";
import { useToast } from "@/hooks/use-toast";
import { loadPhotos } from "@/components/photo-gallery";
import { cn } from "@/lib/utils";

type SortField = "date_desc" | "date_asc" | "name_asc" | "name_desc";

interface PhotoPreviewProps {
  newtId: string;
  newtLabel: string;
  onClose: () => void;
}

function PhotoPreviewModal({ newtId, newtLabel, onClose }: PhotoPreviewProps) {
  const photos = loadPhotos(String(newtId));
  const [idx, setIdx] = useState(0);

  if (photos.length === 0) {
    return (
      <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center" onClick={onClose}>
        <div className="bg-background rounded-xl p-8 text-center shadow-2xl max-w-sm" onClick={e => e.stopPropagation()}>
          <div className="text-muted-foreground mb-4">
            <Eye className="w-12 h-12 mx-auto mb-2 opacity-30" />
            <p className="font-medium">Нет фотографий</p>
            <p className="text-sm mt-1">Для особи <span className="font-mono">{newtLabel}</span> фото не загружены</p>
          </div>
          <Link href={`/newts/${newtId}`}>
            <Button size="sm" onClick={onClose}>Открыть карточку</Button>
          </Link>
          <Button variant="ghost" size="sm" className="ml-2" onClick={onClose}>Закрыть</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center" onClick={onClose}>
      <button className="absolute top-4 right-4 text-white/70 hover:text-white z-10" onClick={onClose}>
        <X className="w-7 h-7" />
      </button>
      <div className="absolute top-4 left-4 text-white/70 text-sm font-mono">{newtLabel}</div>
      {photos.length > 1 && (
        <>
          <button
            className="absolute left-4 text-white/70 hover:text-white"
            onClick={e => { e.stopPropagation(); setIdx(i => (i > 0 ? i - 1 : photos.length - 1)); }}
          >
            <ChevronLeft className="w-9 h-9" />
          </button>
          <button
            className="absolute right-4 text-white/70 hover:text-white"
            onClick={e => { e.stopPropagation(); setIdx(i => (i < photos.length - 1 ? i + 1 : 0)); }}
          >
            <ChevronRight className="w-9 h-9" />
          </button>
        </>
      )}
      <img
        src={photos[idx]}
        alt="Просмотр"
        className="max-w-[85vw] max-h-[85vh] object-contain rounded-lg shadow-2xl"
        onClick={e => e.stopPropagation()}
      />
      <div className="absolute bottom-4 text-white/50 text-sm">{idx + 1} / {photos.length}</div>
    </div>
  );
}

export function ProjectDetail() {
  const params = useParams();
  const projectId = Number(params.projectId);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const [project, setProject] = useState<Project | null>(null);
  const [newts, setNewts] = useState<Newt[]>([]);

  const [projectLoading, setProjectLoading] = useState(true);
  const [newtsLoading, setNewtsLoading] = useState(true);

  useEffect(() => {
    if (!projectId) return;

    setProjectLoading(true);
    setNewtsLoading(true);

    getProject(projectId).then((res) => {
      setProject(res);
      setProjectLoading(false);
    });

    listNewts({ projectId }).then((res) => {
      setNewts(res);
      setNewtsLoading(false);
    });
  }, [projectId]);

  const [isEditing, setIsEditing] = useState(false);
  const [editData, setEditData] = useState({ name: "", description: "", species: "", territory: "" });

  const [search, setSearch] = useState("");
  const [cardTypeFilter, setCardTypeFilter] = useState("all");
  const [sexFilter, setSexFilter] = useState("all");
  const [sortBy, setSortBy] = useState<SortField>("date_desc");

  const [previewNewtId, setPreviewNewtId] = useState<string | null>(null);

  const handleEditClick = () => {
    if (project) {
      setEditData({
        name: project.name,
        description: project.description || "",
        species: project.species || "",
        territory: project.territory || ""
      });
      setIsEditing(true);
    }
  };

  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await updateProjectApi({
        projectId,
        data: {
          ...editData,
          species: editData.species
            ? editData.species.split(",").map(s => s.trim()).filter(Boolean)
            : undefined,
          territory: editData.territory
            ? editData.territory.split(",").map(s => s.trim()).filter(Boolean)
            : undefined,
        }
      });

      setProject(prev =>
        prev
          ? {
              ...prev,
              ...editData,
              species: editData.species,
              territory: editData.territory,
            }
          : prev
      );
      setIsEditing(false);

      toast({ title: "Проект обновлен" });
    } finally {
      setSaving(false);
    }
  };

  const [redirect, setRedirect] = useState(false);

  const handleDelete = async () => {
    await deleteProjectApi(projectId);
    toast({ title: "Проект удален" });
    setRedirect(true);
  };

  const filteredNewts = useMemo(() => {
    let result = newts ?? [];
    if (cardTypeFilter !== "all") result = result.filter(n => n.cardType === cardTypeFilter);
    if (sexFilter !== "all") result = result.filter(n => n.sex === sexFilter);
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      result = result.filter(n =>
        String(n.id).toLowerCase().includes(q) ||
        (n.sex || "").toLowerCase().includes(q) ||
        (n.status || "").toLowerCase().includes(q) ||
        (n.cardType || "").toLowerCase().includes(q)
      );
    }

    const getIdNumber = (id: string | number): number => {
      const match = String(id).match(/-(\d+)$/);
      return match ? Number(match[1]) : 0;
    };

    return [...result].sort((a, b) => {
      switch (sortBy) {
        case "date_desc":
          return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();

        case "date_asc":
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();

        case "name_asc":
          return getIdNumber(a.id) - getIdNumber(b.id);

        case "name_desc":
          return getIdNumber(b.id) - getIdNumber(a.id);

        default:
          return 0;
      }
    });
  }, [newts, cardTypeFilter, sexFilter, search, sortBy]);

  if (projectLoading) {
    return (
      <div className="p-8 max-w-7xl mx-auto space-y-8">
        <Skeleton className="h-10 w-48 mb-6" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full mt-8" />
      </div>
    );
  }

  if (!project) {
    return (
      <div className="p-8 text-center text-muted-foreground">
        Проект не найден
      </div>
    );
  }

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      {previewNewtId && (
        <PhotoPreviewModal
          newtId={previewNewtId}
          newtLabel={String(previewNewtId)}
          onClose={() => setPreviewNewtId(null)}
        />
      )}

      <Link href="/projects">
        <Button variant="ghost" size="sm" className="mb-2 -ml-3 text-muted-foreground hover:text-foreground">
          <ArrowLeft className="w-4 h-4 mr-2" /> Вернуться к списку
        </Button>
      </Link>

      <div className="flex flex-col md:flex-row justify-between gap-4 items-start">
        <div className="space-y-2 flex-1">
          {isEditing ? (
            <div className="space-y-4 max-w-xl">
              <Input 
                value={editData.name}
                onChange={e => setEditData({...editData, name: e.target.value})}
                className="text-2xl font-bold h-auto py-2"
              />
              <Input 
                value={editData.description}
                onChange={e => setEditData({...editData, description: e.target.value})}
                placeholder="Описание"
              />
              <div className="flex gap-4">
                <Input 
                  value={editData.species}
                  onChange={e => setEditData({...editData, species: e.target.value})}
                  placeholder="Целевой вид"
                />
                <Input 
                  value={editData.territory}
                  onChange={e => setEditData({...editData, territory: e.target.value})}
                  placeholder="Территория"
                />
              </div>
              <div className="flex gap-2 pt-2">
                <Button onClick={handleSave} disabled={saving}>
                  {saving ? "Сохранение..." : "Сохранить"}
                </Button>
                <Button variant="outline" onClick={() => setIsEditing(false)}>Отмена</Button>
              </div>
            </div>
          ) : (
            <>
              <h1 className="text-3xl font-bold tracking-tight">{project.name}</h1>
              {project.description && (
                <p className="text-muted-foreground text-lg max-w-3xl">{project.description}</p>
              )}
              <div className="flex flex-wrap gap-4 mt-4 pt-2">
                <div className="flex items-center text-sm font-medium bg-primary/10 text-primary px-3 py-1 rounded-full">
                  <Tag className="w-4 h-4 mr-2" />
                  {project.species || "Вид не указан"}
                </div>
                <div className="flex items-center text-sm font-medium bg-secondary/20 text-secondary-foreground px-3 py-1 rounded-full">
                  <MapPin className="w-4 h-4 mr-2" />
                  {project.territory || "Территория не указана"}
                </div>
              </div>
            </>
          )}
        </div>

        {!isEditing && (
          <div className="flex gap-2">
            <Button variant="outline" size="icon" onClick={handleEditClick} title="Редактировать проект">
              <Edit className="w-4 h-4 text-muted-foreground" />
            </Button>
            <Button onClick={handleDelete} variant="outline" size="icon" className="hover:bg-destructive/10 hover:text-destructive hover:border-destructive/20" title="Удалить проект">
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        )}
      </div>

      <div className="grid gap-6 mt-8">
        <div className="flex justify-between items-end">
          <div>
            <h2 className="text-xl font-bold tracking-tight">Зарегистрированные особи</h2>
            <p className="text-muted-foreground text-sm mt-1">Всего: {project.newtCount}</p>
          </div>
          <Link href={`/cards/new?projectId=${project.id}`}>
            <Button size="sm" className="gap-2">
              <FilePlus className="w-4 h-4" /> Добавить особь
            </Button>
          </Link>
        </div>

        {(newts?.length ?? 0) > 0 && (
          <Card>
            <CardContent className="p-4 space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Поиск по ID, полу, статусу..."
                  value={search}
                  onChange={e => setSearch(e.target.value)}
                  className="pl-9"
                />
              </div>
              <div className="flex flex-wrap gap-3">
                <div className="flex-1 min-w-[140px]">
                  <Select value={cardTypeFilter} onValueChange={setCardTypeFilter}>
                    <SelectTrigger className="h-8 text-sm">
                      <SelectValue placeholder="Тип карточки" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Все типы</SelectItem>
                      <SelectItem value="ИК-1">ИК-1</SelectItem>
                      <SelectItem value="ИК-2">ИК-2</SelectItem>
                      <SelectItem value="КВ-1">КВ-1</SelectItem>
                      <SelectItem value="КВ-2">КВ-2</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex-1 min-w-[140px]">
                  <Select value={sexFilter} onValueChange={setSexFilter}>
                    <SelectTrigger className="h-8 text-sm">
                      <SelectValue placeholder="Пол" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Любой пол</SelectItem>
                      <SelectItem value="male">Самец</SelectItem>
                      <SelectItem value="female">Самка</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex-1 min-w-[160px]">
                  <Select value={sortBy} onValueChange={(v: SortField) => setSortBy(v)}>
                    <SelectTrigger className="h-8 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="date_desc">Сначала новые</SelectItem>
                      <SelectItem value="date_asc">Сначала старые</SelectItem>
                      <SelectItem value="name_asc">ID по возрастанию</SelectItem>
                      <SelectItem value="name_desc">ID по убыванию</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                {(search || cardTypeFilter !== "all" || sexFilter !== "all") && (
                  <button
                    className="text-xs text-primary hover:underline self-center"
                    onClick={() => { setSearch(""); setCardTypeFilter("all"); setSexFilter("all"); }}
                  >
                    Сбросить
                  </button>
                )}
              </div>
              {(search || cardTypeFilter !== "all" || sexFilter !== "all") && (
                <p className="text-xs text-muted-foreground">Найдено: {filteredNewts.length} из {newts?.length ?? 0}</p>
              )}
            </CardContent>
          </Card>
        )}

        {newtsLoading ? (
          <div className="space-y-4">
            {[1, 2, 3].map(i => <Skeleton key={i} className="h-20 w-full" />)}
          </div>
        ) : filteredNewts.length === 0 && (newts?.length ?? 0) === 0 ? (
          <Card className="border-dashed bg-muted/10">
            <CardContent className="flex flex-col items-center justify-center p-12 text-center text-muted-foreground space-y-4">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
                <Eye className="w-6 h-6" />
              </div>
              <div>
                <p className="font-medium text-foreground">Нет записей</p>
                <p className="text-sm mt-1">В этом проекте пока нет зарегистрированных особей.</p>
              </div>
              <Link href={`/cards/new?projectId=${project.id}`}>
                <Button variant="outline" className="mt-2">Создать карточку</Button>
              </Link>
            </CardContent>
          </Card>
        ) : filteredNewts.length === 0 ? (
          <div className="py-8 text-center text-muted-foreground text-sm">
            Ничего не найдено. Попробуйте изменить фильтры.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredNewts.map((newt) => {
              const photos = loadPhotos(String(newt.id));
              return (
                <Card key={newt.id} className="hover:border-primary/50 transition-colors cursor-pointer hover-elevate group overflow-hidden">
                  {photos.length > 0 && (
                    <div
                      className="relative h-32 bg-black/5 overflow-hidden cursor-zoom-in"
                      onClick={() => setPreviewNewtId(String(newt.id))}
                    >
                      <img
                        src={photos[0]}
                        alt="Фото"
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      {photos.length > 1 && (
                        <div className="absolute bottom-1 right-1 bg-black/60 text-white text-xs px-1.5 py-0.5 rounded">
                          +{photos.length - 1}
                        </div>
                      )}
                      <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <ZoomIn className="w-6 h-6 text-white" />
                      </div>
                    </div>
                  )}
                  <Link href={`/newts/${newt.id}`}>
                    <CardHeader className="p-4 pb-2">
                      <div className="flex justify-between items-start">
                        <CardTitle className="text-base font-mono bg-muted px-2 py-1 rounded">ID: {newt.id}</CardTitle>
                        <div className="text-xs font-bold px-2 py-1 bg-accent/30 text-accent-foreground rounded">
                          {newt.cardType}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="p-4 pt-2">
                      <div className="flex justify-between text-sm text-muted-foreground mb-2">
                        <span>{newt.sex || "Пол неизвестен"}</span>
                        <span>{newt.status || "Статус неизвестен"}</span>
                      </div>
                      <div className="text-xs text-muted-foreground border-t pt-2 mt-2 flex items-center justify-between">
                        <span>Добавлен: {format(new Date(newt.createdAt), "dd.MM.yyyy")}</span>
                        {photos.length === 0 && (
                          <span className="text-muted-foreground/50 text-xs">Нет фото</span>
                        )}
                      </div>
                    </CardContent>
                  </Link>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
