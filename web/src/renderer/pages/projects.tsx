import { useEffect, useState, useMemo } from "react";
import { Link } from "wouter";
import {
  listProjects,
  listSpecies,
  listTerritories,
  createProject
} from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from "@/components/ui/dialog";
import { Search, FolderKanban, Plus, MapPin, Tag, ArrowUpDown, CalendarDays, Users, AlignLeft } from "lucide-react";
import { format } from "date-fns";
import { ru } from "date-fns/locale";

type SortField = "date_desc" | "date_asc" | "name_asc" | "name_desc" | "count_desc" | "count_asc";

export function Projects() {
  const [species, setSpecies] = useState<string>("all");
  const [territory, setTerritory] = useState<string>("all");
  const [search, setSearch] = useState<string>("");
  const [sortBy, setSortBy] = useState<SortField>("date_desc");
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [newProject, setNewProject] = useState({ name: "", description: "", species: "", territory: "" });
  
  const [projects, setProjects] = useState<any[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(true);

  const [speciesList, setSpeciesList] = useState<string[]>([]);
  const [territoriesList, setTerritoriesList] = useState<string[]>([]);

  const [isCreating, setIsCreating] = useState(false);

  useEffect(() => {
    (async () => {
      setProjectsLoading(true);

      const [p, s, t] = await Promise.all([
        listProjects(),
        listSpecies(),
        listTerritories(),
      ]);

      setProjects(p);
      setSpeciesList(s);
      setTerritoriesList(t);

      setProjectsLoading(false);
    })();
  }, []);

  const handleCreate = async () => {
    if (!newProject.name) return;

    setIsCreating(true);

    try {
      await createProject({
        ...newProject,
        species: newProject.species
          ? newProject.species.split(",").map(s => s.trim()).filter(Boolean)
          : [],
        territory: newProject.territory
          ? newProject.territory.split(",").map(s => s.trim()).filter(Boolean)
          : [],
      });

      setIsCreateOpen(false);
      setNewProject({ name: "", description: "", species: "", territory: "" });

      const updated = await listProjects();
      setProjects(updated);
    } finally {
      setIsCreating(false);
    }
  };

  const filtered = useMemo(() => {
    let result = projects ?? [];

    if (species !== "all") {
      result = result.filter(p =>
        (p.species || "").split(", ").includes(species)
      );
    }

    if (territory !== "all") {
      result = result.filter(p =>
        (p.territory || "").split(", ").includes(territory)
      );
    }

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      result = result.filter(p =>
        p.name.toLowerCase().includes(q) ||
        (p.description || "").toLowerCase().includes(q) ||
        (p.species || "").toLowerCase().includes(q) ||
        (p.territory || "").toLowerCase().includes(q)
      );
    }

    return [...result].sort((a, b) => {
      switch (sortBy) {
        case "date_desc": return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
        case "date_asc": return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        case "name_asc": return a.name.localeCompare(b.name, "ru");
        case "name_desc": return b.name.localeCompare(a.name, "ru");
        case "count_desc": return (b.newtCount ?? 0) - (a.newtCount ?? 0);
        case "count_asc": return (a.newtCount ?? 0) - (b.newtCount ?? 0);
        default: return 0;
      }
    });
  }, [projects, species, territory, search, sortBy]);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Проекты</h1>
          <p className="text-muted-foreground">Управление исследовательскими проектами и базами данных.</p>
        </div>
        
        <Dialog open={isCreateOpen} onOpenChange={setIsCreateOpen}>
          <DialogTrigger asChild>
            <Button className="gap-2">
              <Plus className="w-4 h-4" /> Новый проект
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Создать новый проект</DialogTitle>
              <DialogDescription>
                Добавьте информацию о новом исследовательском проекте.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="name">Название проекта *</Label>
                <Input 
                  id="name" 
                  value={newProject.name} 
                  onChange={e => setNewProject({...newProject, name: e.target.value})} 
                  placeholder="Например: Мониторинг популяции 2026"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Описание</Label>
                <Input 
                  id="description" 
                  value={newProject.description} 
                  onChange={e => setNewProject({...newProject, description: e.target.value})} 
                  placeholder="Краткое описание целей исследования"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="new-species">Целевой вид</Label>
                  <Input 
                    id="new-species" 
                    value={newProject.species} 
                    onChange={e => setNewProject({...newProject, species: e.target.value})} 
                    placeholder="Lissotriton vulgaris"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="new-territory">Территория</Label>
                  <Input 
                    id="new-territory" 
                    value={newProject.territory} 
                    onChange={e => setNewProject({...newProject, territory: e.target.value})} 
                    placeholder="Заповедник N"
                  />
                </div>
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCreateOpen(false)}>Отмена</Button>
              <Button onClick={handleCreate} disabled={!newProject.name || isCreating}>
                {isCreating ? "Создание..." : "Создать"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardContent className="p-4 space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Поиск по названию, описанию, виду, территории..."
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <Label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                <Tag className="w-3 h-3" /> Вид
              </Label>
              <Select value={species} onValueChange={setSpecies}>
                <SelectTrigger>
                  <SelectValue placeholder="Все виды" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Все виды</SelectItem>
                  {speciesList?.map((s) => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <Label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                <MapPin className="w-3 h-3" /> Территория
              </Label>
              <Select value={territory} onValueChange={setTerritory}>
                <SelectTrigger>
                  <SelectValue placeholder="Все территории" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">Все территории</SelectItem>
                  {territoriesList?.map((t) => (
                    <SelectItem key={t} value={t}>{t}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <Label className="mb-2 block text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                <ArrowUpDown className="w-3 h-3" /> Сортировка
              </Label>
              <Select value={sortBy} onValueChange={(v: SortField) => setSortBy(v)}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="date_desc">
                    <span className="flex items-center gap-2"><CalendarDays className="w-3.5 h-3.5" /> Сначала новые</span>
                  </SelectItem>
                  <SelectItem value="date_asc">
                    <span className="flex items-center gap-2"><CalendarDays className="w-3.5 h-3.5" /> Сначала старые</span>
                  </SelectItem>
                  <SelectItem value="name_asc">
                    <span className="flex items-center gap-2"><AlignLeft className="w-3.5 h-3.5" /> Название А–Я</span>
                  </SelectItem>
                  <SelectItem value="name_desc">
                    <span className="flex items-center gap-2"><AlignLeft className="w-3.5 h-3.5" /> Название Я–А</span>
                  </SelectItem>
                  <SelectItem value="count_desc">
                    <span className="flex items-center gap-2"><Users className="w-3.5 h-3.5" /> Больше особей</span>
                  </SelectItem>
                  <SelectItem value="count_asc">
                    <span className="flex items-center gap-2"><Users className="w-3.5 h-3.5" /> Меньше особей</span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          {(search || species !== "all" || territory !== "all") && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Найдено: {filtered.length}</span>
              <button
                onClick={() => { setSearch(""); setSpecies("all"); setTerritory("all"); }}
                className="text-primary hover:underline"
              >
                Сбросить фильтры
              </button>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
        {projectsLoading ? (
          Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-48 w-full rounded-xl" />
          ))
        ) : filtered.length === 0 ? (
          <div className="col-span-full py-12 text-center text-muted-foreground bg-muted/20 rounded-lg border border-dashed">
            Проекты не найдены. Измените фильтры или создайте новый.
          </div>
        ) : filtered.map((project) => (
          <Link key={project.id} href={`/projects/${project.id}`}>
            <Card className="h-full hover:border-primary/50 transition-all cursor-pointer group hover-elevate">
              <CardHeader className="pb-3">
                <div className="flex justify-between items-start">
                  <CardTitle className="text-xl group-hover:text-primary transition-colors line-clamp-2">
                    {project.name}
                  </CardTitle>
                  <div className="p-2 bg-primary/10 rounded-md text-primary">
                    <FolderKanban className="w-4 h-4" />
                  </div>
                </div>
                {project.description && (
                  <CardDescription className="line-clamp-2 mt-2">{project.description}</CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <div className="space-y-2 mt-2">
                  <div className="flex items-center text-sm text-muted-foreground gap-2">
                    <Tag className="w-3.5 h-3.5" />
                    <span className="truncate">{project.species || "Вид не указан"}</span>
                  </div>
                  <div className="flex items-center text-sm text-muted-foreground gap-2">
                    <MapPin className="w-3.5 h-3.5" />
                    <span className="truncate">{project.territory || "Территория не указана"}</span>
                  </div>
                  <div className="pt-4 flex items-center justify-between border-t mt-4">
                    <div className="text-xs font-medium">
                      {project.newtCount} {project.newtCount === 1 ? 'особь' : (project.newtCount ?? 0) > 1 && (project.newtCount ?? 0) < 5 ? 'особи' : 'особей'}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Создан {format(new Date(project.createdAt), "dd.MM.yyyy")}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
