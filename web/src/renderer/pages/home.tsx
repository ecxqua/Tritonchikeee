import { getProjectStats, type ProjectStats } from "@/lib/api";
import { useEffect, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FolderKanban, ScanSearch, FilePlus, Activity } from "lucide-react";
import { Link } from "wouter";
import { format } from "date-fns";
import { ru } from "date-fns/locale";
import { Skeleton } from "@/components/ui/skeleton";

export function Home() {
  const [stats, setStats] = useState<ProjectStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setIsLoading(true);

    getProjectStats().then((res) => {
      setStats(res);
      setIsLoading(false);
    });
  }, []);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">Обзор</h1>
        <p className="text-muted-foreground">Добро пожаловать в систему распознавания тритонов.</p>
      </div>

      {isLoading ? (
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      ) : stats ? (
        <div className="grid gap-4 md:grid-cols-3">
          <Card className="hover-elevate">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Всего проектов</CardTitle>
              <FolderKanban className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalProjects}</div>
            </CardContent>
          </Card>
          <Card className="hover-elevate">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Особей в базе</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalNewts}</div>
            </CardContent>
          </Card>
          <Card className="hover-elevate">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Распознаваний</CardTitle>
              <ScanSearch className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalRecognitions}</div>
            </CardContent>
          </Card>
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-3 mt-8">
        <Link href="/recognize" className="group">
          <Card className="h-full border-primary/20 bg-primary/5 hover:bg-primary/10 transition-colors cursor-pointer">
            <CardHeader>
              <ScanSearch className="h-8 w-8 text-primary mb-2 group-hover:scale-110 transition-transform" />
              <CardTitle>Опознание</CardTitle>
              <CardDescription>Загрузить фото для распознавания нейросетью</CardDescription>
            </CardHeader>
          </Card>
        </Link>
        <Link href="/projects" className="group">
          <Card className="h-full border-secondary/20 bg-secondary/5 hover:bg-secondary/10 transition-colors cursor-pointer">
            <CardHeader>
              <FolderKanban className="h-8 w-8 text-secondary mb-2 group-hover:scale-110 transition-transform" />
              <CardTitle>Проекты</CardTitle>
              <CardDescription>Просмотр базы данных проектов и особей</CardDescription>
            </CardHeader>
          </Card>
        </Link>
        <Link href="/cards/new" className="group">
          <Card className="h-full border-accent/40 bg-accent/10 hover:bg-accent/20 transition-colors cursor-pointer">
            <CardHeader>
              <FilePlus className="h-8 w-8 text-accent-foreground mb-2 group-hover:scale-110 transition-transform" />
              <CardTitle>Новая карточка</CardTitle>
              <CardDescription>Создать новую карточку особи (ИК-1, ИК-2, КВ-1, КВ-2)</CardDescription>
            </CardHeader>
          </Card>
        </Link>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7 mt-8">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Последняя активность</CardTitle>
            <CardDescription>История действий в системе</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[1, 2, 3].map(i => <Skeleton key={i} className="h-12 w-full" />)}
              </div>
            ) : stats?.recentActivity?.length ? (
              <div className="space-y-6">
                {stats.recentActivity.map((activity, i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div className="w-2 h-2 rounded-full bg-primary mt-1.5" />
                    <div className="flex-1 space-y-1">
                      <p className="text-sm font-medium leading-none">
                        {activity.description}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {format(new Date(activity.timestamp), "dd MMM yyyy, HH:mm", { locale: ru })}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground text-center py-8">
                Нет недавней активности
              </div>
            )}
          </CardContent>
        </Card>
        
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Статистика по видам</CardTitle>
            <CardDescription>Распределение особей в базе</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoading ? (
              <div className="space-y-4">
                {[1, 2].map(i => <Skeleton key={i} className="h-12 w-full" />)}
              </div>
            ) : stats?.speciesBreakdown?.length ? (
              <div className="space-y-4">
                {stats.speciesBreakdown.map((item, i) => (
                  <div key={i} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-sm bg-primary/60" />
                      <span className="text-sm font-medium">{item.species || "Неизвестный вид"}</span>
                    </div>
                    <span className="text-sm font-bold">{item.count}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground text-center py-8">
                Нет данных
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
