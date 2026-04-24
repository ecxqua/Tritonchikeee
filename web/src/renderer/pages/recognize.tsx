import { recognizeNewt, listProjects } from "@/lib/api";
import { useEffect, useState, useRef } from "react";
import { Link } from "wouter";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { UploadCloud, ScanSearch, CheckCircle2, XCircle, ChevronRight, Image as ImageIcon, PlusCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { PhotoGallery } from "@/components/photo-gallery";

export function Recognize() {
  const [photo, setPhoto] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [scope, setScope] = useState<"all" | "by_species" | "by_territory">("all");
  const [projectId, setProjectId] = useState<string>("all");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [expandedNewtId, setExpandedNewtId] = useState<string | null>(null);

  const [result, setResult] = useState<any>(null);
  const [isPending, setIsPending] = useState(false);

  const [projects, setProjects] = useState<{ id: number; name: string }[]>([]);

  useEffect(() => {
    listProjects().then(setProjects);
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setPhoto(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setExpandedNewtId(null);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setPhoto(file);
      setPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setExpandedNewtId(null);
    }
  };

  const runRecognition = async () => {
    if (!photo) return;

    setIsPending(true);
    setResult(null);

    const res = await recognizeNewt({
      photo,
      scope,
      projectId: projectId !== "all" ? Number(projectId) : undefined,
    });

    setResult(res);
    setIsPending(false);
  };

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Опознание особи</h1>
        <p className="text-muted-foreground">Загрузите фото брюшка для распознавания с помощью нейросети.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div className="lg:col-span-5 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Параметры поиска</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Область поиска</label>
                <Select value={scope} onValueChange={(v: any) => setScope(v)}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">По всей базе</SelectItem>
                    <SelectItem value="by_species">Только в пределах вида</SelectItem>
                    <SelectItem value="by_territory">Только на этой территории</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Ограничить проектом (необязательно)</label>
                <Select value={projectId} onValueChange={setProjectId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Все проекты" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Все проекты</SelectItem>
                    {projects?.map(p => (
                      <SelectItem key={p.id} value={p.id.toString()}>{p.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div 
                className={cn(
                  "border-2 border-dashed rounded-xl transition-all duration-200 flex flex-col items-center justify-center p-8 text-center cursor-pointer min-h-[300px]",
                  isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/30 hover:border-primary/50 hover:bg-muted/10",
                  previewUrl ? "border-primary/50 bg-black/5 p-2" : ""
                )}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
                onClick={() => !previewUrl && fileInputRef.current?.click()}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  className="hidden" 
                  accept="image/*" 
                  onChange={handleFileChange}
                />
                
                {previewUrl ? (
                  <div className="relative w-full h-full min-h-[280px] rounded-lg overflow-hidden group">
                    <img src={previewUrl} alt="Preview" className="w-full h-full object-contain" />
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <Button variant="secondary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                        Выбрать другое фото
                      </Button>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center text-primary mb-4">
                      <UploadCloud className="w-8 h-8" />
                    </div>
                    <h3 className="text-lg font-medium mb-1">Перетащите фото сюда</h3>
                    <p className="text-sm text-muted-foreground mb-4">или нажмите для выбора файла (JPEG, PNG)</p>
                    <Button variant="outline" size="sm">Выбрать файл</Button>
                  </>
                )}
              </div>
              
              <Button 
                className="w-full mt-6 gap-2" 
                size="lg" 
                disabled={!photo || isPending}
                onClick={runRecognition}
              >
                {isPending ? (
                  <>Сканирование...</>
                ) : (
                  <><ScanSearch className="w-5 h-5" /> Распознать особь</>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-7">
          {!result && !isPending && (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed rounded-xl p-12 bg-muted/10">
              <ImageIcon className="w-16 h-16 mb-4 opacity-20" />
              <p className="text-lg">Результаты появятся здесь</p>
              <p className="text-sm mt-2 opacity-70 max-w-md text-center">
                Загрузите фотографию брюшка тритона и нажмите "Распознать" для поиска совпадений в базе.
              </p>
            </div>
          )}

          {isPending && (
            <Card className="h-full border-primary/20 bg-primary/5 animate-pulse flex flex-col items-center justify-center p-12 text-center">
              <ScanSearch className="w-16 h-16 text-primary mb-6 animate-bounce" />
              <h3 className="text-xl font-bold mb-2">Нейросеть анализирует изображение...</h3>
              <p className="text-muted-foreground">Это может занять несколько секунд</p>
              <div className="w-64 max-w-full mt-8 h-2 bg-primary/20 rounded-full overflow-hidden">
                <div className="h-full bg-primary animate-[progress_2s_ease-in-out_infinite]" style={{ width: '50%' }}></div>
              </div>
            </Card>
          )}

          {result && !isPending && (
            <div className="space-y-4 animate-in slide-in-from-right-4 duration-500">
              <Card>
                <CardHeader className={cn(
                  "border-b",
                  result.status === "not_found" ? "bg-accent/20" : "bg-primary/10"
                )}>
                  <div className="flex items-start gap-4">
                    {result.status === "not_found" ? (
                      <XCircle className="w-8 h-8 text-accent-foreground mt-1 shrink-0" />
                    ) : (
                      <CheckCircle2 className="w-8 h-8 text-primary mt-1 shrink-0" />
                    )}
                    <div>
                      <CardTitle className="text-2xl">
                        {result.status === "not_found" ? "Совпадений не найдено" : "Найдено совпадение"}
                      </CardTitle>
                      <CardDescription className="mt-1 text-base">
                        {result.status === "not_found" 
                          ? "Эта особь не зарегистрирована в базе данных." 
                          : `Нейросеть нашла ${result.matches?.length || 0} похожих особей.`}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>

                {result.status !== "not_found" && result.matches && result.matches.length > 0 && (
                  <CardContent className="p-0">
                    <div className="divide-y">
                      {result.matches.map((match, i) => (
                        <div key={i}>
                          <div
                            className="p-4 hover:bg-muted/10 transition-colors flex items-center gap-4 cursor-pointer"
                            onClick={() => setExpandedNewtId(expandedNewtId === String(match.newtId) ? null : String(match.newtId))}
                          >
                            <div className="w-14 h-14 rounded overflow-hidden bg-black/10 shrink-0 border border-border/50">
                              <img src={match.photoUrl || "https://placehold.co/100x100?text=?"} alt="Match" className="w-full h-full object-cover" />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-bold font-mono text-base">{match.newtId}</span>
                                {i === 0 && <span className="bg-primary/20 text-primary text-xs font-bold px-2 py-0.5 rounded shrink-0">Лучшее совпадение</span>}
                              </div>
                              <div className="text-sm text-muted-foreground">Уверенность: {match.confidence.toFixed(1)}%</div>
                              <Progress value={match.confidence} className="h-1.5 mt-1.5" />
                            </div>
                            <div className="flex items-center gap-1 shrink-0">
                              <Link href={`/newts/${match.newtId}`} onClick={e => e.stopPropagation()}>
                                <Button variant="outline" size="sm" className="text-xs">
                                  Карточка <ChevronRight className="w-3 h-3 ml-1" />
                                </Button>
                              </Link>
                            </div>
                          </div>
                          {expandedNewtId === String(match.newtId) && (
                            <div className="px-4 pb-4 bg-muted/5 border-t">
                              <p className="text-xs text-muted-foreground py-2 mb-2">Фотографии особи {match.newtId}:</p>
                              <PhotoGallery newtId={String(match.newtId)} editable={false} />
                              <Link href={`/newts/${match.newtId}`}>
                                <Button size="sm" className="w-full mt-3 gap-2">
                                  Открыть полную карточку <ChevronRight className="w-4 h-4" />
                                </Button>
                              </Link>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                )}
              </Card>

              <Card className="border-dashed border-2 border-muted-foreground/20 bg-muted/5">
                <CardContent className="p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div>
                    <p className="font-medium text-sm">
                      {result.status === "not_found" ? "Особь не найдена в базе" : "Это другая особь?"}
                    </p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      Если тритон ещё не зарегистрирован — добавьте его в базу
                    </p>
                  </div>
                  <Link href="/cards/new">
                    <Button variant="outline" size="sm" className="gap-2 shrink-0 border-primary/30 text-primary hover:bg-primary/10">
                      <PlusCircle className="w-4 h-4" /> Добавить новую особь
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
