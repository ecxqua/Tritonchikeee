import { useEffect, useState, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { createCardApi, listProjects } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { ArrowLeft, Save } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { PhotoUploadZone, savePhotos } from "@/components/photo-gallery";

function dataURLtoFile(dataurl: string, filename: string): File {
  const arr = dataurl.split(",");
  const mime = arr[0].match(/:(.*?);/)?.[1] || "image/jpeg";
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);

  while (n--) {
    u8arr[n] = bstr.charCodeAt(n);
  }

  return new File([u8arr], filename, { type: mime });
}

export function NewCard() {
  const [, setLocation] = useLocation();
  const searchString = useSearch();
  const params = new URLSearchParams(searchString);
  const defaultProjectId = params.get("projectId") ? Number(params.get("projectId")) : undefined;

  const { toast } = useToast();
  
  const [cardType, setCardType] = useState<"ИК-1" | "ИК-2" | "КВ-1" | "КВ-2">("ИК1");
  const [projectId, setProjectId] = useState<string>(defaultProjectId ? defaultProjectId.toString() : "none");
  const [data, setData] = useState<Record<string, any>>({});
  const [photos, setPhotos] = useState<string[]>([]);
  
  const [projects, setProjects] = useState<{ id: number; name: string }[]>([]);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    listProjects().then(setProjects);
  }, []);

  const handlePhotosChange = useCallback((newPhotos: string[]) => {
    setPhotos(newPhotos);
  }, []);

  const handleSave = async () => {
    const id = `NEWT-${Math.floor(Math.random() * 10000)
      .toString()
      .padStart(4, "0")}`;

    setIsSaving(true);

    try {
      if (photos.length === 0) {
        toast({ title: "Добавьте хотя бы одно фото", variant: "destructive" });
        return;
      }

      // convert base64 → File[]
      const files = photos.map((base64, i) =>
        dataURLtoFile(base64, `photo-${i}.jpg`)
      );

      const res = await createCardApi({
        cardType,
        species: "unknown",
        projectId: projectId && projectId !== "none" ? projectId : undefined,
        files,
        data: {
          idNumber: id,
          ...data,
        },
      });

      // if you still need local persistence
      savePhotos(res.id, photos);

      toast({ title: "Карточка успешно создана" });

      setLocation(
        projectId && projectId !== "none"
          ? `/projects/${projectId}`
          : "/projects"
      );
    } catch (e) {
      toast({ title: "Ошибка при создании", variant: "destructive" });
      console.log(e);
    } finally {
      setIsSaving(false);
    }
  };

  const updateField = (field: string, value: string) => {
    setData(prev => ({ ...prev, [field]: value }));
  };

  const renderFields = () => {
    const commonFields = (
      <div className="space-y-2 col-span-full mb-4 pb-4 border-b">
        <Label htmlFor="idNumber" className="text-muted-foreground">ID особи (будет сгенерирован автоматически)</Label>
        <Input id="idNumber" disabled placeholder="NEWT-XXXX" className="bg-muted" />
      </div>
    );

    if (cardType === "ИК-1") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          <div className="space-y-2"><Label>Дата заполнения</Label><Input type="date" onChange={e => updateField("dateFilled", e.target.value)} /></div>
          <div className="space-y-2"><Label>Точная дата рождения</Label><Input type="date" onChange={e => updateField("exactBirthDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина тела (L), мм</Label><Input type="number" onChange={e => updateField("bodyLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина хвоста (Lcd), мм</Label><Input type="number" onChange={e => updateField("tailLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Вес (г)</Label><Input type="number" onChange={e => updateField("weight", e.target.value)} /></div>
          <div className="space-y-2"><Label>Пол</Label><Input onChange={e => updateField("sex", e.target.value)} /></div>
          <div className="space-y-2"><Label>Регион происхождения</Label><Input onChange={e => updateField("regionOfOrigin", e.target.value)} /></div>
          <div className="space-y-2"><Label>Измерительный прибор</Label><Input onChange={e => updateField("measurementDevice", e.target.value)} /></div>
          <div className="space-y-2"><Label>Марка весов</Label><Input onChange={e => updateField("scaleBrand", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea onChange={e => updateField("notes", e.target.value)} /></div>
        </div>
      );
    } else if (cardType === "ИК-2") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          <div className="space-y-2"><Label>Дата заполнения</Label><Input type="date" onChange={e => updateField("dateFilled", e.target.value)} /></div>
          <div className="space-y-2"><Label>Дата выпуска</Label><Input type="date" onChange={e => updateField("releaseDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>ID Отца</Label><Input onChange={e => updateField("fatherId", e.target.value)} /></div>
          <div className="space-y-2"><Label>ID Матери</Label><Input onChange={e => updateField("motherId", e.target.value)} /></div>
          <div className="space-y-2"><Label>Общая длина (L+Lcd), см</Label><Input type="number" onChange={e => updateField("totalLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Вес (г)</Label><Input type="number" onChange={e => updateField("weight", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Название водоема</Label><Input onChange={e => updateField("waterBodyName", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea onChange={e => updateField("notes", e.target.value)} /></div>
        </div>
      );
    } else if (cardType === "КВ-1") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          <div className="space-y-2"><Label>Дата встречи</Label><Input type="date" onChange={e => updateField("encounterDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>Время встречи</Label><Input type="time" onChange={e => updateField("encounterTime", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина тела (L), мм</Label><Input type="number" onChange={e => updateField("bodyLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина хвоста (Lcd), мм</Label><Input type="number" onChange={e => updateField("tailLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Вес (г)</Label><Input type="number" onChange={e => updateField("weight", e.target.value)} /></div>
          <div className="space-y-2"><Label>Пол</Label><Input onChange={e => updateField("sex", e.target.value)} /></div>
          <div className="space-y-2"><Label>Номер фото брюшка</Label><Input onChange={e => updateField("bellyPhotoNumber", e.target.value)} /></div>
          <div className="space-y-2"><Label>Статус</Label>
            <Select onValueChange={val => updateField("status", val)}>
              <SelectTrigger><SelectValue placeholder="Выберите статус" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="alive">Жив</SelectItem>
                <SelectItem value="dead">Мертв</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2"><Label>Номер водоема</Label><Input onChange={e => updateField("waterBodyNumber", e.target.value)} /></div>
          <div className="space-y-2"><Label>Измерительный прибор</Label><Input onChange={e => updateField("measurementDevice", e.target.value)} /></div>
          <div className="space-y-2"><Label>Марка весов</Label><Input onChange={e => updateField("scaleBrand", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea onChange={e => updateField("notes", e.target.value)} /></div>
        </div>
      );
    } else {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          <div className="space-y-2"><Label>Дата встречи</Label><Input type="date" onChange={e => updateField("encounterDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>Время встречи</Label><Input type="time" onChange={e => updateField("encounterTime", e.target.value)} /></div>
          <div className="space-y-2"><Label>Общая длина (L+Lcd), см</Label><Input type="number" onChange={e => updateField("totalLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Статус</Label>
            <Select onValueChange={val => updateField("status", val)}>
              <SelectTrigger><SelectValue placeholder="Выберите статус" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="alive">Жив</SelectItem>
                <SelectItem value="dead">Мертв</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2 col-span-full"><Label>Название водоема</Label><Input onChange={e => updateField("waterBodyName", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea onChange={e => updateField("notes", e.target.value)} /></div>
        </div>
      );
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <div>
          <Button variant="ghost" size="sm" onClick={() => window.history.back()} className="-ml-3 mb-2 text-muted-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" /> Назад
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">Новая карточка</h1>
        </div>
        <Button onClick={handleSave} disabled={isSaving} className="bg-primary text-primary-foreground gap-2">
          <Save className="w-4 h-4" /> {isSaving ? "Сохранение..." : "Сохранить"}
        </Button>
      </div>

      <Card>
        <CardHeader className="bg-muted/20 border-b">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <Label>Тип карточки</Label>
              <Select value={cardType} onValueChange={(val: any) => { setCardType(val); setData({}); }}>
                <SelectTrigger className="bg-background">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ИК-1">ИК-1 (Разведение в неволе)</SelectItem>
                  <SelectItem value="ИК-2">ИК-2 (Выпуск в природу)</SelectItem>
                  <SelectItem value="КВ-1">КВ-1 (Встреча в природе — подробная)</SelectItem>
                  <SelectItem value="КВ-2">КВ-2 (Встреча в природе — краткая)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label>Проект (необязательно)</Label>
              <Select value={projectId} onValueChange={setProjectId}>
                <SelectTrigger className="bg-background">
                  <SelectValue placeholder="Без проекта" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">Без проекта</SelectItem>
                  {projects?.map(p => (
                    <SelectItem key={p.id} value={p.id.toString()}>{p.name}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-6 space-y-8">
          {renderFields()}

          <div className="border-t pt-6">
            <CardTitle className="text-base mb-1">Фотографии брюшка</CardTitle>
            <p className="text-xs text-muted-foreground mb-4">Добавьте одну или несколько фотографий для распознавания</p>
            <PhotoUploadZone onChange={handlePhotosChange} />
            {photos.length > 0 && (
              <p className="text-xs text-muted-foreground mt-2">{photos.length} фото выбрано</p>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
