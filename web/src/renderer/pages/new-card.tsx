import { useEffect, useState, useCallback } from "react";
import { useLocation, useSearch } from "wouter";
import { createCardApi, listNewts, listProjects } from "@/lib/api";
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

type CardType = "ИК-1" | "ИК-2" | "КВ-1" | "КВ-2";

export function NewCard() {
  const [, setLocation] = useLocation();
  const searchString = useSearch();
  const params = new URLSearchParams(searchString);
  const defaultProjectId = params.get("projectId") ? Number(params.get("projectId")) : undefined;

  const { toast } = useToast();
  
  const today = new Date().toISOString().split("T")[0];
  const [cardType, setCardType] = useState<CardType>("ИК-1");
  const [projectId, setProjectId] = useState<string>(defaultProjectId ? defaultProjectId.toString() : "none");
  const [data, setData] = useState<Record<string, any>>({});
  const [photos, setPhotos] = useState<string[]>([]);
  
  const [projects, setProjects] = useState<{ id: number; name: string }[]>([]);
  const [registeredIds, setRegisteredIds] = useState<string[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [regionMode, setRegionMode] = useState<"preset" | "custom">("preset");
  const [customRegion, setCustomRegion] = useState("");

  const commonRegions = [
    "Московская область",
    "Ленинградская область",
    "Тверская область",
    "Калужская область",
    "Ярославская область",
  ];

  useEffect(() => {
    const loadData = async () => {
      const projectList = await listProjects();
      setProjects(projectList);

      const idsByProject = await Promise.all(
        projectList.map(async (project) => {
          try {
            const newts = await listNewts({ projectId: project.id });
            return newts.map((n) => n.id);
          } catch {
            return [];
          }
        }),
      );

      setRegisteredIds(Array.from(new Set(idsByProject.flat())).sort());
    };

    loadData().catch(() => {
      setRegisteredIds([]);
    });
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
      const preparedData = {
        ...data,
        exactBirthDate: data.exactBirthDate || data.conditionalBirthDate || "стандарт",
        measurementDevice: data.measurementDevice || "стандарт",
        scaleBrand: data.scaleBrand || "стандарт",
        notes: data.notes || "нет примечаний",
      };

      const missingField = getRequiredFieldsForCard(cardType).find(
        ({ key }) => !preparedData[key] || String(preparedData[key]).trim() === "",
      );

      if (missingField) {
        toast({
          title: `Заполните обязательное поле: ${missingField.label}`,
          variant: "destructive",
        });
        return;
      }

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
          ...preparedData,
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
    setData((prev) => ({ ...prev, [field]: value }));
  };

  const getRequiredFieldsForCard = (type: CardType) => {
    if (type === "ИК-1") {
      return [
        { key: "species", label: "Вид" },
        { key: "dateFilled", label: "Дата заполнения карточки" },
        { key: "bodyLength", label: "Длина тела" },
        { key: "tailLength", label: "Длина хвоста" },
        { key: "conditionalBirthDate", label: "Условный год рождения" },
        { key: "photoNumber", label: "Номер фото" },
        { key: "regionOfOrigin", label: "Регион происхождения" },
      ];
    }
    if (type === "ИК-2") {
      return [
        { key: "species", label: "Вид" },
        { key: "dateFilled", label: "Дата заполнения карточки" },
        { key: "releaseDate", label: "Дата выпуска в водоем" },
        { key: "fatherId", label: "ID самца (родитель)" },
        { key: "motherId", label: "ID самки (родитель)" },
        { key: "totalLength", label: "Общая длина (L+Lcd), см" },
        { key: "weight", label: "Масса, г" },
        { key: "waterBodyName", label: "Название водоема" },
      ];
    }
    if (type === "КВ-1") {
      return [
        { key: "species", label: "Вид" },
        { key: "encounterDate", label: "Дата встречи" },
        { key: "encounterTime", label: "Время встречи" },
        { key: "bodyLength", label: "Длина тела" },
        { key: "tailLength", label: "Длина хвоста" },
        { key: "weight", label: "Масса, г" },
        { key: "sex", label: "Пол" },
        { key: "bellyPhotoNumber", label: "Номер фото брюшной стороны" },
        { key: "status", label: "Статус" },
        { key: "waterBodyNumber", label: "Номер водоема" },
      ];
    }
    return [
      { key: "species", label: "Вид" },
      { key: "encounterDate", label: "Дата встречи" },
      { key: "encounterTime", label: "Время встречи" },
      { key: "totalLength", label: "Общая длина (L+Lcd), см" },
      { key: "status", label: "Статус" },
      { key: "waterBodyName", label: "Название водоема" },
    ];
  };

  const resetDataForCardType = (type: CardType) => {
    setRegionMode("preset");
    setCustomRegion("");
    if (type === "ИК-1") {
      return {
        species: "тритон карелина",
        dateFilled: today,
        bodyLength: "70",
        tailLength: "40",
        weight: "30",
        sex: "",
        exactBirthDate: "",
        conditionalBirthDate: "",
        photoNumber: "",
        regionOfOrigin: "Московская область",
        measurementDevice: "стандарт",
        scaleBrand: "стандарт",
        notes: "нет примечаний",
      };
    }
    if (type === "ИК-2") {
      return {
        species: "тритон карелина",
        dateFilled: today,
        releaseDate: today,
        fatherId: "данные отсутствуют",
        motherId: "данные отсутствуют",
        totalLength: "",
        weight: "30",
        waterBodyName: "",
        notes: "нет примечаний",
      };
    }
    if (type === "КВ-1") {
      return {
        species: "тритон карелина",
        encounterDate: today,
        encounterTime: "",
        bodyLength: "70",
        tailLength: "40",
        weight: "30",
        sex: "",
        bellyPhotoNumber: "",
        status: "",
        waterBodyNumber: "",
        measurementDevice: "стандарт",
        scaleBrand: "стандарт",
        notes: "нет примечаний",
      };
    }
    return {
      species: "тритон карелина",
      encounterDate: today,
      encounterTime: "",
      totalLength: "",
      status: "",
      waterBodyName: "",
      notes: "нет примечаний",
    };
  };

  useEffect(() => {
    setData(resetDataForCardType("ИК-1"));
  }, []);

  const renderFields = () => {
    const commonFields = (
      <div className="space-y-2 col-span-full mb-4 pb-4 border-b">
        <Label htmlFor="idNumber" className="text-muted-foreground">ID особи (будет сгенерирован автоматически)</Label>
        <Input id="idNumber" disabled placeholder="NEWT-XXXX" className="bg-muted" />
      </div>
    );

    const speciesField = (
      <div className="space-y-2">
        <Label>Вид *</Label>
        <Select value={data.species || "тритон карелина"} onValueChange={(val) => updateField("species", val)}>
          <SelectTrigger><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="тритон карелина">Тритон Карелина</SelectItem>
            <SelectItem value="ребристый">Ребристый</SelectItem>
          </SelectContent>
        </Select>
      </div>
    );

    const parentField = (kind: "father" | "mother") => {
      const key = kind === "father" ? "fatherId" : "motherId";
      const label = kind === "father" ? "ID самца (родитель) *" : "ID самки (родитель) *";
      const currentValue = (data[key] || "") as string;
      const filteredIds = registeredIds
        .filter((id) => id.toLowerCase().includes(currentValue.toLowerCase()))
        .slice(0, 20);

      return (
        <div className="space-y-2">
          <Label>{label}</Label>
          <Input
            value={currentValue}
            placeholder="Введите ID или выберите ниже"
            onChange={(e) => updateField(key, e.target.value)}
          />
          <div className="max-h-32 overflow-y-auto rounded border p-2 space-y-1 bg-muted/20">
            <Button type="button" variant="outline" size="sm" className="w-full justify-start" onClick={() => updateField(key, "данные отсутствуют")}>
              данные отсутствуют
            </Button>
            {filteredIds.map((id) => (
              <Button
                key={`${key}-${id}`}
                type="button"
                variant="ghost"
                size="sm"
                className="w-full justify-start"
                onClick={() => updateField(key, id)}
              >
                {id}
              </Button>
            ))}
          </div>
        </div>
      );
    };

    if (cardType === "ИК-1") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          {speciesField}
          <div className="space-y-2"><Label>Дата заполнения карточки *</Label><Input type="date" value={data.dateFilled || today} onChange={(e) => updateField("dateFilled", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина тела (L), мм *</Label><Input type="number" value={data.bodyLength || "70"} onChange={(e) => updateField("bodyLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина хвоста (Lcd), мм *</Label><Input type="number" value={data.tailLength || "40"} onChange={(e) => updateField("tailLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Масса, г.</Label><Input type="number" value={data.weight || "30"} onChange={(e) => updateField("weight", e.target.value)} /></div>
          <div className="space-y-2"><Label>Пол</Label>
            <Select value={data.sex || ""} onValueChange={(val) => updateField("sex", val)}>
              <SelectTrigger><SelectValue placeholder="Выберите пол" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="мужской">Мужской</SelectItem>
                <SelectItem value="женский">Женский</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2"><Label>Точный год рождения (дд.мм.гггг)</Label><Input placeholder="дд.мм.гггг" value={data.exactBirthDate || ""} onChange={(e) => updateField("exactBirthDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>Условный год рождения (дд.мм.гггг) *</Label><Input placeholder="дд.мм.гггг" value={data.conditionalBirthDate || ""} onChange={(e) => updateField("conditionalBirthDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>Номер фото индивидуального рисунка *</Label><Input value={data.photoNumber || ""} onChange={(e) => updateField("photoNumber", e.target.value)} /></div>
          <div className="space-y-2">
            <Label>Регион происхождения особи *</Label>
            <Select
              value={regionMode === "custom" ? "custom" : (data.regionOfOrigin || "Московская область")}
              onValueChange={(val) => {
                if (val === "custom") {
                  setRegionMode("custom");
                  updateField("regionOfOrigin", customRegion);
                } else {
                  setRegionMode("preset");
                  updateField("regionOfOrigin", val);
                }
              }}
            >
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>
                {commonRegions.map((region) => (
                  <SelectItem key={region} value={region}>{region}</SelectItem>
                ))}
                <SelectItem value="custom">Другое (ввести вручную)</SelectItem>
              </SelectContent>
            </Select>
            {regionMode === "custom" && (
              <Input
                placeholder="Введите регион"
                value={customRegion}
                onChange={(e) => {
                  setCustomRegion(e.target.value);
                  updateField("regionOfOrigin", e.target.value);
                }}
              />
            )}
          </div>
          <div className="space-y-2"><Label>Марка устройства для измерения длины</Label><Input value={data.measurementDevice || "стандарт"} onChange={(e) => updateField("measurementDevice", e.target.value)} /></div>
          <div className="space-y-2"><Label>Марка весов</Label><Input value={data.scaleBrand || "стандарт"} onChange={(e) => updateField("scaleBrand", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea value={data.notes || "нет примечаний"} onChange={(e) => updateField("notes", e.target.value)} /></div>
        </div>
      );
    }

    if (cardType === "ИК-2") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          {speciesField}
          <div className="space-y-2"><Label>Дата заполнения карточки *</Label><Input type="date" value={data.dateFilled || today} onChange={(e) => updateField("dateFilled", e.target.value)} /></div>
          <div className="space-y-2"><Label>Дата выпуска в водоем *</Label><Input type="date" value={data.releaseDate || today} onChange={(e) => updateField("releaseDate", e.target.value)} /></div>
          {parentField("father")}
          {parentField("mother")}
          <div className="space-y-2"><Label>Общая длина (L+Lcd), см *</Label><Input type="number" value={data.totalLength || ""} onChange={(e) => updateField("totalLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Масса, г. *</Label><Input type="number" value={data.weight || "30"} onChange={(e) => updateField("weight", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Название водоема *</Label><Input value={data.waterBodyName || ""} onChange={(e) => updateField("waterBodyName", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea value={data.notes || "нет примечаний"} onChange={(e) => updateField("notes", e.target.value)} /></div>
        </div>
      );
    }

    if (cardType === "КВ-1") {
      return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {commonFields}
          {speciesField}
          <div className="space-y-2"><Label>Дата встречи (дд.мм.гггг) *</Label><Input type="date" value={data.encounterDate || today} onChange={(e) => updateField("encounterDate", e.target.value)} /></div>
          <div className="space-y-2"><Label>Время встречи *</Label><Input type="time" value={data.encounterTime || ""} onChange={(e) => updateField("encounterTime", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина тела (L), мм *</Label><Input type="number" value={data.bodyLength || "70"} onChange={(e) => updateField("bodyLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Длина хвоста (Lcd), мм *</Label><Input type="number" value={data.tailLength || "40"} onChange={(e) => updateField("tailLength", e.target.value)} /></div>
          <div className="space-y-2"><Label>Масса, г. *</Label><Input type="number" value={data.weight || "30"} onChange={(e) => updateField("weight", e.target.value)} /></div>
          <div className="space-y-2"><Label>Пол *</Label>
            <Select value={data.sex || ""} onValueChange={(val) => updateField("sex", val)}>
              <SelectTrigger><SelectValue placeholder="Выберите пол" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="мужской">Мужской</SelectItem>
                <SelectItem value="женский">Женский</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2"><Label>Номер фото брюшной стороны *</Label><Input value={data.bellyPhotoNumber || ""} onChange={(e) => updateField("bellyPhotoNumber", e.target.value)} /></div>
          <div className="space-y-2"><Label>Статус (жив/мертв) *</Label>
            <Select value={data.status || ""} onValueChange={(val) => updateField("status", val)}>
              <SelectTrigger><SelectValue placeholder="Выберите статус" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="alive">Жив</SelectItem>
                <SelectItem value="dead">Мертв</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2"><Label>Номер водоема *</Label><Input value={data.waterBodyNumber || ""} onChange={(e) => updateField("waterBodyNumber", e.target.value)} /></div>
          <div className="space-y-2"><Label>Марка устройства для измерения длины</Label><Input value={data.measurementDevice || "стандарт"} onChange={(e) => updateField("measurementDevice", e.target.value)} /></div>
          <div className="space-y-2"><Label>Марка весов</Label><Input value={data.scaleBrand || "стандарт"} onChange={(e) => updateField("scaleBrand", e.target.value)} /></div>
          <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea value={data.notes || "нет примечаний"} onChange={(e) => updateField("notes", e.target.value)} /></div>
        </div>
      );
    }

    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {commonFields}
        {speciesField}
        <div className="space-y-2"><Label>Дата встречи (дд.мм.гггг) *</Label><Input type="date" value={data.encounterDate || today} onChange={(e) => updateField("encounterDate", e.target.value)} /></div>
        <div className="space-y-2"><Label>Время встречи *</Label><Input type="time" value={data.encounterTime || ""} onChange={(e) => updateField("encounterTime", e.target.value)} /></div>
        <div className="space-y-2"><Label>Общая длина (L+Lcd), см *</Label><Input type="number" value={data.totalLength || ""} onChange={(e) => updateField("totalLength", e.target.value)} /></div>
        <div className="space-y-2"><Label>Статус (жив/мертв) *</Label>
          <Select value={data.status || ""} onValueChange={(val) => updateField("status", val)}>
            <SelectTrigger><SelectValue placeholder="Выберите статус" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="alive">Жив</SelectItem>
              <SelectItem value="dead">Мертв</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-2 col-span-full"><Label>Название водоема *</Label><Input value={data.waterBodyName || ""} onChange={(e) => updateField("waterBodyName", e.target.value)} /></div>
        <div className="space-y-2 col-span-full"><Label>Примечания</Label><Textarea value={data.notes || "нет примечаний"} onChange={(e) => updateField("notes", e.target.value)} /></div>
      </div>
    );
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
              <Select value={cardType} onValueChange={(val: any) => { setCardType(val); setData(resetDataForCardType(val)); }}>
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
