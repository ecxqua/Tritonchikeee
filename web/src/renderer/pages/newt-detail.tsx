import { useEffect, useState } from "react";
import { useParams, Link } from "wouter";
import {
  getNewt,
  getNewtCards,
  getNewtHistory,
  updateNewtCardApi
} from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { ArrowLeft, Save, Trash2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { PhotoGallery } from "@/components/photo-gallery";

export function NewtDetail() {
  const params = useParams();
  const newtId = params.newtId || "";
  const { toast } = useToast();

  const [newt, setNewt] = useState<Newt | null>(null);
  const [cards, setCards] = useState<NewtCard[]>([]);
  const [history, setHistory] = useState<HistoryRecord[]>([]);

  const [newtLoading, setNewtLoading] = useState(true);
  const [cardLoading, setCardLoading] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(true);

  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editData, setEditData] = useState<Record<number, Record<string, any>>>({});

  useEffect(() => {
    if (!newtId) return;

    setNewtLoading(true);
    setCardLoading(true);
    setHistoryLoading(true);

    getNewt(newtId).then((res) => {
      setNewt(res);
      setNewtLoading(false);
    });

    getNewtCards(newtId).then((res) => {
      setCards(res);
      setCardLoading(false);
    });

    getNewtHistory(newtId).then((res) => {
      setHistory(res);
      setHistoryLoading(false);
    });
  }, [newtId]);

  const handleEditStart = (index: number) => {
    setEditData(prev => ({
      ...prev,
      [index]: cards[index]?.data ? { ...cards[index].data } : {},
    }));
    setEditingIndex(index);
  };

  const handleSave = async (index: number) => {
    const card = cards[index];
    if (!card) return;

    setSaving(true);

    try {
      await updateNewtCardApi({
        newtId,
        cardType: card.cardType,
        data: editData[index],
      });

      const updated = [...cards];
      updated[index] = {
        ...card,
        data: editData[index],
      };

      setCards(updated);
      setEditingIndex(null);

      toast({ title: "Карточка обновлена" });
    } catch {
      toast({ title: "Ошибка при сохранении", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  const renderField = (
    index: number,
    key: string,
    label: string,
    value: any,
    type = "text",
    isParentId = false
  ) => {
    const isEditing = editingIndex === index;

    if (!isEditing) {
      const displayValue = value || "—";
      const canLinkParent =
        isParentId &&
        typeof value === "string" &&
        value.trim() !== "" &&
        value !== "данные отсутствуют";

      return (
        <div className="py-3 border-b last:border-0 border-border/50">
          <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">
            {label}
          </dt>
          <dd className="text-sm font-medium">
            {canLinkParent ? (
              <Link href={`/newts/${value}`}>
                <span className="text-primary underline cursor-pointer">{displayValue}</span>
              </Link>
            ) : (
              displayValue
            )}
          </dd>
        </div>
      );
    }

    return (
      <div className="py-3 border-b last:border-0 border-border/50 space-y-2">
        <Label className="text-xs uppercase tracking-wider text-muted-foreground">
          {label}
        </Label>
        <Input
          type={type}
          value={editData[index]?.[key] || ""}
          onChange={(e) =>
            setEditData(prev => ({
              ...prev,
              [index]: {
                ...prev[index],
                [key]: e.target.value,
              },
            }))
          }
          className="h-8"
        />
      </div>
    );
  };

  const renderCard = (card: NewtCard, index: number) => {
    const data = editingIndex === index ? editData[index] : card.data;

    const commonFields = (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8">
        <div className="space-y-0">
          {renderField(index, "species", "Вид", data?.species)}
          {renderField(index, "dateFilled", "Дата заполнения", data?.dateFilled, "date")}
          {renderField(index, "conditionalBirthDate", "Условный год рождения", data?.conditionalBirthDate)}
          {renderField(index, "exactBirthDate", "Точная дата рождения", data?.exactBirthDate, "date")}
          {renderField(index, "bodyLength", "Длина тела (L), мм", data?.bodyLength, "number")}
          {renderField(index, "tailLength", "Длина хвоста (Lcd), мм", data?.tailLength, "number")}
          {renderField(index, "weight", "Вес (г)", data?.weight, "number")}
          {renderField(index, "sex", "Пол", data?.sex)}
        </div>

        <div className="space-y-0">
          {renderField(index, "photoNumber", "Номер фото", data?.photoNumber)}
          {renderField(index, "regionOfOrigin", "Регион происхождения", data?.regionOfOrigin)}
          {renderField(index, "releaseDate", "Дата выпуска", data?.releaseDate, "date")}
          {renderField(index, "fatherId", "ID отца", data?.fatherId, "text", true)}
          {renderField(index, "motherId", "ID матери", data?.motherId, "text", true)}
          {renderField(index, "measurementDevice", "Измерительный прибор", data?.measurementDevice)}
          {renderField(index, "scaleBrand", "Марка весов", data?.scaleBrand)}
          {renderField(index, "notes", "Примечания", data?.notes)}
        </div>
      </div>
    );

    return (
      <Card key={index}>
        <CardHeader className="bg-muted/30 border-b flex flex-row justify-between items-center">
          <CardTitle className="text-lg">
            Карточка ({card.cardType})
          </CardTitle>

          <div className="flex gap-2">
            {editingIndex === index ? (
              <>
                <Button onClick={() => handleSave(index)} disabled={saving}>
                  <Save className="w-4 h-4 mr-2" /> Сохранить
                </Button>
                <Button variant="outline" onClick={() => setEditingIndex(null)}>
                  Отмена
                </Button>
              </>
            ) : (
              <Button variant="outline" onClick={() => handleEditStart(index)}>
                Редактировать
              </Button>
            )}
            <Button
              variant="destructive"
              disabled={deleting || saving}
              onClick={async () => {
                const cardNumber = index + 1;
                if (!window.confirm(`Вы точно хотите удалить карточку №${cardNumber} (${card.cardType})? Это действие необратимо.`)) return;
                setDeleting(true);
                try {
                  const config = await window.api.getConfig();
                  const res = await fetch(`${config.apiBaseUrl}/newts/${newtId}/card?cardType=${encodeURIComponent(card.cardType)}`, {
                    method: "DELETE",
                  });

                  if (!res.ok) {
                    throw new Error(`Delete card failed: ${res.status}`);
                  }

                  setCards((prev) => prev.filter((_, i) => i !== index));
                  toast({ title: "Карточка удалена" });
                } catch {
                  toast({
                    title: "Не удалось удалить карточку. Возможно, endpoint не реализован на backend.",
                    variant: "destructive",
                  });
                } finally {
                  setDeleting(false);
                }
              }}
            >
              <Trash2 className="w-4 h-4 mr-2" /> Удалить карточку
            </Button>
          </div>
        </CardHeader>

        <CardContent className="p-6 space-y-6">
          {commonFields}

          <div className="pt-4 border-t">
            <PhotoGallery
              newtId={newtId}
              photos={card.photos}
              editable={editingIndex === index}
            />
          </div>
        </CardContent>
      </Card>
      
    );
  };

  if (newtLoading) {
    return (
      <div className="p-8 max-w-5xl mx-auto space-y-8">
        <Skeleton className="h-10 w-48 mb-6" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (!newt) {
    return <div className="p-8 text-center text-muted-foreground">Особь не найдена</div>;
  }

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-8 animate-in fade-in duration-500">

      <Link href={newt.projectId ? `/projects/${newt.projectId}` : "/projects"}>
        <Button variant="ghost" size="sm">
          <ArrowLeft className="w-4 h-4 mr-2" /> Назад
        </Button>
      </Link>

      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold font-mono">{newt.id}</h1>
          <div className="text-sm text-muted-foreground mt-1">
            {cards.map(c => c.cardType).join(", ")}
          </div>
        </div>
        <Button
          variant="destructive"
          disabled={deleting || saving}
          onClick={async () => {
            if (!window.confirm(`Вы точно хотите удалить особь ${newt.id}? Это действие необратимо.`)) return;

            setDeleting(true);
            try {
              const config = await window.api.getConfig();
              const res = await fetch(`${config.apiBaseUrl}/newts/${newtId}`, {
                method: "DELETE",
              });

              if (!res.ok) {
                throw new Error(`Delete newt failed: ${res.status}`);
              }

              toast({ title: "Особь удалена" });
              window.history.back();
            } catch {
              toast({
                title: "Не удалось удалить особь. Возможно, endpoint не реализован на backend.",
                variant: "destructive",
              });
            } finally {
              setDeleting(false);
            }
          }}
        >
          <Trash2 className="w-4 h-4 mr-2" /> Удалить особь
        </Button>
      </div>

      <div className="space-y-6">
        {cardLoading ? (
          <Skeleton className="h-64 w-full" />
        ) : (
          cards.map((card, index) => renderCard(card, index))
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>История</CardTitle>
        </CardHeader>
        <CardContent>
          {historyLoading ? (
            <Skeleton className="h-24 w-full" />
          ) : history.length === 0 ? (
            <div className="text-sm text-muted-foreground">
              История пуста
            </div>
          ) : (
            history.map((record) => (
              <div key={record.id} className="py-2 border-b text-sm">
                {record.field}: {record.oldValue} → {record.newValue}
              </div>
            ))
          )}
        </CardContent>
      </Card>

    </div>
  );
}