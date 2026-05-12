import { useEffect, useState } from "react";
import { Link } from "wouter";
import { ArrowLeft, Save } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";

export function Settings() {
  const { toast } = useToast();
  const [apiBaseUrl, setApiBaseUrl] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    window.api
      .getConfig()
      .then((c) => {
        setApiBaseUrl(c.apiBaseUrl ?? "");
      })
      .catch(() => {
        toast({ title: "Не удалось загрузить конфигурацию", variant: "destructive" });
      })
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    const trimmed = apiBaseUrl.trim();
    if (!trimmed) {
      toast({ title: "Укажите URL API", variant: "destructive" });
      return;
    }
    try {
      new URL(trimmed);
    } catch {
      toast({ title: "Некорректный URL", variant: "destructive" });
      return;
    }

    setSaving(true);
    try {
      const next = await window.api.setConfig({ apiBaseUrl: trimmed });
      setApiBaseUrl(next.apiBaseUrl);
      toast({ title: "Настройки сохранены" });
    } catch {
      toast({ title: "Не удалось сохранить", variant: "destructive" });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="p-8 max-w-2xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div>
        <Link href="/">
          <Button variant="ghost" size="sm" className="-ml-3 mb-2 text-muted-foreground">
            <ArrowLeft className="w-4 h-4 mr-2" /> Назад
          </Button>
        </Link>
        <h1 className="text-3xl font-bold tracking-tight">Настройки</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Подключение к API</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {loading ? (
            <p className="text-sm text-muted-foreground">Загрузка…</p>
          ) : (
            <>
              <div className="space-y-2">
                <Label htmlFor="apiBaseUrl">apiBaseUrl</Label>
                <Input
                  id="apiBaseUrl"
                  value={apiBaseUrl}
                  onChange={(e) => setApiBaseUrl(e.target.value)}
                  placeholder="https://example.com/api/v1"
                  className="font-mono text-sm"
                />
              </div>
              <Button onClick={handleSave} disabled={saving} className="gap-2">
                <Save className="w-4 h-4" />
                {saving ? "Сохранение…" : "Сохранить"}
              </Button>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
