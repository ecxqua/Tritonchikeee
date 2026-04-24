import { useState, useRef, useEffect, useCallback } from "react";
import { Camera, X, ZoomIn, Upload } from "lucide-react";
import { cn } from "@/lib/utils";

const STORAGE_KEY_PREFIX = "newt-photos-";

/* -----------------------------
   OPTIONAL localStorage helpers
------------------------------ */
export function loadPhotos(newtId: string): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_PREFIX + newtId);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function savePhotos(newtId: string, photos: string[]) {
  try {
    localStorage.setItem(STORAGE_KEY_PREFIX + newtId, JSON.stringify(photos));
  } catch {}
}

/* =========================================================
   PHOTO GALLERY (READ ONLY / EDIT MODE)
   NOW SUPPORTS API PHOTOS DIRECTLY
========================================================= */

interface PhotoGalleryProps {
  newtId: string;
  photos?: string[];        // ✅ IMPORTANT: API input
  editable?: boolean;
}

export function PhotoGallery({
  newtId,
  photos: apiPhotos,
  editable = false,
}: PhotoGalleryProps) {

  // IMPORTANT FIX:
  // API overrides localStorage completely
  const [photos, setPhotos] = useState<string[]>(() => apiPhotos ?? []);

  const [lightboxIdx, setLightboxIdx] = useState<number | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  // Sync when API updates
  useEffect(() => {
    if (apiPhotos) {
      setPhotos(apiPhotos);
    }
  }, [apiPhotos]);

  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsDataURL(file);
    });

  const addFiles = useCallback(async (files: FileList) => {
    const newPhotos = await Promise.all(
      Array.from(files).map(fileToBase64)
    );

    setPhotos((prev) => {
      const updated = [...prev, ...newPhotos];
      savePhotos(newtId, updated);
      return updated;
    });
  }, [newtId]);

  const removePhoto = (idx: number) => {
    setPhotos((prev) => {
      const updated = prev.filter((_, i) => i !== idx);
      savePhotos(newtId, updated);
      return updated;
    });
    setLightboxIdx(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  };

  return (
    <div className="space-y-3">

      {/* EMPTY STATE */}
      {photos.length === 0 && !editable && (
        <div className="flex flex-col items-center justify-center border-2 border-dashed border-muted-foreground/25 rounded-lg aspect-[3/4] bg-muted/10 text-muted-foreground p-6 text-center">
          <Camera className="w-8 h-8 mb-2 opacity-50" />
          <p className="text-sm">Фото отсутствует</p>
        </div>
      )}

      {/* GRID */}
      {photos.length > 0 && (
        <div className="grid grid-cols-2 gap-2">
          {photos.map((src, idx) => (
            <div
              key={idx}
              className="relative group rounded-lg overflow-hidden border aspect-square bg-black/5 cursor-pointer"
              onClick={() => setLightboxIdx(idx)}
            >
              <img
                src={src}
                alt={`Фото ${idx + 1}`}
                className="w-full h-full object-cover"
              />

              <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <ZoomIn className="w-5 h-5 text-white" />
              </div>

              {editable && (
                <button
                  className="absolute top-1 right-1 w-5 h-5 rounded-full bg-destructive text-white opacity-0 group-hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation();
                    removePhoto(idx);
                  }}
                >
                  <X className="w-3 h-3" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* UPLOAD */}
      {editable && (
        <>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />

          <div
            className={cn(
              "border-2 border-dashed rounded-lg p-4 text-center cursor-pointer",
              isDragging ? "border-primary bg-primary/5" : "border-muted-foreground/25"
            )}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
          >
            <Upload className="w-5 h-5 mx-auto mb-1 opacity-50" />
            Добавить фото
          </div>
        </>
      )}

      {/* LIGHTBOX */}
      {lightboxIdx !== null && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
          onClick={() => setLightboxIdx(null)}
        >
          <img
            src={photos[lightboxIdx]}
            className="max-w-[90vw] max-h-[90vh] object-contain"
          />
        </div>
      )}
    </div>
  );
}

/* =========================================================
   UPLOAD ONLY COMPONENT
========================================================= */

export function PhotoUploadZone({
  onChange,
}: {
  onChange: (photos: string[]) => void;
}) {
  const [photos, setPhotos] = useState<string[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const fileToBase64 = (file: File): Promise<string> =>
    new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsDataURL(file);
    });

  const addFiles = useCallback(async (files: FileList) => {
    const newPhotos = await Promise.all(
      Array.from(files).map(fileToBase64)
    );

    setPhotos((prev) => {
      const updated = [...prev, ...newPhotos];
      onChange(updated);
      return updated;
    });
  }, [onChange]);

  const removePhoto = (idx: number) => {
    setPhotos((prev) => {
      const updated = prev.filter((_, i) => i !== idx);
      onChange(updated);
      return updated;
    });
  };

  return (
    <div className="space-y-3">

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={(e) => e.target.files && addFiles(e.target.files)}
      />

      {photos.length > 0 && (
        <div className="grid grid-cols-3 gap-2">
          {photos.map((src, idx) => (
            <div key={idx} className="relative aspect-square">
              <img src={src} className="w-full h-full object-cover rounded" />
              <button
                className="absolute top-1 right-1 bg-red-500 text-white rounded w-5 h-5"
                onClick={() => removePhoto(idx)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <div
        className={cn(
          "border-2 border-dashed rounded-lg p-6 text-center cursor-pointer",
          isDragging ? "border-primary bg-primary/5" : ""
        )}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        }}
        onClick={() => fileRef.current?.click()}
      >
        <Upload className="w-5 h-5 mx-auto mb-2 opacity-50" />
        Добавить фотографии
      </div>
    </div>
  );
}