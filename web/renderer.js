function navigateTo(page) {
    window.location.href = page;
}

const CARD_DRAFT_KEY = "triton_card_template_draft_v1";

const CARD_TEMPLATE_DEFS = {
    IK1: {
        title: "ИК-1",
        rows: [
            [{ key: "specimen_id", label: "1. ID-номер особи", type: "text" }],
            [{ key: "card_fill_date", label: "2. Дата заполнения карточки", type: "date" }],
            [
                { key: "len_l_mm", label: "3. Длина тела (L), мм", type: "text" },
                { key: "len_lcd_mm", label: "3. Длина хвоста (Lcd), мм", type: "text" }
            ],
            [{ key: "mass_g", label: "4. Масса, г", type: "text" }],
            [
                {
                    key: "sex",
                    label: "5. Пол",
                    type: "select",
                    options: [
                        { value: "", label: "— не указано —" },
                        { value: "male", label: "Самец" },
                        { value: "female", label: "Самка" },
                        { value: "unknown", label: "Неизвестно" }
                    ]
                }
            ],
            [{ key: "birth_exact", label: "6. Год рождения особи точный (дд.мм.гггг)", type: "text", placeholder: "дд.мм.гггг" }],
            [{ key: "birth_conditional", label: "7. Условный год рождения особи (дд.мм.гггг)", type: "text", placeholder: "дд.мм.гггг" }],
            [{ key: "pattern_photo_no", label: "8. Номер фото индивидуального рисунка", type: "text" }],
            [{ key: "origin_region", label: "9. Регион происхождения особи", type: "text" }],
            [{ key: "length_device_brand", label: "10. Марка устройства для измерения длины", type: "text" }],
            [{ key: "scale_brand", label: "11. Марка весов для взвешивания особи", type: "text" }],
            [{ key: "notes", label: "12. Примечания", type: "textarea", rows: 4 }]
        ]
    },
    IK2: {
        title: "ИК-2",
        rows: [
            [{ key: "specimen_id", label: "1. ID-номер особи", type: "text" }],
            [{ key: "card_fill_date", label: "2. Дата заполнения карточки", type: "date" }],
            [{ key: "release_date", label: "3. Дата выпуска в водоём", type: "date" }],
            [{ key: "father_id", label: "4. ID самца (родитель)", type: "parentId" }],
            [{ key: "mother_id", label: "5. ID самки (родитель)", type: "parentId" }],
            [{ key: "total_len_cm", label: "6. Общая длина (L + Lcd), см", type: "text" }],
            [{ key: "mass_g", label: "7. Масса, г", type: "text" }],
            [{ key: "pond_name", label: "8. Название водоёма", type: "text" }],
            [{ key: "notes", label: "9. Примечания", type: "textarea", rows: 4 }]
        ]
    },
    KV1: {
        title: "КВ-1",
        rows: [
            [{ key: "specimen_id", label: "1. ID-номер особи", type: "text" }],
            [{ key: "card_fill_date", label: "2. Дата заполнения карточки", type: "date" }],
            [{ key: "shoot_or_obs_date", label: "3. Дата съёмки / наблюдения (опционально)", type: "date" }],
            [{ key: "len_l_mm", label: "4. Длина тела (L), мм — от кончика морды до переднего края клоаки", type: "text" }],
            [{ key: "len_lcd_mm", label: "5. Длина хвоста (Lcd), мм — от клоаки до конца хвоста", type: "text" }],
            [{ key: "mass_g", label: "6. Масса, г", type: "text" }],
            [
                {
                    key: "sex",
                    label: "7. Пол",
                    type: "select",
                    options: [
                        { value: "", label: "— не указано —" },
                        { value: "male", label: "Самец" },
                        { value: "female", label: "Самка" },
                        { value: "unknown", label: "Неизвестно" }
                    ]
                }
            ],
            [{ key: "belly_photo_no", label: "8. Номер фото индивидуального рисунка брюшной стороны", type: "text" }],
            [
                {
                    key: "status",
                    label: "9. Статус",
                    type: "select",
                    options: [
                        { value: "", label: "— не указано —" },
                        { value: "alive", label: "Жив" },
                        { value: "dead", label: "Мёртв" }
                    ]
                }
            ],
            [{ key: "pond_number", label: "10. Номер водоёма, в котором особь обнаружена", type: "text" }],
            [{ key: "length_device_brand", label: "11. Марка устройства для измерения длины", type: "text" }],
            [{ key: "scale_brand", label: "12. Марка весов для взвешивания особи", type: "text" }],
            [{ key: "notes", label: "13. Примечания", type: "textarea", rows: 4 }]
        ]
    },
    KV2: {
        title: "КВ-2",
        rows: [
            [{ key: "specimen_id", label: "1. ID-номер особи", type: "text" }],
            [{ key: "meet_date", label: "2. Дата встречи (дд.мм.гггг)", type: "date" }],
            [{ key: "meet_time", label: "3. Время встречи (ч. мин.)", type: "text", placeholder: "например: 14 30" }],
            [{ key: "total_len_cm", label: "4. Общая длина (L + Lcd), см", type: "text" }],
            [
                {
                    key: "status",
                    label: "5. Статус",
                    type: "select",
                    options: [
                        { value: "", label: "— не указано —" },
                        { value: "alive", label: "Жив" },
                        { value: "dead", label: "Мёртв" }
                    ]
                }
            ],
            [{ key: "pond_name", label: "6. Название водоёма", type: "text" }],
            [{ key: "notes", label: "7. Примечания (подробная информация)", type: "textarea", rows: 5 }]
        ]
    }
};

const path = window.location.pathname;

if (path.includes("main_analysis.html")) {
    setupIdentificationPage();
}
if (path.includes("database.html")) {
    setupProjectsPage();
}
if (path.includes("add.species.html")) {
    setupCreateCardPage();
}

function setupIdentificationPage() {
    const modeInputs = document.querySelectorAll("input[name='searchMode']");
    const speciesWrap = document.getElementById("speciesWrap");
    const territoryWrap = document.getElementById("territoryWrap");
    const renameNeeded = document.getElementById("renameNeeded");
    const newFilename = document.getElementById("newFilename");
    const runBtn = document.getElementById("runIdentifyBtn");
    const pipelineLog = document.getElementById("pipelineLog");
    const foundResult = document.getElementById("foundResult");
    const notFoundResult = document.getElementById("notFoundResult");
    const simulateNotFound = document.getElementById("simulateNotFound");

    function updateModeFields() {
        const selected = document.querySelector("input[name='searchMode']:checked")?.value;
        speciesWrap.classList.toggle("hidden", selected !== "species");
        territoryWrap.classList.toggle("hidden", selected !== "territory");
    }

    modeInputs.forEach((input) => input.addEventListener("change", updateModeFields));
    updateModeFields();

    renameNeeded.addEventListener("change", () => {
        newFilename.classList.toggle("hidden", !renameNeeded.checked);
    });

    runBtn.addEventListener("click", () => {
        pipelineLog.innerHTML = "";
        foundResult.classList.add("hidden");
        notFoundResult.classList.add("hidden");

        const steps = [
            "Фото загружено и проверено.",
            "YOLO: выделение контуров тритона.",
            "Формирование эмбеддинга признаков.",
            "Сравнение с записями БД."
        ];

        steps.forEach((step, index) => {
            setTimeout(() => {
                const item = document.createElement("li");
                item.textContent = step;
                pipelineLog.appendChild(item);
            }, index * 350);
        });

        setTimeout(() => {
            if (simulateNotFound.checked) {
                notFoundResult.classList.remove("hidden");
                showToast("Обработка завершена: совпадений нет.");
                return;
            }
            foundResult.classList.remove("hidden");
            showToast("Обработка завершена: найдено совпадение.");
        }, steps.length * 350 + 200);
    });
}

function setupProjectsPage() {
    const projectsList = document.getElementById("projectsList");
    const projectCardPanel = document.getElementById("projectCardPanel");
    const historyBox = document.getElementById("historyBox");
    const idSearch = document.getElementById("projectIdSearch");
    const speciesFilter = document.getElementById("speciesFilter");
    const territoryFilter = document.getElementById("territoryFilter");
    const runSearchBtn = document.getElementById("runProjectSearchBtn");
    const clearBtn = document.getElementById("clearProjectSearchBtn");
    const saveBtn = document.getElementById("saveCardBtn");
    const showHistoryBtn = document.getElementById("showHistoryBtn");
    const deleteBtn = document.getElementById("deleteCardBtn");

    const cardId = document.getElementById("cardId");
    const cardSpecies = document.getElementById("cardSpecies");
    const cardTerritory = document.getElementById("cardTerritory");
    const cardNotes = document.getElementById("cardNotes");

    let selectedId = null;
    let projectData = [
        { id: "TR-1001", species: "Triturus cristatus", territory: "Калуга", notes: "Стабильная популяция", history: [] },
        { id: "TR-1002", species: "Lissotriton vulgaris", territory: "Татарстан", notes: "Наблюдение у берега", history: ["22.03: создана карта"] },
        { id: "TR-1003", species: "Triturus dobrogicus", territory: "Воронеж", notes: "Повторный осмотр в апреле", history: [] }
    ];

    function renderList(items) {
        projectsList.innerHTML = "";
        if (!items.length) {
            projectsList.innerHTML = "<p class='subtitle'>Ничего не найдено по текущим условиям.</p>";
            return;
        }
        items.forEach((item) => {
            const el = document.createElement("button");
            el.className = "project-item";
            el.innerHTML = `<strong>${item.id}</strong><br>${item.species} · ${item.territory}`;
            el.addEventListener("click", () => openCard(item.id));
            if (item.id === selectedId) {
                el.classList.add("active");
            }
            projectsList.appendChild(el);
        });
    }

    function applyFilters() {
        const idValue = idSearch.value.trim().toLowerCase();
        const speciesValue = speciesFilter.value.trim().toLowerCase();
        const territoryValue = territoryFilter.value.trim().toLowerCase();
        const filtered = projectData.filter((item) => {
            const idOk = !idValue || item.id.toLowerCase().includes(idValue);
            const speciesOk = !speciesValue || item.species.toLowerCase().includes(speciesValue);
            const territoryOk = !territoryValue || item.territory.toLowerCase().includes(territoryValue);
            return idOk && speciesOk && territoryOk;
        });
        renderList(filtered);
    }

    function openCard(id) {
        const item = projectData.find((project) => project.id === id);
        if (!item) return;
        selectedId = id;
        cardId.value = item.id;
        cardSpecies.value = item.species;
        cardTerritory.value = item.territory;
        cardNotes.value = item.notes;
        historyBox.classList.add("hidden");
        projectCardPanel.classList.remove("hidden");
        applyFilters();
    }

    runSearchBtn.addEventListener("click", applyFilters);
    clearBtn.addEventListener("click", () => {
        idSearch.value = "";
        speciesFilter.value = "";
        territoryFilter.value = "";
        applyFilters();
    });

    saveBtn.addEventListener("click", () => {
        const item = projectData.find((project) => project.id === selectedId);
        if (!item) return;
        item.species = cardSpecies.value.trim();
        item.territory = cardTerritory.value.trim();
        item.notes = cardNotes.value.trim();
        item.history.push(`${new Date().toLocaleDateString("ru-RU")}: обновлены данные карты`);
        applyFilters();
        showToast("Изменения карты сохранены (шаблонно).");
    });

    showHistoryBtn.addEventListener("click", () => {
        const item = projectData.find((project) => project.id === selectedId);
        if (!item) return;
        if (!item.history.length) {
            historyBox.innerHTML = "История изменений отсутствует.";
        } else {
            historyBox.innerHTML = item.history.map((row) => `<div>${row}</div>`).join("");
        }
        historyBox.classList.remove("hidden");
    });

    deleteBtn.addEventListener("click", () => {
        if (!selectedId) return;
        projectData = projectData.filter((item) => item.id !== selectedId);
        selectedId = null;
        projectCardPanel.classList.add("hidden");
        applyFilters();
        showToast("Карта удалена из шаблонного списка.");
    });

    applyFilters();

    const urlParams = new URLSearchParams(window.location.search);
    const openId = (urlParams.get("id") || urlParams.get("highlight") || "").trim();
    if (openId) {
        idSearch.value = openId;
        applyFilters();
        const match = projectData.find((project) => project.id.toLowerCase() === openId.toLowerCase());
        if (match) {
            openCard(match.id);
        }
    }
}

function renderCardTemplateFields(templateKey, root, values = {}) {
    const def = CARD_TEMPLATE_DEFS[templateKey];
    if (!def || !root) return;
    root.innerHTML = "";
    const titleEl = document.getElementById("templateSectionTitle");
    if (titleEl) {
        titleEl.textContent = `Поля шаблона: ${def.title}`;
    }

    def.rows.forEach((row) => {
        if (row.length === 1) {
            root.appendChild(buildFieldControl(row[0], values));
        } else {
            const wrap = document.createElement("div");
            wrap.className = "field-row-split";
            row.forEach((field) => wrap.appendChild(buildFieldControl(field, values)));
            root.appendChild(wrap);
        }
    });

    bindParentIdLinks(root);
}

function buildFieldControl(field, values) {
    const wrap = document.createElement("div");
    wrap.className = "field-row full";

    if (field.type === "parentId") {
        const parentWrap = document.createElement("div");
        parentWrap.className = "parent-link-row";

        const label = document.createElement("label");
        label.htmlFor = `field-${field.key}`;
        label.textContent = field.label;

        const input = document.createElement("input");
        input.type = "text";
        input.id = `field-${field.key}`;
        input.dataset.fieldKey = field.key;
        input.dataset.fieldType = "parentId";
        input.placeholder = "ID из базы, например TR-1002";
        input.value = values[field.key] ?? "";

        const link = document.createElement("a");
        link.className = "parent-db-link";
        link.textContent = "Открыть карту в проектах";
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.dataset.parentLink = field.key;

        parentWrap.appendChild(label);
        parentWrap.appendChild(input);
        parentWrap.appendChild(link);
        wrap.appendChild(parentWrap);
        updateParentDbLink(link, input.value);
        return wrap;
    }

    const label = document.createElement("label");
    label.htmlFor = `field-${field.key}`;
    label.textContent = field.label;

    let control;
    if (field.type === "textarea") {
        control = document.createElement("textarea");
        control.rows = field.rows || 3;
    } else if (field.type === "select") {
        control = document.createElement("select");
        field.options.forEach((opt) => {
            const option = document.createElement("option");
            option.value = opt.value;
            option.textContent = opt.label;
            control.appendChild(option);
        });
        control.value = values[field.key] ?? "";
    } else {
        control = document.createElement("input");
        control.type = field.type === "date" ? "date" : "text";
        if (field.placeholder) {
            control.placeholder = field.placeholder;
        }
        control.value = values[field.key] ?? "";
    }

    control.id = `field-${field.key}`;
    control.dataset.fieldKey = field.key;
    if (field.type !== "textarea") {
        control.dataset.fieldType = field.type;
    } else {
        control.dataset.fieldType = "textarea";
    }

    wrap.appendChild(label);
    wrap.appendChild(control);
    return wrap;
}

function updateParentDbLink(link, rawId) {
    const trimmed = rawId.trim();
    if (!trimmed) {
        link.href = "database.html";
        link.setAttribute("aria-disabled", "true");
        link.title = "Введите ID, чтобы перейти к карте в базе";
        return;
    }
    link.removeAttribute("aria-disabled");
    link.title = "Открыть карту в разделе проектов";
    const encoded = encodeURIComponent(trimmed);
    link.href = `database.html?id=${encoded}`;
}

function bindParentIdLinks(root) {
    root.querySelectorAll(`input[data-field-type="parentId"]`).forEach((input) => {
        const link = root.querySelector(`a[data-parent-link="${input.dataset.fieldKey}"]`);
        if (!link) return;
        updateParentDbLink(link, input.value);
        input.addEventListener("input", () => updateParentDbLink(link, input.value));
    });
}

function collectCardFieldValues(root) {
    const data = {};
    root.querySelectorAll("[data-field-key]").forEach((el) => {
        const key = el.dataset.fieldKey;
        if (!key) return;
        data[key] = el.value;
    });
    return data;
}

function formatCardPreviewValue(templateKey, key, raw) {
    const trimmed = (raw ?? "").trim();
    if (!trimmed) return "—";
    if (key === "sex") {
        if (trimmed === "male") return "Самец";
        if (trimmed === "female") return "Самка";
        if (trimmed === "unknown") return "Неизвестно";
    }
    if (key === "status") {
        if (trimmed === "alive") return "Жив";
        if (trimmed === "dead") return "Мёртв";
    }
    return trimmed;
}

function buildCardPreviewHtml(templateKey, fields, photoFileName) {
    const def = CARD_TEMPLATE_DEFS[templateKey];
    const labelByKey = {};
    def.rows.forEach((row) => {
        row.forEach((field) => {
            labelByKey[field.key] = field.label;
        });
    });

    const items = Object.keys(fields)
        .map((key) => {
            const val = formatCardPreviewValue(templateKey, key, fields[key]);
            if (val === "—") return null;
            const label = labelByKey[key] || key;
            return `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(val)}</dd>`;
        })
        .filter(Boolean);

    const photoHtml =
        photoFileName && photoFileName !== "не выбрано"
            ? `<dt>Файл фото</dt><dd>${escapeHtml(photoFileName)}</dd>`
            : `<dt>Файл фото</dt><dd>—</dd>`;

    return `
        <p><strong>Шаблон:</strong> ${escapeHtml(def.title)}</p>
        <dl>
            ${photoHtml}
            ${items.join("")}
        </dl>
        <p class="hint-text" style="margin-top:10px">Пустые поля в предпросмотре скрыты; их можно заполнить позже.</p>
    `;
}

function escapeHtml(str) {
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function saveCardDraft(templateKey, fields, photoName) {
    const payload = {
        template: templateKey,
        fields,
        photoName: photoName || "",
        savedAt: new Date().toISOString()
    };
    localStorage.setItem(CARD_DRAFT_KEY, JSON.stringify(payload));
}

function loadCardDraft() {
    const raw = localStorage.getItem(CARD_DRAFT_KEY);
    if (!raw) return null;
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

function setupCreateCardPage() {
    const form = document.getElementById("createCardForm");
    const templateSelect = document.getElementById("cardTemplateSelect");
    const templateRoot = document.getElementById("templateFieldsRoot");
    const photoInput = document.getElementById("cardPhotoFile");
    const resetBtn = document.getElementById("resetCardFormBtn");
    const saveDraftBtn = document.getElementById("saveDraftBtn");
    const loadDraftBtn = document.getElementById("loadDraftBtn");
    const previewPanel = document.getElementById("cardPreviewPanel");
    const preview = document.getElementById("cardPreview");

    function currentTemplate() {
        return templateSelect.value;
    }

    function renderTemplate(presetValues = {}) {
        renderCardTemplateFields(currentTemplate(), templateRoot, presetValues);
    }

    templateSelect.addEventListener("change", () => {
        renderTemplate({});
        previewPanel.classList.add("hidden");
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const fields = collectCardFieldValues(templateRoot);
        const photoName = photoInput.files[0]?.name || "не выбрано";
        preview.innerHTML = buildCardPreviewHtml(currentTemplate(), fields, photoName);
        previewPanel.classList.remove("hidden");
        saveCardDraft(currentTemplate(), fields, photoName);
        showToast("Данные сохранены в черновик; предпросмотр обновлён.");
    });

    saveDraftBtn.addEventListener("click", () => {
        const fields = collectCardFieldValues(templateRoot);
        const photoName = photoInput.files[0]?.name || "";
        saveCardDraft(currentTemplate(), fields, photoName);
        showToast("Черновик сохранён локально.");
    });

    loadDraftBtn.addEventListener("click", () => {
        const draft = loadCardDraft();
        if (!draft || !draft.template) {
            showToast("Черновик не найден.");
            return;
        }
        templateSelect.value = draft.template;
        renderTemplate(draft.fields || {});
        if (draft.photoName) {
            showToast(`Черновик загружен (файл фото: ${draft.photoName || "не выбран"} — выберите снова при необходимости).`);
        } else {
            showToast("Черновик загружен.");
        }
        previewPanel.classList.add("hidden");
    });

    resetBtn.addEventListener("click", () => {
        form.reset();
        previewPanel.classList.add("hidden");
        renderTemplate({});
        showToast("Форма очищена.");
    });

    const initialDraft = loadCardDraft();
    if (initialDraft && initialDraft.template && CARD_TEMPLATE_DEFS[initialDraft.template]) {
        templateSelect.value = initialDraft.template;
        renderTemplate(initialDraft.fields || {});
    } else {
        renderTemplate({});
    }
}

function showToast(message) {
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 2100);
}