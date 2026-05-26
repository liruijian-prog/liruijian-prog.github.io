const STORAGE_KEYS = {
  settings: "talent-funnel.reader.settings",
  progress: "talent-funnel.reader.progress",
  bookmarks: "talent-funnel.reader.bookmarks",
  annotations: "talent-funnel.reader.annotations",
};

const DEFAULT_SETTINGS = {
  theme: "paper",
  fontSize: 18,
  lineHeight: 1.95,
  contentWidth: 760,
};

const PANEL_IDS = [
  "notesOverlay",
  "glossaryOverlay",
  "referencesOverlay",
  "termOverlay",
  "noteEditorOverlay",
];

const GLOSSARY_EXCLUDE_TAGS = new Set([
  "A",
  "BUTTON",
  "CODE",
  "PRE",
  "SCRIPT",
  "STYLE",
  "TEXTAREA",
  "MATH",
  "MROW",
  "MSUP",
  "MSUB",
  "MSUBSUP",
  "MFRAC",
  "MI",
  "MN",
  "MO",
  "MTEXT",
  "SEMANTICS",
  "ANNOTATION",
]);

const state = {
  manifest: null,
  chapters: new Map(),
  currentChapter: null,
  currentChapterId: null,
  searchIndex: [],
  glossary: { entries: [] },
  glossaryById: new Map(),
  glossaryByAlias: new Map(),
  glossaryRegex: null,
  currentChapterTerms: [],
  settings: loadJson(STORAGE_KEYS.settings, DEFAULT_SETTINGS),
  progress: loadJson(STORAGE_KEYS.progress, { currentChapterId: null, chapterScroll: {} }),
  bookmarks: new Set(loadJson(STORAGE_KEYS.bookmarks, [])),
  annotations: loadJson(STORAGE_KEYS.annotations, {}),
  contentVersion: null,
  noteMode: false,
  currentNoteTarget: null,
};

const dom = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheDom();
  bindEvents();
  applySettings();
  registerServiceWorker();
  init().catch((error) => {
    console.error(error);
    showToast("阅读器加载失败，请检查内容构建是否完成。", 5000);
  });
});

async function init() {
  await loadManifest();
  await Promise.all([loadSearchIndex(), loadGlossary()]);
  renderAppChrome();
  renderToc();

  const initialChapterId =
    chapterIdFromHash() ||
    state.progress.currentChapterId ||
    state.manifest.chapters[0]?.id;

  if (initialChapterId) {
    await openChapter(initialChapterId, { restoreScroll: true, updateHash: false });
  }

  updateOverallProgress();
  await checkForUpdates({ silent: true });
}

function cacheDom() {
  const ids = [
    "appBackdrop",
    "tocDrawer",
    "settingsDrawer",
    "drawerToc",
    "closeTocButton",
    "closeSettingsButton",
    "searchOverlay",
    "closeSearchButton",
    "searchInput",
    "searchResults",
    "searchMeta",
    "lightbox",
    "lightboxImage",
    "lightboxCaption",
    "closeLightboxButton",
    "notesOverlay",
    "closeNotesButton",
    "chapterNotesMeta",
    "chapterNotesList",
    "glossaryOverlay",
    "closeGlossaryButton",
    "glossarySearchInput",
    "glossarySummary",
    "glossaryList",
    "referencesOverlay",
    "closeReferencesButton",
    "referencesMeta",
    "referencesList",
    "termOverlay",
    "closeTermButton",
    "termChapter",
    "termTitle",
    "termShort",
    "termEnglish",
    "termDefinition",
    "noteEditorOverlay",
    "closeNoteEditorButton",
    "noteEditorTitle",
    "noteContextPreview",
    "noteTextarea",
    "deleteNoteButton",
    "saveNoteButton",
    "openTocButton",
    "openSearchButton",
    "openSettingsButton",
    "checkUpdatesButton",
    "continueReadingButton",
    "bookmarkButton",
    "topbarTitle",
    "sectionEyebrow",
    "coverImage",
    "bookTitle",
    "bookSubtitle",
    "bookDescription",
    "bookVersion",
    "bookStats",
    "railToc",
    "overallProgressLabel",
    "overallProgressBar",
    "sectionPill",
    "chapterOrder",
    "chapterTitle",
    "chapterSubtitle",
    "chapterWordCount",
    "chapterReadingMinutes",
    "chapterUpdatedAt",
    "noteModeButton",
    "chapterNotesButton",
    "chapterNotesCount",
    "glossaryButton",
    "chapterTermsCount",
    "referenceCardsButton",
    "referenceCardsCount",
    "dockNoteModeButton",
    "dockChapterNotesButton",
    "dockGlossaryButton",
    "dockReferenceCardsButton",
    "readingProgressBar",
    "chapterContent",
    "chapterShell",
    "prevChapterButton",
    "nextChapterButton",
    "fontSizeRange",
    "lineHeightRange",
    "contentWidthRange",
    "fontSizeValue",
    "lineHeightValue",
    "contentWidthValue",
    "toast",
  ];

  for (const id of ids) {
    dom[id] = document.getElementById(id);
  }
}

function bindEvents() {
  dom.openTocButton.addEventListener("click", () => openDrawer("toc"));
  dom.openSettingsButton.addEventListener("click", () => openDrawer("settings"));
  dom.closeTocButton.addEventListener("click", closeDrawers);
  dom.closeSettingsButton.addEventListener("click", closeDrawers);
  dom.openSearchButton.addEventListener("click", openSearch);
  dom.closeSearchButton.addEventListener("click", closeSearch);
  dom.searchOverlay.addEventListener("click", (event) => {
    if (event.target === dom.searchOverlay) {
      closeSearch();
    }
  });
  dom.checkUpdatesButton.addEventListener("click", () => checkForUpdates({ silent: false }));
  dom.appBackdrop.addEventListener("click", closeDrawers);
  dom.closeLightboxButton.addEventListener("click", closeLightbox);
  dom.lightbox.addEventListener("click", (event) => {
    if (event.target === dom.lightbox) {
      closeLightbox();
    }
  });

  for (const panelId of PANEL_IDS) {
    dom[panelId].addEventListener("click", (event) => {
      if (event.target === dom[panelId]) {
        closePanel(panelId);
      }
    });
  }

  dom.closeNotesButton.addEventListener("click", () => closePanel("notesOverlay"));
  dom.closeGlossaryButton.addEventListener("click", () => closePanel("glossaryOverlay"));
  dom.closeReferencesButton.addEventListener("click", () => closePanel("referencesOverlay"));
  dom.closeTermButton.addEventListener("click", () => closePanel("termOverlay"));
  dom.closeNoteEditorButton.addEventListener("click", () => closePanel("noteEditorOverlay"));
  dom.deleteNoteButton.addEventListener("click", deleteCurrentNote);
  dom.saveNoteButton.addEventListener("click", saveCurrentNote);

  dom.searchInput.addEventListener("input", handleSearchInput);
  dom.glossarySearchInput.addEventListener("input", renderGlossaryList);

  dom.continueReadingButton.addEventListener("click", () => {
    const targetId = state.currentChapterId || state.progress.currentChapterId || state.manifest.chapters[0]?.id;
    if (targetId) {
      openChapter(targetId, { restoreScroll: true, updateHash: true });
    }
  });
  dom.bookmarkButton.addEventListener("click", toggleBookmark);
  dom.prevChapterButton.addEventListener("click", () => openAdjacentChapter(-1));
  dom.nextChapterButton.addEventListener("click", () => openAdjacentChapter(1));
  window.addEventListener("scroll", throttle(updateReadingProgress, 120), { passive: true });
  window.addEventListener("hashchange", onHashChange);

  dom.fontSizeRange.addEventListener("input", onSettingsInput);
  dom.lineHeightRange.addEventListener("input", onSettingsInput);
  dom.contentWidthRange.addEventListener("input", onSettingsInput);

  document.querySelectorAll(".theme-chip").forEach((button) => {
    button.addEventListener("click", () => {
      state.settings.theme = button.dataset.theme;
      persistJson(STORAGE_KEYS.settings, state.settings);
      applySettings();
    });
  });

  bindReaderToolButton(dom.noteModeButton, toggleNoteMode);
  bindReaderToolButton(dom.dockNoteModeButton, toggleNoteMode);
  bindReaderToolButton(dom.chapterNotesButton, openNotesPanel);
  bindReaderToolButton(dom.dockChapterNotesButton, openNotesPanel);
  bindReaderToolButton(dom.glossaryButton, openGlossaryPanel);
  bindReaderToolButton(dom.dockGlossaryButton, openGlossaryPanel);
  bindReaderToolButton(dom.referenceCardsButton, openReferencesPanel);
  bindReaderToolButton(dom.dockReferenceCardsButton, openReferencesPanel);

  dom.chapterContent.addEventListener("click", (event) => {
    const termTrigger = event.target.closest(".term-trigger");
    if (termTrigger) {
      event.preventDefault();
      openTermCard(termTrigger.dataset.termId);
      return;
    }

    const image = event.target.closest("img.reader-image");
    if (image) {
      openLightbox(image);
      return;
    }

    if (!state.noteMode) {
      return;
    }

    const block = event.target.closest("[data-block-id]");
    if (!block) {
      return;
    }

    if (event.target.closest("a, button, input, textarea")) {
      return;
    }

    openNoteEditor(block.dataset.blockId);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") {
      return;
    }
    closeDrawers();
    closeSearch();
    closeLightbox();
    closeAllPanels();
  });
}

function bindReaderToolButton(element, handler) {
  if (!element) {
    return;
  }
  element.addEventListener("click", handler);
}

async function loadManifest({ bustCache = false } = {}) {
  const response = await fetch(withCacheBust("content/book.json", bustCache), {
    cache: bustCache ? "no-store" : "default",
  });
  state.manifest = await response.json();
  state.contentVersion = state.contentVersion || state.manifest.book.version;
}

async function loadSearchIndex() {
  const response = await fetch("content/search-index.json");
  const payload = await response.json();
  state.searchIndex = payload.items || [];
}

async function loadGlossary() {
  const path = state.manifest?.book?.glossaryPath || "content/glossary.json";
  const response = await fetch(path);
  const payload = await response.json();

  state.glossary = payload;
  state.glossaryById = new Map();
  state.glossaryByAlias = new Map();

  const aliases = [];
  for (const entry of payload.entries || []) {
    state.glossaryById.set(entry.id, entry);
    for (const alias of entry.aliases || []) {
      const cleanAlias = (alias || "").trim();
      if (!cleanAlias) {
        continue;
      }
      state.glossaryByAlias.set(cleanAlias, entry);
      aliases.push(cleanAlias);
    }
  }

  const uniqueAliases = [...new Set(aliases)].sort((left, right) => right.length - left.length);
  if (uniqueAliases.length) {
    const pattern = uniqueAliases.map((alias) => escapeRegExp(alias)).join("|");
    state.glossaryRegex = new RegExp(pattern, "g");
  }
}

function renderAppChrome() {
  const { book } = state.manifest;
  dom.coverImage.src = book.coverImagePath;
  dom.bookTitle.textContent = book.title;
  dom.bookSubtitle.textContent = book.subtitle;
  dom.bookDescription.textContent = book.description;
  dom.bookVersion.textContent = `版本 ${book.version}`;
  dom.bookStats.textContent = `${book.chapterCount} 篇 · ${formatWan(book.totalChineseChars)} 万字 · ${book.glossaryCount} 个术语`;
  dom.topbarTitle.textContent = book.shortTitle;
  dom.bookmarkButton.disabled = false;

  dom.fontSizeRange.value = String(state.settings.fontSize);
  dom.lineHeightRange.value = String(state.settings.lineHeight);
  dom.contentWidthRange.value = String(state.settings.contentWidth);
  applySettings();
}

function renderToc() {
  const html = state.manifest.sections
    .map((section) => {
      const items = state.manifest.chapters.filter((chapter) => chapter.sectionId === section.id);
      const itemHtml = items
        .map((chapter) => {
          const progress = chapterProgress(chapter.id);
          const isBookmarked = state.bookmarks.has(chapter.id);
          return `
            <button class="toc-item" data-chapter-id="${chapter.id}" type="button">
              <span class="toc-item__title">${escapeHtml(chapter.title)}</span>
              <span class="toc-item__meta">
                <span>${chapter.readingMinutes} 分钟</span>
                <span>${Math.round(progress * 100)}%${isBookmarked ? " · 已藏" : ""}</span>
              </span>
            </button>
          `;
        })
        .join("");
      return `
        <section class="toc-section">
          <h3 class="toc-section__title">${escapeHtml(section.title)}</h3>
          ${itemHtml}
        </section>
      `;
    })
    .join("");

  dom.railToc.innerHTML = html;
  dom.drawerToc.innerHTML = html;

  [dom.railToc, dom.drawerToc].forEach((container) => {
    container.querySelectorAll("[data-chapter-id]").forEach((button) => {
      button.addEventListener("click", async () => {
        const chapterId = button.dataset.chapterId;
        closeDrawers();
        await openChapter(chapterId, { restoreScroll: true, updateHash: true });
      });
    });
  });

  syncTocSelection();
}

async function openChapter(chapterId, { restoreScroll = false, updateHash = true } = {}) {
  const chapterMeta = state.manifest.chapters.find((item) => item.id === chapterId);
  if (!chapterMeta) {
    return;
  }

  let chapter = state.chapters.get(chapterId);
  if (!chapter) {
    const response = await fetch(chapterMeta.contentPath);
    chapter = await response.json();
    state.chapters.set(chapterId, chapter);
  }

  state.currentChapter = chapter;
  state.currentChapterId = chapterId;
  state.progress.currentChapterId = chapterId;
  persistJson(STORAGE_KEYS.progress, state.progress);

  dom.sectionEyebrow.textContent = chapter.sectionTitle;
  dom.sectionPill.textContent = chapter.sectionTitle;
  dom.chapterOrder.textContent = `第 ${chapter.order} / ${state.manifest.chapters.length} 篇`;
  dom.chapterTitle.textContent = chapter.title;

  const subtitleText = chapter.subtitle || chapter.summary || "";
  dom.chapterSubtitle.textContent = subtitleText;
  dom.chapterSubtitle.classList.toggle("hidden", !subtitleText);

  dom.chapterWordCount.textContent = `${chapter.wordCount.toLocaleString()} 字`;
  dom.chapterReadingMinutes.textContent = `${chapter.readingMinutes} 分钟阅读`;
  dom.chapterUpdatedAt.textContent = `更新于 ${chapter.updatedAt}`;
  dom.chapterContent.innerHTML = chapter.html;
  dom.bookmarkButton.textContent = state.bookmarks.has(chapterId) ? "取消收藏" : "收藏本章";
  dom.bookmarkButton.classList.toggle("is-active", state.bookmarks.has(chapterId));

  decorateRenderedContent();
  updateChapterTools();
  updateAdjacentButtons(chapterId);
  syncTocSelection();
  updateOverallProgress();
  renderGlossaryList();

  if (updateHash) {
    history.replaceState(null, "", `#${chapterId}`);
  }

  requestAnimationFrame(() => {
    const ratio = restoreScroll ? chapterProgress(chapterId) : 0;
    scrollToChapterRatio(ratio);
    updateReadingProgress();
  });
}

function decorateRenderedContent() {
  renderMath(dom.chapterContent);
  dom.chapterContent.querySelectorAll("a[target='_blank']").forEach((anchor) => {
    anchor.title = "在新窗口打开";
  });

  state.currentChapterTerms = decorateGlossaryTerms(dom.chapterContent);
  renderAnnotationState();
}

function renderMath(root) {
  if (!window.katex) {
    return;
  }

  root.querySelectorAll("[data-math]").forEach((node) => {
    const expression = node.dataset.math || "";
    const displayMode = node.classList.contains("math-block");
    try {
      window.katex.render(expression, node, {
        throwOnError: false,
        displayMode,
        output: "htmlAndMathml",
        strict: "warn",
      });
      node.removeAttribute("data-math");
      node.classList.add("math-rendered");
    } catch (error) {
      node.textContent = displayMode ? `$$${expression}$$` : `$${expression}$`;
      node.classList.add("math-render-failed");
    }
  });
}

function updateChapterTools() {
  const notesCount = getChapterNotes(state.currentChapterId).length;
  const referenceCount = state.currentChapter?.referenceCards?.length || 0;
  const termsCount = state.currentChapterTerms.length;

  dom.chapterNotesCount.textContent = String(notesCount);
  dom.chapterTermsCount.textContent = String(termsCount);
  dom.referenceCardsCount.textContent = String(referenceCount);

  dom.referenceCardsButton.classList.toggle("hidden", referenceCount === 0);
  dom.dockReferenceCardsButton.classList.toggle("hidden", referenceCount === 0);

  dom.dockChapterNotesButton.textContent = notesCount ? `章内 ${notesCount}` : "章内";
  dom.dockGlossaryButton.textContent = termsCount ? `术语 ${termsCount}` : "术语";
  dom.dockReferenceCardsButton.textContent = referenceCount ? `脚注 ${referenceCount}` : "脚注";
  updateNoteModeButtons();
}

function updateAdjacentButtons(chapterId) {
  const index = state.manifest.chapters.findIndex((item) => item.id === chapterId);
  const previous = state.manifest.chapters[index - 1];
  const next = state.manifest.chapters[index + 1];

  dom.prevChapterButton.disabled = !previous;
  dom.nextChapterButton.disabled = !next;
  dom.prevChapterButton.textContent = previous ? `上一章 · ${previous.title}` : "已到第一章";
  dom.nextChapterButton.textContent = next ? `下一章 · ${next.title}` : "已到最后一章";
}

function openAdjacentChapter(step) {
  const index = state.manifest.chapters.findIndex((item) => item.id === state.currentChapterId);
  const target = state.manifest.chapters[index + step];
  if (target) {
    openChapter(target.id, { restoreScroll: false, updateHash: true });
  }
}

function chapterProgress(chapterId) {
  return clamp(state.progress.chapterScroll?.[chapterId] ?? 0, 0, 1);
}

function updateReadingProgress() {
  if (!state.currentChapterId) {
    return;
  }

  const top = dom.chapterShell.getBoundingClientRect().top + window.scrollY;
  const range = Math.max(dom.chapterShell.offsetHeight - window.innerHeight * 0.72, 1);
  const ratio = clamp((window.scrollY - top) / range, 0, 1);

  dom.readingProgressBar.style.width = `${ratio * 100}%`;
  state.progress.chapterScroll[state.currentChapterId] = ratio;
  persistJson(STORAGE_KEYS.progress, state.progress);
  updateOverallProgress();
}

function updateOverallProgress() {
  const allRatios = state.manifest.chapters.map((chapter) => chapterProgress(chapter.id));
  const total = allRatios.reduce((sum, value) => sum + value, 0);
  const average = state.manifest.chapters.length ? total / state.manifest.chapters.length : 0;
  dom.overallProgressLabel.textContent = `${Math.round(average * 100)}%`;
  dom.overallProgressBar.style.width = `${average * 100}%`;
}

function syncTocSelection() {
  document.querySelectorAll("[data-chapter-id]").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.chapterId === state.currentChapterId);
  });
}

function scrollToChapterRatio(ratio) {
  const top = dom.chapterShell.getBoundingClientRect().top + window.scrollY;
  const range = Math.max(dom.chapterShell.offsetHeight - window.innerHeight * 0.72, 1);
  window.scrollTo({ top: top + range * ratio, behavior: "auto" });
}

function onSettingsInput() {
  state.settings.fontSize = Number(dom.fontSizeRange.value);
  state.settings.lineHeight = Number(dom.lineHeightRange.value);
  state.settings.contentWidth = Number(dom.contentWidthRange.value);
  persistJson(STORAGE_KEYS.settings, state.settings);
  applySettings();
}

function applySettings() {
  document.body.classList.remove("theme-paper", "theme-sepia", "theme-midnight");
  document.body.classList.add(`theme-${state.settings.theme}`);
  document.documentElement.style.setProperty("--reader-font-size", `${state.settings.fontSize}px`);
  document.documentElement.style.setProperty("--reader-line-height", String(state.settings.lineHeight));
  document.documentElement.style.setProperty("--reader-width", `${state.settings.contentWidth}px`);

  if (dom.fontSizeValue) {
    dom.fontSizeValue.textContent = `${state.settings.fontSize}px`;
  }
  if (dom.lineHeightValue) {
    dom.lineHeightValue.textContent = state.settings.lineHeight.toFixed(2);
  }
  if (dom.contentWidthValue) {
    dom.contentWidthValue.textContent = `${state.settings.contentWidth}px`;
  }

  document.querySelectorAll(".theme-chip").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.theme === state.settings.theme);
  });
}

function toggleBookmark() {
  if (!state.currentChapterId) {
    return;
  }
  if (state.bookmarks.has(state.currentChapterId)) {
    state.bookmarks.delete(state.currentChapterId);
  } else {
    state.bookmarks.add(state.currentChapterId);
  }
  persistJson(STORAGE_KEYS.bookmarks, [...state.bookmarks]);
  dom.bookmarkButton.textContent = state.bookmarks.has(state.currentChapterId) ? "取消收藏" : "收藏本章";
  renderToc();
}

function openDrawer(kind) {
  dom.appBackdrop.hidden = false;
  if (kind === "toc") {
    dom.tocDrawer.classList.add("is-open");
    dom.tocDrawer.setAttribute("aria-hidden", "false");
  } else {
    dom.settingsDrawer.classList.add("is-open");
    dom.settingsDrawer.setAttribute("aria-hidden", "false");
  }
}

function closeDrawers() {
  dom.appBackdrop.hidden = true;
  dom.tocDrawer.classList.remove("is-open");
  dom.settingsDrawer.classList.remove("is-open");
  dom.tocDrawer.setAttribute("aria-hidden", "true");
  dom.settingsDrawer.setAttribute("aria-hidden", "true");
}

function openSearch() {
  dom.searchOverlay.classList.remove("hidden");
  dom.searchOverlay.setAttribute("aria-hidden", "false");
  dom.searchInput.focus();
  syncBodyModalState();
}

function closeSearch() {
  dom.searchOverlay.classList.add("hidden");
  dom.searchOverlay.setAttribute("aria-hidden", "true");
  dom.searchInput.value = "";
  dom.searchResults.innerHTML = "";
  dom.searchMeta.textContent = "输入关键词后开始检索。";
  syncBodyModalState();
}

function handleSearchInput() {
  const query = dom.searchInput.value.trim();
  if (!query) {
    dom.searchMeta.textContent = "输入关键词后开始检索。";
    dom.searchResults.innerHTML = "";
    return;
  }

  const normalized = query.toLowerCase();
  const results = state.searchIndex
    .map((entry) => {
      const source = `${entry.title} ${entry.subtitle} ${entry.plainText}`.toLowerCase();
      const position = source.indexOf(normalized);
      if (position === -1) {
        return null;
      }
      const titleSource = `${entry.title} ${entry.subtitle}`.toLowerCase();
      let score = 0;
      if (titleSource.includes(normalized)) {
        score += 100;
      }
      if (entry.sectionTitle.toLowerCase().includes(normalized)) {
        score += 20;
      }
      score += Math.max(0, 18 - Math.floor(position / 360));
      return {
        ...entry,
        position,
        score,
        snippet: buildSnippet(entry.plainText, query),
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.score - left.score || left.position - right.position)
    .slice(0, 24);

  dom.searchMeta.textContent = `找到 ${results.length} 条结果`;
  dom.searchResults.innerHTML = results
    .map(
      (entry) => `
        <button class="search-result" type="button" data-search-chapter="${entry.id}">
          <div class="eyebrow">${escapeHtml(entry.sectionTitle)}</div>
          <h3>${escapeHtml(entry.title)}</h3>
          <p>${highlightSnippet(entry.snippet, query)}</p>
        </button>
      `,
    )
    .join("");

  dom.searchResults.querySelectorAll("[data-search-chapter]").forEach((button) => {
    button.addEventListener("click", async () => {
      const chapterId = button.dataset.searchChapter;
      closeSearch();
      await openChapter(chapterId, { restoreScroll: false, updateHash: true });
    });
  });
}

function openLightbox(image) {
  dom.lightboxImage.src = image.currentSrc || image.src;
  dom.lightboxImage.alt = image.alt || "";
  dom.lightboxCaption.textContent = image.alt || "";
  dom.lightbox.classList.remove("hidden");
  dom.lightbox.setAttribute("aria-hidden", "false");
  syncBodyModalState();
}

function closeLightbox() {
  dom.lightbox.classList.add("hidden");
  dom.lightbox.setAttribute("aria-hidden", "true");
  syncBodyModalState();
}

function toggleNoteMode() {
  state.noteMode = !state.noteMode;
  updateNoteModeButtons();
  if (state.noteMode) {
    showToast("批注模式已开启，点按段落即可写注。", 2400);
  } else {
    showToast("已退出批注模式。", 1800);
  }
}

function updateNoteModeButtons() {
  dom.noteModeButton.classList.toggle("is-active", state.noteMode);
  dom.dockNoteModeButton.classList.toggle("is-active", state.noteMode);
  dom.chapterContent.classList.toggle("note-mode", state.noteMode);
  dom.noteModeButton.textContent = state.noteMode ? "退出批注" : "批注模式";
  dom.dockNoteModeButton.textContent = state.noteMode ? "退出批注" : "批注";
}

function renderAnnotationState() {
  const chapterNotes = chapterAnnotations(state.currentChapterId);
  dom.chapterContent.querySelectorAll("[data-block-id]").forEach((block) => {
    const note = chapterNotes[block.dataset.blockId];
    block.classList.toggle("has-note", Boolean(note));
    if (note) {
      block.setAttribute("title", note.text);
    } else {
      block.removeAttribute("title");
    }
  });
  updateChapterTools();
}

function openNotesPanel() {
  renderNotesList();
  openPanel("notesOverlay");
}

function renderNotesList() {
  const notes = getChapterNotes(state.currentChapterId);
  dom.chapterNotesMeta.textContent = notes.length
    ? `本章共 ${notes.length} 条批注，按段落顺序排列。`
    : "本章还没有批注。先打开批注模式，再点正文段落写第一条。";

  dom.chapterNotesList.innerHTML = notes
    .map(
      (note) => `
        <article class="note-card">
          <div class="note-card__body">
            <p class="note-card__anchor">${escapeHtml(note.preview)}</p>
            <p class="note-card__text">${escapeHtml(note.text)}</p>
            <p class="note-card__time">${escapeHtml(note.updatedAt)}</p>
          </div>
          <div class="note-card__actions">
            <button class="secondary-button" type="button" data-jump-note="${note.blockId}">定位段落</button>
            <button class="ghost-button" type="button" data-edit-note="${note.blockId}">编辑</button>
          </div>
        </article>
      `,
    )
    .join("");

  dom.chapterNotesList.querySelectorAll("[data-jump-note]").forEach((button) => {
    button.addEventListener("click", () => {
      const blockId = button.dataset.jumpNote;
      closePanel("notesOverlay");
      scrollToBlock(blockId);
    });
  });

  dom.chapterNotesList.querySelectorAll("[data-edit-note]").forEach((button) => {
    button.addEventListener("click", () => {
      openNoteEditor(button.dataset.editNote);
    });
  });
}

function openNoteEditor(blockId) {
  const blockMeta = state.currentChapter?.blockIndex?.find((item) => item.id === blockId);
  if (!blockMeta) {
    return;
  }

  const note = chapterAnnotations(state.currentChapterId)[blockId];
  state.currentNoteTarget = {
    chapterId: state.currentChapterId,
    blockId,
    preview: blockMeta.preview,
    type: blockMeta.type,
  };

  dom.noteEditorTitle.textContent = note ? "编辑批注" : "新建批注";
  dom.noteContextPreview.textContent = blockMeta.preview;
  dom.noteTextarea.value = note?.text || "";
  dom.deleteNoteButton.disabled = !note;
  openPanel("noteEditorOverlay");

  window.setTimeout(() => {
    dom.noteTextarea.focus();
    dom.noteTextarea.setSelectionRange(dom.noteTextarea.value.length, dom.noteTextarea.value.length);
  }, 40);
}

function saveCurrentNote() {
  const target = state.currentNoteTarget;
  if (!target) {
    return;
  }

  const text = dom.noteTextarea.value.trim();
  if (!text) {
    showToast("批注内容不能为空。", 2200);
    return;
  }

  if (!state.annotations[target.chapterId]) {
    state.annotations[target.chapterId] = {};
  }

  state.annotations[target.chapterId][target.blockId] = {
    blockId: target.blockId,
    preview: target.preview,
    text,
    updatedAt: new Date().toLocaleString("zh-CN", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }),
  };

  persistJson(STORAGE_KEYS.annotations, state.annotations);
  closePanel("noteEditorOverlay");
  renderAnnotationState();
  renderNotesList();
  showToast("批注已保存。", 1800);
}

function deleteCurrentNote() {
  const target = state.currentNoteTarget;
  if (!target) {
    return;
  }

  const chapterNotes = state.annotations[target.chapterId];
  if (!chapterNotes?.[target.blockId]) {
    closePanel("noteEditorOverlay");
    return;
  }

  delete chapterNotes[target.blockId];
  if (!Object.keys(chapterNotes).length) {
    delete state.annotations[target.chapterId];
  }

  persistJson(STORAGE_KEYS.annotations, state.annotations);
  closePanel("noteEditorOverlay");
  renderAnnotationState();
  renderNotesList();
  showToast("批注已删除。", 1800);
}

function chapterAnnotations(chapterId) {
  return state.annotations[chapterId] || {};
}

function getChapterNotes(chapterId) {
  const notesById = chapterAnnotations(chapterId);
  const orderMap = new Map(
    (state.currentChapter?.blockIndex || []).map((block, index) => [block.id, index]),
  );

  return Object.values(notesById).sort((left, right) => {
    const leftOrder = orderMap.get(left.blockId) ?? Number.MAX_SAFE_INTEGER;
    const rightOrder = orderMap.get(right.blockId) ?? Number.MAX_SAFE_INTEGER;
    return leftOrder - rightOrder;
  });
}

function scrollToBlock(blockId) {
  const block = dom.chapterContent.querySelector(`[data-block-id="${CSS.escape(blockId)}"]`);
  if (!block) {
    showToast("这条批注对应的段落在当前版本里没有找到。", 2600);
    return;
  }

  block.scrollIntoView({ behavior: "smooth", block: "center" });
  block.classList.add("is-flashed");
  window.setTimeout(() => {
    block.classList.remove("is-flashed");
  }, 1400);
}

function openGlossaryPanel() {
  dom.glossarySearchInput.value = "";
  renderGlossaryList();
  openPanel("glossaryOverlay");
}

function renderGlossaryList() {
  const query = dom.glossarySearchInput.value.trim().toLowerCase();
  const currentTermIds = new Set(state.currentChapterTerms.map((item) => item.id));

  const entries = [...(state.glossary.entries || [])]
    .filter((entry) => {
      if (!query) {
        return true;
      }
      const haystack = [
        entry.title,
        entry.short,
        entry.english,
        entry.chapter,
        entry.definition,
        ...(entry.aliases || []),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(query);
    })
    .sort((left, right) => {
      const leftCurrent = currentTermIds.has(left.id) ? 1 : 0;
      const rightCurrent = currentTermIds.has(right.id) ? 1 : 0;
      if (rightCurrent !== leftCurrent) {
        return rightCurrent - leftCurrent;
      }
      return left.title.localeCompare(right.title, "zh-Hans-CN");
    });

  dom.glossarySummary.textContent = query
    ? `检索到 ${entries.length} 个术语。`
    : `当前章节命中 ${state.currentChapterTerms.length} 个术语，以下按“本章优先”排序。`;

  dom.glossaryList.innerHTML = entries
    .map(
      (entry) => `
        <button class="glossary-item${currentTermIds.has(entry.id) ? " is-current" : ""}" type="button" data-term-id="${entry.id}">
          <div class="glossary-item__top">
            <strong>${escapeHtml(entry.title)}</strong>
            <span>${escapeHtml(entry.chapter || "")}</span>
          </div>
          <p>${escapeHtml(entry.definition)}</p>
          <div class="glossary-item__meta">
            ${entry.short ? `<span>${escapeHtml(entry.short)}</span>` : ""}
            ${entry.english ? `<span>${escapeHtml(entry.english)}</span>` : ""}
          </div>
        </button>
      `,
    )
    .join("");

  dom.glossaryList.querySelectorAll("[data-term-id]").forEach((button) => {
    button.addEventListener("click", () => {
      openTermCard(button.dataset.termId);
    });
  });
}

function openTermCard(termId) {
  const entry = state.glossaryById.get(termId);
  if (!entry) {
    return;
  }

  dom.termChapter.textContent = entry.chapter || "术语卡";
  dom.termTitle.textContent = entry.title;
  dom.termShort.textContent = entry.short || "无缩写";
  dom.termShort.classList.toggle("is-muted", !entry.short);
  dom.termEnglish.textContent = entry.english || "无英文名";
  dom.termEnglish.classList.toggle("is-muted", !entry.english);
  dom.termDefinition.textContent = entry.definition;
  openPanel("termOverlay");
}

function openReferencesPanel() {
  renderReferenceCards();
  openPanel("referencesOverlay");
}

function renderReferenceCards() {
  const cards = state.currentChapter?.referenceCards || [];
  dom.referencesMeta.textContent = cards.length
    ? `本章共 ${cards.length} 张脚注卡，正文末尾的资料来源已被折叠到这里。`
    : "本章暂无脚注卡片。";

  dom.referencesList.innerHTML = cards
    .map(
      (card) => `
        <article class="reference-card">
          <div class="reference-card__head">
            <p class="eyebrow">脚注卡 ${escapeHtml(card.id)}</p>
            <h3>${escapeHtml(card.title)}</h3>
          </div>
          <div class="reference-card__body">${card.html}</div>
        </article>
      `,
    )
    .join("");
}

function openPanel(panelId) {
  closeDrawers();
  dom[panelId].classList.remove("hidden");
  dom[panelId].setAttribute("aria-hidden", "false");
  syncBodyModalState();
}

function closePanel(panelId) {
  dom[panelId].classList.add("hidden");
  dom[panelId].setAttribute("aria-hidden", "true");
  if (panelId === "noteEditorOverlay") {
    state.currentNoteTarget = null;
  }
  syncBodyModalState();
}

function closeAllPanels() {
  for (const panelId of PANEL_IDS) {
    closePanel(panelId);
  }
}

function syncBodyModalState() {
  const hasModal =
    !dom.searchOverlay.classList.contains("hidden") ||
    !dom.lightbox.classList.contains("hidden") ||
    PANEL_IDS.some((panelId) => !dom[panelId].classList.contains("hidden"));
  document.body.classList.toggle("body-modal-open", hasModal);
}

function decorateGlossaryTerms(root) {
  if (!state.glossaryRegex) {
    return [];
  }

  const counts = new Map();
  const nodes = [];
  const walker = document.createTreeWalker(
    root,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        if (!node.textContent.trim()) {
          return NodeFilter.FILTER_REJECT;
        }
        const parent = node.parentElement;
        if (!parent) {
          return NodeFilter.FILTER_REJECT;
        }
        if (parent.closest(".term-trigger")) {
          return NodeFilter.FILTER_REJECT;
        }
        if (GLOSSARY_EXCLUDE_TAGS.has(parent.tagName)) {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    },
  );

  while (walker.nextNode()) {
    nodes.push(walker.currentNode);
  }

  for (const node of nodes) {
    replaceTermsInTextNode(node, counts);
  }

  return [...counts.keys()]
    .map((termId) => state.glossaryById.get(termId))
    .filter(Boolean);
}

function replaceTermsInTextNode(textNode, counts) {
  const sourceText = textNode.textContent;
  state.glossaryRegex.lastIndex = 0;

  let lastAcceptedIndex = 0;
  let found = false;
  const fragment = document.createDocumentFragment();

  for (let match = state.glossaryRegex.exec(sourceText); match; match = state.glossaryRegex.exec(sourceText)) {
    const alias = match[0];
    const entry = state.glossaryByAlias.get(alias);
    if (!entry) {
      continue;
    }

    const start = match.index;
    const end = start + alias.length;
    if (isAsciiLike(alias) && !hasAsciiBoundary(sourceText, start, end)) {
      continue;
    }

    if ((counts.get(entry.id) || 0) >= 3) {
      continue;
    }

    if (start > lastAcceptedIndex) {
      fragment.append(sourceText.slice(lastAcceptedIndex, start));
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "term-trigger";
    button.dataset.termId = entry.id;
    button.textContent = alias;
    fragment.append(button);

    counts.set(entry.id, (counts.get(entry.id) || 0) + 1);
    lastAcceptedIndex = end;
    found = true;
  }

  if (!found) {
    return;
  }

  if (lastAcceptedIndex < sourceText.length) {
    fragment.append(sourceText.slice(lastAcceptedIndex));
  }

  textNode.replaceWith(fragment);
}

function isAsciiLike(value) {
  return /^[A-Za-z0-9._+\-]+$/.test(value);
}

function hasAsciiBoundary(text, start, end) {
  const left = text[start - 1] || "";
  const right = text[end] || "";
  return !/[A-Za-z0-9_]/.test(left) && !/[A-Za-z0-9_]/.test(right);
}

async function checkForUpdates({ silent = false } = {}) {
  const response = await fetch(withCacheBust("content/book.json", true), { cache: "no-store" });
  const freshManifest = await response.json();
  if (freshManifest.book.version !== state.contentVersion) {
    showToast(`检测到新版本 ${freshManifest.book.version}，点击刷新以载入。`, 5200, () => {
      location.reload();
    });
    return;
  }
  if (!silent) {
    showToast("已经是最新版本。", 2400);
  }
}

function showToast(message, duration = 2600, onClick = null) {
  dom.toast.textContent = message;
  dom.toast.classList.remove("hidden");
  dom.toast.onclick = onClick;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    dom.toast.classList.add("hidden");
    dom.toast.onclick = null;
  }, duration);
}

function onHashChange() {
  const chapterId = chapterIdFromHash();
  if (chapterId && chapterId !== state.currentChapterId) {
    openChapter(chapterId, { restoreScroll: true, updateHash: false });
  }
}

function chapterIdFromHash() {
  const hash = location.hash.replace(/^#/, "").trim();
  return hash || null;
}

function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) {
    return;
  }
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch((error) => {
      console.warn("service worker register failed", error);
    });
  });
}

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (error) {
    return fallback;
  }
}

function persistJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function withCacheBust(url, force) {
  if (!force) {
    return url;
  }
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}ts=${Date.now()}`;
}

function throttle(fn, wait) {
  let timer = null;
  return (...args) => {
    if (timer) {
      return;
    }
    timer = window.setTimeout(() => {
      fn(...args);
      timer = null;
    }, wait);
  };
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function formatWan(count) {
  return (count / 10000).toFixed(1);
}

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function buildSnippet(text, query) {
  const normalizedText = text.toLowerCase();
  const normalizedQuery = query.toLowerCase();
  const index = normalizedText.indexOf(normalizedQuery);
  if (index === -1) {
    return text.slice(0, 120);
  }
  const start = Math.max(index - 34, 0);
  const end = Math.min(index + query.length + 56, text.length);
  const prefix = start > 0 ? "…" : "";
  const suffix = end < text.length ? "…" : "";
  return `${prefix}${text.slice(start, end)}${suffix}`;
}

function highlightSnippet(snippet, query) {
  const safeSnippet = escapeHtml(snippet);
  const escaped = escapeRegExp(query);
  return safeSnippet.replace(new RegExp(escaped, "gi"), (match) => `<mark>${match}</mark>`);
}

function escapeRegExp(value = "") {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
