// Search page: query presets, mode-specific controls, researcher shortlist, and outreach drafts.
(function () {
  "use strict";

  var STORAGE_KEY = "snlp_researcher_shortlist_v1";
  var memoryStore = [];
  var textarea = document.getElementById("q");

  // Presets fill the query box without searching, as before.
  var presets = document.querySelector("[data-presets]");
  if (presets && textarea) {
    presets.hidden = false;
    presets.querySelectorAll("[data-preset]").forEach(function (button) {
      button.addEventListener("click", function () {
        textarea.value = button.dataset.preset;
        textarea.focus();
      });
    });
  }

  // Fields marked data-mode-only="<mode>" are shown only for that retrieval mode.
  var modeSelect = document.querySelector("[data-mode-select]");
  if (modeSelect) {
    var toggle = function () {
      document.querySelectorAll("[data-mode-only]").forEach(function (field) {
        field.hidden = field.dataset.modeOnly !== modeSelect.value;
      });
    };
    modeSelect.addEventListener("change", toggle);
    toggle();
  }

  function loadShortlist() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return memoryStore.slice();
      }
      var parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        memoryStore = parsed;
        return parsed;
      }
    } catch (e) {
      // Fall back to in-memory shortlist when storage is restricted.
    }
    return memoryStore.slice();
  }

  function saveShortlist(items) {
    memoryStore = items.slice();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(memoryStore));
    } catch (e) {
      // Ignore quota/privacy mode errors; in-memory state still works on page.
    }
  }

  function entryFromDataset(ds) {
    return {
      authorId: ds.authorId || "",
      name: ds.authorName || "Researcher",
      institution: ds.institution || "",
      authorUrl: ds.authorUrl || "",
      confidence: ds.confidence || "",
      similarity: ds.similarity || "",
      nPapers: ds.nPapers || "",
      paperTitle: ds.paperTitle || "N/A",
      paperYear: ds.paperYear || "N/A",
      doiUrl: ds.doiUrl || "",
      topics: ds.topics || "",
      query: ds.query || ""
    };
  }

  function buildOutreachDraft(entry) {
    var topicLine = entry.query ? entry.query : "shared research areas";
    var subject = "Subject: Research collaboration inquiry — " + topicLine;
    var paperClause =
      entry.paperTitle && entry.paperTitle !== "N/A"
        ? 'your publication "' + entry.paperTitle + '" (' + entry.paperYear + ")"
        : "your published research profile";
    var focusAreas = entry.topics
      ? entry.topics.split(",").slice(0, 3).join(", ").trim()
      : topicLine;
    var instNote = entry.institution ? " at " + entry.institution : "";

    return [
      subject,
      "",
      "Dear Dr. " + entry.name + ",",
      "",
      "I came across " + paperClause + instNote + ' while exploring work on "' + topicLine + '". ' +
        "Your contributions in " + focusAreas + " align closely with our current inquiry, " +
        "and I would value a brief conversation regarding potential academic collaboration or research guidance.",
      "",
      "Best regards,",
      "[Your Name]",
      "[Your Affiliation]"
    ].join("\n");
  }

  function buildMarkdownBrief(items) {
    var lines = [
      "# Researcher Collaboration Shortlist Brief",
      "",
      "Generated via Research Discovery & Collaboration Engine (`https://research.dpashine.tech`)",
      ""
    ];
    items.forEach(function (item, idx) {
      var heading = (idx + 1) + ". **" + item.name + "**" + (item.institution ? " — " + item.institution : "");
      lines.push(heading);
      lines.push(
        "   - **Confidence & Score:** " + (item.confidence || "N/A") +
          " · Similarity `" + (item.similarity || "N/A") + "` · `" + (item.nPapers || "0") + "` indexed papers"
      );
      var paperLine = '   - **Matched Publication:** "' + item.paperTitle + '" (' + item.paperYear + ")";
      if (item.doiUrl) {
        paperLine += " — " + item.doiUrl;
      }
      lines.push(paperLine);
      if (item.topics) {
        lines.push("   - **Key Topics:** " + item.topics);
      }
      if (item.query) {
        lines.push("   - **Search Context:** `" + item.query + "`");
      }
      if (item.authorUrl) {
        lines.push("   - **OpenAlex Profile:** " + item.authorUrl);
      }
      lines.push("");
    });
    return lines.join("\n");
  }

  function csvEscape(val) {
    var str = String(val == null ? "" : val);
    if (/[",\r\n]/.test(str)) {
      return '"' + str.replace(/"/g, '""') + '"';
    }
    return str;
  }

  function buildCsvBrief(items) {
    var headers = [
      "Name",
      "Institution",
      "Confidence",
      "Similarity",
      "Indexed_Papers",
      "Matched_Paper",
      "Year",
      "DOI_URL",
      "Topics",
      "Query",
      "OpenAlex_URL"
    ];
    var rows = [headers.join(",")];
    items.forEach(function (item) {
      rows.push(
        [
          item.name,
          item.institution,
          item.confidence,
          item.similarity,
          item.nPapers,
          item.paperTitle,
          item.paperYear,
          item.doiUrl,
          item.topics,
          item.query,
          item.authorUrl
        ]
          .map(csvEscape)
          .join(",")
      );
    });
    return rows.join("\r\n");
  }

  function copyText(text, onDone) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { onDone(true); },
        function () { fallbackCopy(text, onDone); }
      );
    } else {
      fallbackCopy(text, onDone);
    }
  }

  function fallbackCopy(text, onDone) {
    try {
      var temp = document.createElement("textarea");
      temp.value = text;
      temp.setAttribute("readonly", "");
      temp.style.position = "fixed";
      temp.style.opacity = "0";
      document.body.appendChild(temp);
      temp.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(temp);
      onDone(Boolean(ok));
    } catch (e) {
      onDone(false);
    }
  }

  function downloadFile(filename, content, mimeType) {
    var blob = new Blob([content], { type: mimeType });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(function () { URL.revokeObjectURL(url); }, 500);
  }

  function flashFeedback(el, message) {
    if (!el) {
      return;
    }
    el.hidden = false;
    el.textContent = message;
    clearTimeout(el._timer);
    el._timer = setTimeout(function () {
      el.textContent = "";
      el.hidden = true;
    }, 2600);
  }

  var drawer = document.querySelector("[data-shortlist-drawer]");
  var countBadge = drawer ? drawer.querySelector("[data-shortlist-count]") : null;
  var listEl = drawer ? drawer.querySelector("[data-shortlist-list]") : null;
  var drawerStatus = drawer ? drawer.querySelector("[data-shortlist-status]") : null;
  var cardContainers = Array.prototype.slice.call(document.querySelectorAll("[data-card-actions]"));

  function syncCardButtons(items) {
    var selectedIds = {};
    items.forEach(function (it) {
      selectedIds[it.authorId] = true;
    });
    cardContainers.forEach(function (container) {
      container.hidden = false;
      var toggleBtn = container.querySelector("[data-shortlist-toggle]");
      if (!toggleBtn) {
        return;
      }
      var isSaved = Boolean(selectedIds[container.dataset.authorId]);
      toggleBtn.setAttribute("aria-pressed", isSaved ? "true" : "false");
      toggleBtn.classList.toggle("is-shortlisted", isSaved);
      toggleBtn.textContent = isSaved ? "✓ Shortlisted" : "+ Shortlist";
    });
  }

  function renderDrawer() {
    var items = loadShortlist();
    syncCardButtons(items);
    if (!drawer || !listEl || !countBadge) {
      return;
    }
    countBadge.textContent = String(items.length);
    drawer.hidden = items.length === 0;
    listEl.textContent = "";

    items.forEach(function (item) {
      var li = document.createElement("li");
      li.className = "shortlist-item";

      var info = document.createElement("div");
      info.className = "shortlist-item-info";

      var titleRow = document.createElement("p");
      titleRow.className = "shortlist-item-name";
      var strong = document.createElement("strong");
      strong.textContent = item.name;
      titleRow.appendChild(strong);
      if (item.institution) {
        var instSpan = document.createElement("span");
        instSpan.className = "shortlist-item-inst";
        instSpan.textContent = " · " + item.institution;
        titleRow.appendChild(instSpan);
      }
      info.appendChild(titleRow);

      var paperRow = document.createElement("p");
      paperRow.className = "shortlist-item-paper muted";
      paperRow.textContent =
        'Matched paper: "' + item.paperTitle + '" (' + item.paperYear + ")" +
        (item.query ? " · Query: " + item.query : "");
      info.appendChild(paperRow);

      var actions = document.createElement("div");
      actions.className = "shortlist-item-actions";

      var draftBtn = document.createElement("button");
      draftBtn.type = "button";
      draftBtn.className = "button button-secondary button-small";
      draftBtn.textContent = "Copy Outreach Draft";
      draftBtn.addEventListener("click", function () {
        copyText(buildOutreachDraft(item), function (ok) {
          flashFeedback(drawerStatus, ok ? "Copied outreach draft for " + item.name + "." : "Clipboard copy unavailable.");
        });
      });

      var removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "button button-secondary button-small";
      removeBtn.setAttribute("aria-label", "Remove " + item.name + " from shortlist");
      removeBtn.textContent = "Remove";
      removeBtn.addEventListener("click", function () {
        var next = loadShortlist().filter(function (entry) {
          return entry.authorId !== item.authorId;
        });
        saveShortlist(next);
        renderDrawer();
      });

      actions.appendChild(draftBtn);
      actions.appendChild(removeBtn);
      li.appendChild(info);
      li.appendChild(actions);
      listEl.appendChild(li);
    });
  }

  cardContainers.forEach(function (container) {
    var toggleBtn = container.querySelector("[data-shortlist-toggle]");
    var outreachBtn = container.querySelector("[data-copy-outreach]");
    var feedbackEl = container.querySelector("[data-action-feedback]");

    if (toggleBtn) {
      toggleBtn.addEventListener("click", function () {
        var current = loadShortlist();
        var entry = entryFromDataset(container.dataset);
        var exists = current.some(function (it) { return it.authorId === entry.authorId; });
        var next;
        if (exists) {
          next = current.filter(function (it) { return it.authorId !== entry.authorId; });
          flashFeedback(feedbackEl, "Removed from shortlist");
        } else {
          next = current.concat([entry]);
          flashFeedback(feedbackEl, "Added to shortlist");
        }
        saveShortlist(next);
        renderDrawer();
      });
    }

    if (outreachBtn) {
      outreachBtn.addEventListener("click", function () {
        var entry = entryFromDataset(container.dataset);
        copyText(buildOutreachDraft(entry), function (ok) {
          flashFeedback(feedbackEl, ok ? "Copied outreach draft ✓" : "Copy unavailable");
        });
      });
    }
  });

  if (drawer) {
    drawer.querySelectorAll("[data-shortlist-export]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        var items = loadShortlist();
        if (!items.length) {
          return;
        }
        if (btn.dataset.shortlistExport === "csv") {
          downloadFile("researcher-shortlist.csv", buildCsvBrief(items), "text/csv;charset=utf-8");
          flashFeedback(drawerStatus, "Downloaded researcher-shortlist.csv");
        } else {
          downloadFile("researcher-shortlist.md", buildMarkdownBrief(items), "text/markdown;charset=utf-8");
          flashFeedback(drawerStatus, "Downloaded researcher-shortlist.md");
        }
      });
    });

    var copyBriefBtn = drawer.querySelector("[data-shortlist-copy-brief]");
    if (copyBriefBtn) {
      copyBriefBtn.addEventListener("click", function () {
        var items = loadShortlist();
        if (!items.length) {
          return;
        }
        copyText(buildMarkdownBrief(items), function (ok) {
          flashFeedback(drawerStatus, ok ? "Copied Markdown shortlist brief to clipboard ✓" : "Clipboard copy unavailable.");
        });
      });
    }

    var clearBtn = drawer.querySelector("[data-shortlist-clear]");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        saveShortlist([]);
        renderDrawer();
      });
    }
  }

  // Live Grounded Reasoning Trigger via POST /api/v1/reason
  var runReasoningBtn = document.querySelector("[data-run-reasoning]");
  var liveProviderSelect = document.querySelector("[data-live-provider]");
  var reasoningBody = document.getElementById("reasoning-body");
  var reasoningMeta = document.getElementById("reasoning-meta");

  if (runReasoningBtn && reasoningBody && reasoningMeta) {
    runReasoningBtn.addEventListener("click", function () {
      var queryInput = document.getElementById("q");
      var query = queryInput ? queryInput.value.trim() : "";
      if (!query) {
        return;
      }

      var cards = document.querySelectorAll(".researcher-card");
      var papers = [];
      cards.forEach(function (card) {
        var actions = card.querySelector("[data-card-actions]");
        var ds = actions ? actions.dataset : {};
        var citeEl = card.querySelector(".evidence-title cite");
        var title = ds.paperTitle || (citeEl ? citeEl.textContent.replace(/[“”]/g, "").trim() : "");
        var year = ds.paperYear || "2024";
        var doiUrl = ds.doiUrl || "";
        var author = ds.authorName || "";
        var abstractEl = card.querySelector(".abstract-text");
        var abstract = abstractEl ? abstractEl.textContent.trim() : "";
        if (title && title !== "N/A" && papers.length < 8) {
          papers.push({
            paper_index: papers.length + 1,
            title: title,
            year: year,
            abstract: abstract,
            doi: doiUrl,
            authors: author ? [author] : []
          });
        }
      });

      var provider = liveProviderSelect ? liveProviderSelect.value : "auto";
      runReasoningBtn.disabled = true;
      runReasoningBtn.textContent = "Synthesizing with " + provider.toUpperCase() + "…";
      reasoningBody.hidden = false;
      reasoningMeta.hidden = false;
      reasoningBody.innerHTML = '<div class="reasoning-loading">Grounding reasoning in ' + papers.length + ' OpenAlex publications…</div>';

      fetch("/api/v1/reason", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          provider: provider,
          papers: papers
        })
      })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        runReasoningBtn.disabled = false;
        runReasoningBtn.textContent = "Regenerate Answer";
        if (data.error && !data.answer) {
          reasoningBody.innerHTML = '<p class="alert alert-error">Reasoning error: ' + (data.message || data.error) + '</p>';
          return;
        }

        var prov = data.provider_used || provider;
        var model = data.model_name || "grounded-model";
        var lat = Math.round(data.execution_time_ms || 0);
        var isFallback = data.is_fallback ? ' <span class="badge badge-warning" title="' + (data.error || "") + '">Extractive Fallback</span>' : '';
        reasoningMeta.innerHTML =
          '<span class="badge badge-reasoning-provider">Provider: <strong>' + prov + '</strong> (' + model + ')</span>' +
          '<span class="badge badge-reasoning-latency">Latency: ' + lat + 'ms</span>' +
          isFallback +
          '<span class="badge badge-reasoning-citations">' + (data.citations ? data.citations.length : 0) + ' Grounded Citations</span>';

        var formatted = (data.answer || "")
          .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
          .replace(/\n\n/g, "<br><br>")
          .replace(/\n/g, "<br>")
          .replace(/\[Paper #(\d+)[^\]]*\]/g, '<span class="citation-pill" data-citation="$1">$&</span>');

        reasoningBody.innerHTML = formatted;
      })
      .catch(function (err) {
        runReasoningBtn.disabled = false;
        runReasoningBtn.textContent = "Retry Grounded Answer";
        reasoningBody.innerHTML = '<p class="alert alert-error">Failed to reach reasoning service: ' + err.message + '</p>';
      });
    });
  }

  renderDrawer();
})();
