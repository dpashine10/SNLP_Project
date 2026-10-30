// University integration page: interactive Group 6 -> Group 7 bridge simulator.
(function () {
  "use strict";

  var simulator = document.querySelector("[data-bridge-simulator]");
  if (!simulator) {
    return;
  }
  simulator.hidden = false;

  var SCENARIOS = {
    at_risk: {
      id: "at_risk",
      label: "At-Risk CS 521 (natural language processing)",
      studentMessage:
        "I am falling behind on the transformer attention coding lab in CS 521 and worried I might fail next week's midterm.",
      triage: {
        category: "Academic Support",
        urgency: "Elevated (Early-Warning)",
        course: "CS 521 -> natural language processing",
        decision: "handoff_eligible"
      },
      bridgeAction: "forward",
      query: "natural language processing",
      guardrailNote: ""
    },
    no_risk: {
      id: "no_risk",
      label: "Routine Inquiry (no_risk guardrail)",
      studentMessage:
        "Hello registrar team, could you confirm how long an official PDF transcript request takes for a summer internship application?",
      triage: {
        category: "Administrative / Registrar",
        urgency: "Routine",
        course: "None detected",
        decision: "no_risk (blocked)"
      },
      bridgeAction: "block_no_risk",
      query: "",
      guardrailNote:
        "Guardrail active (no_risk): Routine administrative messages do not indicate academic risk. Zero HTTP requests were sent to Group 7 (/api/v1/recommend was not called)."
    },
    needs_topic: {
      id: "needs_topic",
      label: "Unmapped Course HIST 499 (needs_topic guardrail)",
      studentMessage:
        "I am struggling to narrow down primary archival sources and methodology for my HIST 499 senior thesis seminar.",
      triage: {
        category: "Academic Support",
        urgency: "Elevated (Early-Warning)",
        course: "HIST 499 (unmapped in catalog)",
        decision: "needs_topic (awaiting staff override)"
      },
      bridgeAction: "block_needs_topic",
      query: "",
      defaultOverride: "information retrieval and archival text analysis",
      guardrailNote:
        "Guardrail active (needs_topic): Course code HIST 499 is not in the verified topic catalog. Zero HTTP requests were sent to Group 7 until a staff reviewer supplies a sanitized research topic below."
    }
  };

  var scenarioButtons = Array.prototype.slice.call(simulator.querySelectorAll("[data-scenario]"));
  var messageEl = simulator.querySelector("[data-sim-message]");
  var catEl = simulator.querySelector("[data-sim-category]");
  var urgencyEl = simulator.querySelector("[data-sim-urgency]");
  var courseEl = simulator.querySelector("[data-sim-course]");
  var decisionEl = simulator.querySelector("[data-sim-decision]");
  var bridgeBadgeEl = simulator.querySelector("[data-sim-bridge-badge]");
  var requestJsonEl = simulator.querySelector("[data-sim-request-json]");
  var overrideBox = simulator.querySelector("[data-sim-override-box]");
  var overrideForm = simulator.querySelector("[data-sim-override-form]");
  var overrideInput = simulator.querySelector("[data-sim-topic-input]");
  var overrideBtn = simulator.querySelector("[data-sim-override-submit]");
  var statusEl = simulator.querySelector("[data-sim-status]");
  var resultsContainer = simulator.querySelector("[data-sim-results]");
  var rawDrawer = simulator.querySelector("[data-sim-raw-drawer]");
  var responseJsonEl = simulator.querySelector("[data-sim-response-json]");

  var activeRequestId = 0;

  function setActiveButton(scenarioId) {
    scenarioButtons.forEach(function (btn) {
      var active = btn.dataset.scenario === scenarioId;
      btn.setAttribute("aria-pressed", active ? "true" : "false");
      btn.classList.toggle("is-active", active);
    });
  }

  function setBusy(busy) {
    scenarioButtons.forEach(function (btn) {
      btn.disabled = busy;
    });
    if (overrideBtn) {
      overrideBtn.disabled = busy;
    }
  }

  function safeHttpUrl(raw) {
    var text = String(raw || "").trim();
    if (text.indexOf("https://") === 0 || text.indexOf("http://") === 0) {
      return text;
    }
    return "";
  }

  function setStatusBanner(kind, text) {
    if (!statusEl) {
      return;
    }
    statusEl.hidden = false;
    statusEl.className = "alert sim-status-banner " + kind;
    statusEl.textContent = text;
  }

  function clearResults() {
    if (resultsContainer) {
      resultsContainer.textContent = "";
      resultsContainer.hidden = true;
    }
    if (rawDrawer) {
      rawDrawer.hidden = true;
      rawDrawer.open = false;
    }
    if (responseJsonEl) {
      responseJsonEl.textContent = "";
    }
  }

  function updateLocalTriagePane(scenario, decisionOverride) {
    if (messageEl) {
      messageEl.textContent = scenario.studentMessage;
    }
    if (catEl) {
      catEl.textContent = scenario.triage.category;
    }
    if (urgencyEl) {
      urgencyEl.textContent = scenario.triage.urgency;
    }
    if (courseEl) {
      courseEl.textContent = scenario.triage.course;
    }
    if (decisionEl) {
      decisionEl.textContent = decisionOverride || scenario.triage.decision;
    }
  }

  function renderRecommendationCards(payload) {
    if (!resultsContainer) {
      return;
    }
    resultsContainer.textContent = "";
    if (payload.reasoning && payload.reasoning.answer) {
      var rBox = document.createElement("div");
      rBox.className = "card reasoning-card sim-reasoning-card";
      var prov = payload.reasoning.provider_used || "ai";
      var model = payload.reasoning.model_name || "";
      var rHeader = document.createElement("div");
      rHeader.className = "reasoning-header";
      var rTitle = document.createElement("h3");
      rTitle.className = "reasoning-title";
      rTitle.innerHTML = '<span class="reasoning-icon">✦</span> Grounded Advisor AI Reasoning (' + prov + ' · ' + model + ')';
      rHeader.appendChild(rTitle);
      rBox.appendChild(rHeader);
      var rBody = document.createElement("div");
      rBody.className = "reasoning-body";
      rBody.innerHTML = (payload.reasoning.answer || "")
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/\n\n/g, "<br><br>").replace(/\n/g, "<br>")
        .replace(/\[Paper #(\d+)[^\]]*\]/g, '<span class="citation-pill">$&</span>');
      rBox.appendChild(rBody);
      resultsContainer.appendChild(rBox);
    }

    var results = Array.isArray(payload.results) ? payload.results : [];
    if (!results.length) {
      var empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No publication matches returned for this topic.";
      resultsContainer.appendChild(empty);
      resultsContainer.hidden = false;
      return;
    }

    var list = document.createElement("ol");
    list.className = "result-list sim-result-list";

    results.forEach(function (rec) {
      var li = document.createElement("li");
      var card = document.createElement("article");
      card.className = "card researcher-card sim-researcher-card";

      var header = document.createElement("header");
      header.className = "researcher-header";

      var headingWrap = document.createElement("div");
      headingWrap.className = "researcher-heading";

      var h3 = document.createElement("h3");
      h3.className = "researcher-name";
      var rankSpan = document.createElement("span");
      rankSpan.className = "rank";
      rankSpan.textContent = "#" + String(rec.rank || "");
      h3.appendChild(rankSpan);
      h3.appendChild(document.createTextNode(" " + String(rec.name || "Researcher")));
      headingWrap.appendChild(h3);

      var subMeta = document.createElement("p");
      subMeta.className = "sim-card-submeta muted";
      subMeta.textContent = "Matched publications: " + String(rec.matched_papers || 0) + " · Advisory review required";
      var profileUrl = safeHttpUrl(rec.author_id);
      if (profileUrl) {
        subMeta.appendChild(document.createTextNode(" · "));
        var link = document.createElement("a");
        link.className = "external-link";
        link.href = profileUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "OpenAlex Profile ↗";
        subMeta.appendChild(link);
      }
      headingWrap.appendChild(subMeta);

      var badges = document.createElement("div");
      badges.className = "researcher-badges";

      var conf = rec.confidence || {};
      var level = String(conf.level || "MEDIUM").toUpperCase();
      var levelClass =
        level === "HIGH"
          ? "badge-confidence-high"
          : level === "LOW"
            ? "badge-confidence-low"
            : "badge-confidence-medium";
      var confBadge = document.createElement("span");
      confBadge.className = "badge badge-confidence " + levelClass;
      var confPct = typeof conf.score === "number" ? Math.round(conf.score * 100) + "%" : "";
      confBadge.textContent = confPct ? level + " · " + confPct : level;
      if (conf.explanation) {
        confBadge.title = String(conf.explanation);
      }
      badges.appendChild(confBadge);

      var scoreBadge = document.createElement("span");
      scoreBadge.className = "badge badge-score";
      var rawScore = typeof rec.score === "number" ? rec.score.toFixed(4) : String(rec.score || "0");
      scoreBadge.textContent = "Score: " + rawScore;
      badges.appendChild(scoreBadge);

      header.appendChild(headingWrap);
      header.appendChild(badges);
      card.appendChild(header);

      if (conf.explanation) {
        var confNote = document.createElement("p");
        confNote.className = "confidence-note";
        confNote.textContent = String(conf.explanation);
        card.appendChild(confNote);
      }

      var evidenceList = Array.isArray(rec.evidence) ? rec.evidence : [];
      if (evidenceList.length > 0) {
        var evBox = document.createElement("div");
        evBox.className = "evidence";
        var evLabel = document.createElement("p");
        evLabel.className = "evidence-label";
        evLabel.textContent = "Supporting OpenAlex Publications";
        evBox.appendChild(evLabel);

        var ul = document.createElement("ul");
        ul.className = "supporting-paper-list sim-evidence-list";
        evidenceList.slice(0, 3).forEach(function (ev) {
          var item = document.createElement("li");
          item.className = "supporting-paper-item";

          var titleSpan = document.createElement("span");
          titleSpan.className = "supporting-paper-title";
          var cite = document.createElement("cite");
          cite.textContent = "“" + String(ev.title || "Untitled") + "”";
          titleSpan.appendChild(cite);
          var simText = typeof ev.similarity === "number" ? ev.similarity.toFixed(4) : String(ev.similarity || "");
          var authorsCount = ev.authors_count ? " · " + String(ev.authors_count) + " author(s)" : "";
          titleSpan.appendChild(
            document.createTextNode(" (" + String(ev.year || "N/A") + ") · sim " + simText + authorsCount)
          );
          item.appendChild(titleSpan);

          var workUrl = safeHttpUrl(ev.paper_id);
          if (workUrl) {
            var workLink = document.createElement("a");
            workLink.className = "evidence-link";
            workLink.href = workUrl;
            workLink.target = "_blank";
            workLink.rel = "noopener noreferrer";
            workLink.textContent = "OpenAlex ↗";
            item.appendChild(workLink);
          }
          ul.appendChild(item);
        });
        evBox.appendChild(ul);
        card.appendChild(evBox);
      }

      var topics = Array.isArray(rec.topics) ? rec.topics : [];
      if (topics.length > 0) {
        var topicWrap = document.createElement("div");
        topicWrap.className = "topics";
        var topicUl = document.createElement("ul");
        topicUl.className = "badge-list";
        topics.slice(0, 6).forEach(function (t) {
          var tLi = document.createElement("li");
          tLi.className = "badge";
          tLi.textContent = String(t);
          topicUl.appendChild(tLi);
        });
        topicWrap.appendChild(topicUl);
        card.appendChild(topicWrap);
      }

      li.appendChild(card);
      list.appendChild(li);
    });

    resultsContainer.appendChild(list);
    resultsContainer.hidden = false;
  }

  function dispatchSanitizedTopic(topic, sourceLabel) {
    var cleanTopic = String(topic || "").trim();
    if (!cleanTopic) {
      setStatusBanner("alert-warning", "Enter a non-empty research topic before sending an override to Group 7.");
      return;
    }

    var requestBody = {
      query: cleanTopic,
      top_k: 5,
      include_evidence: true,
      include_reasoning: true
    };
    if (bridgeBadgeEl) {
      bridgeBadgeEl.className = "badge badge-confidence badge-confidence-high";
      bridgeBadgeEl.textContent = "POST /api/v1/recommend · 0 PII fields";
    }
    if (requestJsonEl) {
      requestJsonEl.textContent =
        "POST /api/v1/recommend\nContent-Type: application/json\n\n" +
        JSON.stringify(requestBody, null, 2);
    }

    clearResults();
    var reqId = ++activeRequestId;
    setBusy(true);
    setStatusBanner(
      "alert-warning",
      "Calling live Group 7 /api/v1/recommend with sanitized topic “" + cleanTopic + "” (warming up model on first call if unloaded)…"
    );

    fetch("/api/v1/recommend", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(requestBody)
    })
      .then(function (resp) {
        return resp.json().then(function (body) {
          return { ok: resp.ok, status: resp.status, body: body };
        });
      })
      .then(function (result) {
        if (reqId !== activeRequestId) {
          return;
        }
        setBusy(false);
        if (!result.ok) {
          var errMsg = (result.body && result.body.message) ? result.body.message : ("HTTP " + result.status);
          setStatusBanner("alert-error", "Group 7 API returned an error: " + errMsg);
          return;
        }
        var data = result.body || {};
        var strategy = (data.model_info && data.model_info.strategy) || "top_k_average_author_discount";
        var ms = typeof data.execution_time_ms === "number" ? data.execution_time_ms.toFixed(1) + " ms" : "live";
        setStatusBanner(
          "alert-success",
          sourceLabel +
            ": Returned " +
            String(data.total_results || 0) +
            " researcher publication matches for “" +
            cleanTopic +
            "” in " +
            ms +
            " (" +
            strategy +
            "). Human staff review required — zero student message text crossed the bridge."
        );
        renderRecommendationCards(data);
        if (rawDrawer && responseJsonEl) {
          responseJsonEl.textContent = JSON.stringify(data, null, 2);
          rawDrawer.hidden = false;
        }
      })
      .catch(function () {
        if (reqId !== activeRequestId) {
          return;
        }
        setBusy(false);
        setStatusBanner("alert-error", "Could not reach /api/v1/recommend on this server. Check that the Flask service is running.");
      });
  }

  function selectScenario(scenarioId) {
    var scenario = SCENARIOS[scenarioId];
    if (!scenario) {
      return;
    }
    activeRequestId += 1;
    setBusy(false);
    setActiveButton(scenarioId);
    updateLocalTriagePane(scenario);

    if (scenario.bridgeAction === "forward") {
      if (overrideBox) {
        overrideBox.hidden = true;
      }
      dispatchSanitizedTopic(scenario.query, "Bridge handoff (CS 521 -> topic)");
      return;
    }

    clearResults();
    if (bridgeBadgeEl) {
      bridgeBadgeEl.className = "badge badge-confidence badge-confidence-medium";
      bridgeBadgeEl.textContent =
        scenario.bridgeAction === "block_no_risk"
          ? "Blocked · no_risk (0 HTTP calls)"
          : "Paused · needs_topic (0 HTTP calls)";
    }
    if (requestJsonEl) {
      requestJsonEl.textContent =
        "// Outbound HTTP call suppressed by Group 6 bridge guardrail (" +
        (scenario.bridgeAction === "block_no_risk" ? "no_risk" : "needs_topic") +
        ").\n// Student message text and identifiers remain inside Group 6.";
    }
    if (overrideBox) {
      var showOverride = scenario.bridgeAction === "block_needs_topic";
      overrideBox.hidden = !showOverride;
      if (showOverride && overrideInput && !overrideInput.value.trim()) {
        overrideInput.value = scenario.defaultOverride || "";
      }
    }
    setStatusBanner("alert-warning", scenario.guardrailNote);
  }

  scenarioButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      selectScenario(btn.dataset.scenario);
    });
  });

  if (overrideForm && overrideInput) {
    overrideForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var customTopic = overrideInput.value.trim();
      updateLocalTriagePane(SCENARIOS.needs_topic, "advisor_override -> handoff_sent");
      dispatchSanitizedTopic(customTopic, "Advisor topic override (HIST 499)");
    });
  }

  // Initialize local preview with CS 521 staged (ready to run in one click without firing unsolicited POST on page load).
  setActiveButton("");
  updateLocalTriagePane(SCENARIOS.at_risk, "ready_to_simulate (click a preset above)");
  if (bridgeBadgeEl) {
    bridgeBadgeEl.className = "badge badge-diversity";
    bridgeBadgeEl.textContent = "Select a scenario above";
  }
  if (requestJsonEl) {
    requestJsonEl.textContent =
      'POST /api/v1/recommend\nContent-Type: application/json\n\n{\n  "query": "natural language processing",\n  "top_k": 5,\n  "include_evidence": true\n}';
  }
})();

