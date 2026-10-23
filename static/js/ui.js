// Visual behaviour only: sidebar rail/expand, sliding nav indicator, staggered entrances.
(function () {
  "use strict";

  var root = document.documentElement;
  var narrow = window.matchMedia("(max-width: 900px)");

  // ---- Sidebar: desktop = icon rail <-> expanded; small screens = overlay ----
  var toggle = document.getElementById("sidebar-toggle");
  var sidebar = document.getElementById("sidebar");
  var scrim = document.querySelector("[data-sidebar-scrim]");
  var sideBody = sidebar ? sidebar.querySelector(".side-body") : null;

  // Stagger index drives the sequential fade/slide when the sidebar expands.
  if (sidebar) {
    sidebar.querySelectorAll(".side-link, .panel-title, .field, .stat, .panel > p").forEach(function (el, i) {
      el.style.setProperty("--i", i);
    });
  }

  function isOpen() {
    return !root.classList.contains("sidebar-collapsed");
  }

  function setInert(el, on) {
    if (!el) {
      return;
    }
    if (on) {
      el.setAttribute("inert", "");
    } else {
      el.removeAttribute("inert");
    }
  }

  function sync() {
    var open = isOpen();
    if (toggle) {
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }
    // Overlay: hidden entirely when closed. Desktop: the rail icons stay usable,
    // only the settings/statistics body leaves the tab order (form values still submit).
    setInert(sidebar, !open && narrow.matches);
    setInert(sideBody, !open && !narrow.matches);
  }

  function setOpen(open, remember) {
    root.classList.toggle("sidebar-collapsed", !open);
    if (remember && !narrow.matches) {
      try {
        localStorage.setItem("sidebar-open", open ? "1" : "0");
      } catch (e) {}
    }
    sync();
  }

  function clearContextTargets() {
    if (!sidebar) {
      return;
    }
    sidebar.querySelectorAll(".context-panel.is-targeted").forEach(function (panel) {
      panel.classList.remove("is-targeted");
    });
    sidebar.querySelectorAll("[data-context-target][aria-current='true']").forEach(function (link) {
      link.removeAttribute("aria-current");
    });
  }

  function revealContextTarget(targetId, focusTarget) {
    if (!sidebar || !targetId) {
      return;
    }
    var target = document.getElementById(targetId);
    if (!target) {
      return;
    }
    var panel = target.closest(".context-panel") || target;
    clearContextTargets();
    setOpen(true, true);
    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        panel.classList.add("is-targeted");
        var link = sidebar.querySelector('[data-context-target="' + targetId + '"]');
        if (link) {
          link.setAttribute("aria-current", "true");
        }
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
        if (focusTarget && typeof target.focus === "function") {
          target.focus({ preventScroll: true });
        }
      });
    });
  }

  if (toggle && sidebar) {
    toggle.addEventListener("click", function () {
      setOpen(!isOpen(), true);
    });
    if (scrim) {
      scrim.addEventListener("click", function () {
        setOpen(false, false);
        toggle.focus();
      });
    }
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && narrow.matches && isOpen()) {
        setOpen(false, false);
        toggle.focus();
      }
    });
    // Dataset / search shortcuts open the sidebar first so their target is visible.
    sidebar.querySelectorAll("[data-side-expand]").forEach(function (link) {
      link.addEventListener("click", function (event) {
        var targetId = link.getAttribute("data-context-target");
        if (targetId && link.getAttribute("href").charAt(0) === "#") {
          event.preventDefault();
          revealContextTarget(targetId, true);
        } else if (!isOpen()) {
          setOpen(true, true);
        }
      });
    });
    var onBreakpoint = function () {
      var stored = null;
      try {
        stored = localStorage.getItem("sidebar-open");
      } catch (e) {}
      setOpen(!narrow.matches && stored !== "0", false);
    };
    if (narrow.addEventListener) {
      narrow.addEventListener("change", onBreakpoint);
    }
    sync();

    if (window.location.hash === "#stats-heading" || window.location.hash === "#config-heading") {
      revealContextTarget(window.location.hash.slice(1), true);
    }
  }

  // ---- University integration health ----------------------------------------
  var group6Health = document.querySelector("[data-group6-health]");
  if (group6Health) {
    var group6Dot = group6Health.querySelector(".service-health-dot");
    var group6Text = group6Health.querySelector("span:last-child");

    var renderGroup6Health = function (health) {
      var state = health && health.state ? health.state : "unreachable";
      var label = state === "ready" || state === "ok"
        ? "reachable"
        : state === "incompatible"
          ? "wrong endpoint"
          : state === "unhealthy"
            ? "unhealthy"
            : "unreachable";
      group6Health.classList.remove("is-ready", "is-warning", "is-error");
      group6Health.classList.add(state === "ready" || state === "ok" ? "is-ready" : state === "incompatible" || state === "unhealthy" ? "is-warning" : "is-error");
      group6Health.title = (health && health.detail) || "Group 6 integration status unavailable.";
      if (group6Text) {
        group6Text.textContent = "Group 6: " + label;
      }
      if (group6Dot) {
        group6Dot.setAttribute("aria-label", label);
      }
    };

    var refreshGroup6Health = function () {
      fetch("/api/integration-status", { cache: "no-store" })
        .then(function (response) { return response.json(); })
        .then(renderGroup6Health)
        .catch(function () { renderGroup6Health({ state: "unreachable", detail: "Could not query the Group 6 health endpoint." }); });
    };

    refreshGroup6Health();
    window.setInterval(refreshGroup6Health, 15000);
  }

  // ---- Header tabs: one segment slides between tabs -------------------------
  var nav = document.querySelector(".main-nav");
  if (nav) {
    var links = Array.prototype.slice.call(nav.querySelectorAll(".nav-link"));
    var active = nav.querySelector(".nav-link[aria-current='page']");
    var indicator = document.createElement("span");
    indicator.className = "nav-indicator";
    indicator.setAttribute("aria-hidden", "true");
    nav.appendChild(indicator);

    var place = function (link, hover) {
      if (!link) {
        indicator.classList.remove("is-ready");
        return;
      }
      var navBox = nav.getBoundingClientRect();
      var box = link.getBoundingClientRect();
      indicator.style.width = box.width + "px";
      indicator.style.transform = "translateX(" + (box.left - navBox.left) + "px)";
      indicator.classList.add("is-ready");
      indicator.classList.toggle("is-hover", Boolean(hover) && link !== active);
    };

    var rest = function () {
      place(active, false);
    };

    if (active) {
      nav.classList.add("has-indicator");
      // Start where the previous page's tab was, then glide to this one.
      var from = null;
      try {
        from = sessionStorage.getItem("nav-from");
        sessionStorage.setItem("nav-from", active.getAttribute("href"));
      } catch (e) {}
      var start = links.filter(function (l) {
        return l.getAttribute("href") === from;
      })[0];
      indicator.style.transition = "none";
      place(start && start !== active ? start : active, false);
      void indicator.offsetWidth;
      indicator.style.transition = "";
      if (start && start !== active) {
        requestAnimationFrame(rest);
      }
    }

    links.forEach(function (link) {
      link.addEventListener("mouseenter", function () {
        place(link, true);
      });
      link.addEventListener("focus", function () {
        place(link, true);
      });
      link.addEventListener("blur", rest);
    });
    nav.addEventListener("mouseleave", rest);
    window.addEventListener("resize", rest);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(rest);
    }
  }

  // ---- Result cards: stagger index for the entrance animation ---------------
  document.querySelectorAll(".result-list > li").forEach(function (item, i) {
    item.style.setProperty("--i", Math.min(i, 10));
  });
  document.querySelectorAll(".compare-column").forEach(function (item, i) {
    item.style.animationDelay = i * 40 + "ms";
  });
})();
