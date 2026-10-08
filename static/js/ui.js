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
    // Statistics / Settings shortcuts open the sidebar first so their target is visible.
    sidebar.querySelectorAll("[data-side-expand]").forEach(function (link) {
      link.addEventListener("click", function () {
        if (!isOpen()) {
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
