// Search page: query presets and showing the alpha slider for hybrid mode only.
(function () {
  "use strict";

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
})();
