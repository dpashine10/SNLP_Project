// Behaviour shared by every page: live range values and busy submit buttons.
(function () {
  "use strict";

  // Show a range input's current value in its <output> element.
  document.querySelectorAll("input[type='range'][data-output]").forEach(function (input) {
    var output = document.getElementById(input.dataset.output);
    if (!output) {
      return;
    }
    var decimals = Number(input.dataset.decimals || 0);
    var update = function () {
      output.value = Number(input.value).toFixed(decimals);
    };
    input.addEventListener("input", update);
    update();
  });

  // Mark the submit button as busy while a form request is running.
  var busyButtons = document.querySelectorAll("button[type='submit'][data-busy-label]");
  busyButtons.forEach(function (button) {
    var idleLabel = button.textContent;
    button.form.addEventListener("submit", function () {
      button.setAttribute("aria-busy", "true");
      button.textContent = button.dataset.busyLabel;
    });
    // Restore the label when the page comes back from the back/forward cache.
    window.addEventListener("pageshow", function (event) {
      if (event.persisted) {
        button.removeAttribute("aria-busy");
        button.textContent = idleLabel;
      }
    });
  });
})();
