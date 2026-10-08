// Data tables: click a column header to sort, type in the filter box to narrow rows.
(function () {
  "use strict";

  function cellValue(row, index) {
    return row.cells[index].textContent.trim();
  }

  function compare(a, b) {
    var numA = Number(a);
    var numB = Number(b);
    if (a !== "" && b !== "" && !isNaN(numA) && !isNaN(numB)) {
      return numA - numB;
    }
    return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
  }

  function makeSortable(table) {
    var headers = Array.prototype.slice.call(table.tHead.rows[0].cells);
    var body = table.tBodies[0];

    headers.forEach(function (th, index) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "sort-button";
      button.textContent = th.textContent;
      var indicator = document.createElement("span");
      indicator.className = "sort-indicator";
      indicator.setAttribute("aria-hidden", "true");
      button.appendChild(indicator);
      th.textContent = "";
      th.appendChild(button);

      button.addEventListener("click", function () {
        var ascending = th.getAttribute("aria-sort") !== "ascending";
        headers.forEach(function (other) {
          other.removeAttribute("aria-sort");
        });
        th.setAttribute("aria-sort", ascending ? "ascending" : "descending");

        var rows = Array.prototype.slice.call(body.rows);
        rows.sort(function (rowA, rowB) {
          var result = compare(cellValue(rowA, index), cellValue(rowB, index));
          return ascending ? result : -result;
        });
        rows.forEach(function (row) {
          body.appendChild(row);
        });
      });
    });
  }

  function makeFilterable(label) {
    var table = document.getElementById(label.dataset.tableFilter);
    var input = label.querySelector("input");
    var count = document.getElementById(label.dataset.tableFilter + "-count");
    if (!table || !input) {
      return;
    }
    label.hidden = false;
    input.addEventListener("input", function () {
      var needle = input.value.trim().toLowerCase();
      var rows = table.tBodies[0].rows;
      var shown = 0;
      Array.prototype.forEach.call(rows, function (row) {
        var match = row.textContent.toLowerCase().indexOf(needle) !== -1;
        row.hidden = !match;
        shown += match ? 1 : 0;
      });
      if (count) {
        count.textContent = needle ? shown + " of " + rows.length + " rows" : rows.length + " rows";
      }
    });
  }

  document.querySelectorAll("table[data-sortable]").forEach(makeSortable);
  document.querySelectorAll("[data-table-filter]").forEach(makeFilterable);
})();
