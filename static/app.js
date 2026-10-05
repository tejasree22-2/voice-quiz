document.addEventListener("DOMContentLoaded", () => {
  fetch("/health")
    .then((res) => res.json())
    .then((data) => {
      document.getElementById("status").textContent = `Server: ${data.status}`;
    })
    .catch(() => {
      document.getElementById("status").textContent = "Server unreachable.";
    });
});
