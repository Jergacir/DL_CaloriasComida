document.addEventListener("DOMContentLoaded", function () {
  // Variables globales
  let selectedFile = null;
  // Object URL para la imagen subida (usado para preview sin pasar por backend)
  let currentPreviewURL = null;

  // Elementos DOM
  const uploadArea = document.getElementById("uploadArea");
  const fileInput = document.getElementById("fileInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loading = document.getElementById("loading");
  const uploadSection = document.getElementById("uploadSection");
  const resultsSection = document.getElementById("resultsSection");
  const errorMessage = document.getElementById("errorMessage");
  const reset = document.getElementById("reset");

  // Evento para resetear análisis
  reset.addEventListener("click", resetAnalysis);

  // Click en área de upload
  uploadArea.addEventListener("click", () => {
    fileInput.click();
  });

  // Selección de archivo
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      selectedFile = file;
      uploadArea.innerHTML = `
                    <div class="upload-icon">✅</div>
                    <div class="upload-text">Archivo seleccionado:</div>
                    <div class="upload-hint">${file.name}</div>
                `;
      analyzeBtn.style.display = "block";
    }
  });

  // Drag and drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      selectedFile = file;
      fileInput.files = e.dataTransfer.files;
      uploadArea.innerHTML = `
                    <div class="upload-icon">✅</div>
                    <div class="upload-text">Archivo seleccionado:</div>
                    <div class="upload-hint">${file.name}</div>
                `;
      analyzeBtn.style.display = "block";
    }
  });

  // Analizar imagen
  analyzeBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    // Crear preview local (object URL) para mostrar la imagen sin depender del backend
    if (currentPreviewURL) {
      URL.revokeObjectURL(currentPreviewURL);
      currentPreviewURL = null;
    }
    currentPreviewURL = URL.createObjectURL(selectedFile);

    // Ocultar upload, mostrar loading
    uploadSection.style.display = "none";
    loading.style.display = "block";
    errorMessage.style.display = "none";

    // Crear FormData
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      // Hacer request
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        // Mostrar resultados (usar currentPreviewURL para la imagen)
        displayResults(data);
      } else {
        // Mostrar error
        showError(data.error || "Error desconocido");
        // Liberar preview si hubo error
        if (currentPreviewURL) {
          URL.revokeObjectURL(currentPreviewURL);
          currentPreviewURL = null;
        }
      }
    } catch (error) {
      showError("Error de conexión: " + error.message);
      if (currentPreviewURL) {
        URL.revokeObjectURL(currentPreviewURL);
        currentPreviewURL = null;
      }
    } finally {
      loading.style.display = "none";
    }
  });

  // Mostrar resultados
  function displayResults(data) {
    const res = data.resultado;
    const top3 = data.top3;

    // Mostrar imagen
    const imgSrc = currentPreviewURL
      ? currentPreviewURL
      : data.imagen_url || "";
    document.getElementById("imagePreview").innerHTML = `
                <img src="${imgSrc}" alt="Imagen analizada">
            `;

    // Métricas
    document.getElementById("categoria").textContent = res.clase;
    document.getElementById("confianza").textContent =
      res.probabilidad.toFixed(1) + "%";

    if (res.calorias) {
      document.getElementById("calorias").textContent =
        Math.round(res.calorias) + " kcal";
      document.getElementById("caloriasCard").style.display = "block";
    } else {
      document.getElementById("caloriasCard").style.display = "none";
    }

    // Top-3
    const top3HTML = top3
      .map(
        (item, index) => `
                <div class="top3-item">
                    <div class="top3-rank">${index + 1}</div>
                    <div class="top3-name">${item.clase}</div>
                    <div class="top3-prob">${item.probabilidad.toFixed(
                      1
                    )}%</div>
                </div>
            `
      )
      .join("");

    document.getElementById("top3List").innerHTML = top3HTML;

    // Mostrar sección de resultados
    resultsSection.style.display = "block";
  }

  // Mostrar error
  function showError(message) {
    errorMessage.textContent = "❌ " + message;
    errorMessage.style.display = "block";
    uploadSection.style.display = "block";
  }

  // Reset análisis
  function resetAnalysis() {
    selectedFile = null;
    if (currentPreviewURL) {
      URL.revokeObjectURL(currentPreviewURL);
      currentPreviewURL = null;
    }
    fileInput.value = "";
    analyzeBtn.style.display = "none";
    resultsSection.style.display = "none";
    uploadSection.style.display = "block";
    errorMessage.style.display = "none";

    uploadArea.innerHTML = `
                <div class="upload-icon">📸</div>
                <div class="upload-text">Arrastra una imagen aquí</div>
                <div class="upload-hint">o haz clic para seleccionar</div>
            `;
  }
});
