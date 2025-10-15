// Background Remover JavaScript

document.addEventListener("DOMContentLoaded", function () {
  // Get DOM elements
  const uploadForm = document.getElementById("uploadForm");
  const fileInput = document.getElementById("fileInput");
  const fileLabel = document.querySelector(".file-upload-label span");
  const processBtn = document.getElementById("processBtn");
  const uploadSection = document.querySelector(".upload-section");
  const loadingSection = document.getElementById("loadingSection");
  const resultsSection = document.getElementById("resultsSection");
  const errorSection = document.getElementById("errorSection");
  const originalImage = document.getElementById("originalImage");
  const processedImage = document.getElementById("processedImage");
  const downloadBtn = document.getElementById("downloadBtn");
  const errorMessage = document.getElementById("errorMessage");

  // File input change handler
  fileInput.addEventListener("change", function (e) {
    const file = e.target.files[0];
    if (file) {
      fileLabel.textContent = file.name;

      // Show preview of selected image
      const reader = new FileReader();
      reader.onload = function (e) {
        // You could add a preview here if desired
      };
      reader.readAsDataURL(file);
    } else {
      fileLabel.textContent = "Choose Image";
    }
  });

  // Form submit handler
  uploadForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      showError("Please select an image file.");
      return;
    }

    // Validate file type
    const allowedTypes = ["image/png", "image/jpeg", "image/jpg", "image/gif", "image/bmp", "image/tiff"];
    if (!allowedTypes.includes(file.type)) {
      showError("Please select a valid image file (PNG, JPG, GIF, BMP, or TIFF).");
      return;
    }

    // Validate file size (16MB max)
    const maxSize = 16 * 1024 * 1024; // 16MB in bytes
    if (file.size > maxSize) {
      showError("File size must be less than 16MB.");
      return;
    }

    // Show loading state
    showLoading();

    // Create FormData object
    const formData = new FormData();
    formData.append("file", file);
    formData.append("method", "ultimate");

    // Send request to server
    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        if (data.success) {
          showResults(data, file);
        } else {
          showError(data.error || "An error occurred while processing the image.");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        showError("Failed to connect to the server. Please try again.");
      });
  });

  // Show loading state
  function showLoading() {
    hideAllSections();
    loadingSection.style.display = "block";
    processBtn.disabled = true;
    processBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Processing...</span>';
  }

  // Show results
  function showResults(data, originalFile) {
    hideAllSections();

    // Set original image
    const originalReader = new FileReader();
    originalReader.onload = function (e) {
      originalImage.src = e.target.result;
    };
    originalReader.readAsDataURL(originalFile);

    // Set processed image
    processedImage.src = data.preview_url;

    // Set download link
    downloadBtn.href = data.download_url;
    downloadBtn.download = `background_removed_${originalFile.name.split(".")[0]}.png`;

    // Show results section
    resultsSection.style.display = "block";

    // Reset form
    resetFormState();
  }

  // Show error
  function showError(message) {
    hideAllSections();
    errorMessage.textContent = message;
    errorSection.style.display = "block";
    resetFormState();
  }

  // Hide all sections
  function hideAllSections() {
    uploadSection.style.display = "none";
    loadingSection.style.display = "none";
    resultsSection.style.display = "none";
    errorSection.style.display = "none";
  }

  // Reset form state
  function resetFormState() {
    processBtn.disabled = false;
    processBtn.innerHTML = '<i class="fas fa-magic"></i><span>Remove Background</span>';
  }

  // Reset form (global function)
  window.resetForm = function () {
    // Reset form
    uploadForm.reset();
    fileLabel.textContent = "Choose Image";

    // Show upload section
    hideAllSections();
    uploadSection.style.display = "block";

    // Reset form state
    resetFormState();
  };

  // Drag and drop functionality
  const fileUploadLabel = document.querySelector(".file-upload-label");

  fileUploadLabel.addEventListener("dragover", function (e) {
    e.preventDefault();
    fileUploadLabel.classList.add("drag-over");
  });

  fileUploadLabel.addEventListener("dragleave", function (e) {
    e.preventDefault();
    fileUploadLabel.classList.remove("drag-over");
  });

  fileUploadLabel.addEventListener("drop", function (e) {
    e.preventDefault();
    fileUploadLabel.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      fileInput.files = files;
      fileLabel.textContent = files[0].name;
    }
  });

  // Add drag over styling
  const style = document.createElement("style");
  style.textContent = `
        .file-upload-label.drag-over {
            border-color: #5a6fd8 !important;
            background: #e8ecff !important;
            transform: scale(1.02);
        }
    `;
  document.head.appendChild(style);

  // Image loading error handlers
  originalImage.addEventListener("error", function () {
    console.error("Failed to load original image");
  });

  processedImage.addEventListener("error", function () {
    console.error("Failed to load processed image");
    showError("Failed to load the processed image. Please try again.");
  });

  // Download tracking
  downloadBtn.addEventListener("click", function () {
    // Track download event (you can add analytics here)
    console.log("Image downloaded");

    // Optional: Show success message
    setTimeout(() => {
      const downloadText = downloadBtn.querySelector("span");
      const originalText = downloadText.textContent;
      downloadText.textContent = "Downloaded!";
      downloadBtn.style.background = "linear-gradient(135deg, #28a745, #20c997)";

      setTimeout(() => {
        downloadText.textContent = originalText;
        downloadBtn.style.background = "linear-gradient(135deg, #28a745, #20c997)";
      }, 2000);
    }, 100);
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", function (e) {
    // Escape key to reset form
    if (e.key === "Escape") {
      resetForm();
    }

    // Enter key to submit form (when file is selected)
    if (e.key === "Enter" && fileInput.files.length > 0) {
      e.preventDefault();
      uploadForm.dispatchEvent(new Event("submit"));
    }
  });

  // Prevent default drag behaviors on the document
  document.addEventListener("dragover", function (e) {
    e.preventDefault();
  });

  document.addEventListener("drop", function (e) {
    e.preventDefault();
  });
});
