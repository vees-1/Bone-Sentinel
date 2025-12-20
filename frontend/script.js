const imageInput = document.getElementById("imageInput");
const fileNameText = document.getElementById("fileName");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");
const loader = document.getElementById("loader");

const resultDiv = document.getElementById("result");
const predictionText = document.getElementById("prediction");
const confidenceText = document.getElementById("confidence");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files[0];
  if (!selectedFile) return;

  fileNameText.textContent = selectedFile.name;
  previewContainer.classList.remove("hidden");
  analyzeBtn.classList.remove("hidden");

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
  };
  reader.readAsDataURL(selectedFile);
});

analyzeBtn.addEventListener("click", () => {
  loader.classList.remove("hidden");
  resultDiv.classList.add("hidden");

  setTimeout(() => {
    const isFracture = Math.random() > 0.5;
    const confidence = (Math.random() * 15 + 80).toFixed(2);

    predictionText.textContent =
      "Prediction: " + (isFracture ? "Abnormal (Fracture)" : "Normal");
    confidenceText.textContent = "Confidence: " + confidence + "%";

    loader.classList.add("hidden");
    resultDiv.classList.remove("hidden");
  }, 1800);
});
