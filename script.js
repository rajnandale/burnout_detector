const API_URL = "http://127.0.0.1:8000/predict";

// DOM elements
const feedbackInput = document.getElementById("feedbackInput");
const submitBtn = document.getElementById("submitBtn");
const resultArea = document.getElementById("resultArea");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const wordCount = document.getElementById("wordCount");

// Word count functionality
feedbackInput.addEventListener("input", () => {
  const words = feedbackInput.value.trim().split(/\s+/).filter(word => word.length > 0);
  wordCount.textContent = words.length;
});

// Submit functionality
submitBtn.addEventListener("click", async () => {
  const text = feedbackInput.value.trim();
  
  if (!text) {
    showError("Please enter some text to analyze.");
    return;
  }
  
  if (text.split(/\s+/).length < 5) {
    showError("Please provide more detailed feedback (at least 5 words) for better analysis.");
    return;
  }
  
  await analyzeBurnout(text);
});

// Enter key to submit
feedbackInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && e.ctrlKey) {
    submitBtn.click();
  }
});

async function analyzeBurnout(text) {
  // Show loading state
  showLoading();
  hideError();
  hideResults();
  
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();
    displayResults(data);
    
  } catch (error) {
    console.error("Error:", error);
    showError("Failed to analyze your feedback. Please try again or check if the server is running.");
  } finally {
    hideLoading();
  }
}

function displayResults(data) {
  // Display risk level
  const riskLevelDiv = document.getElementById("riskLevel");
  const riskLevel = data.risk_level;
  
  let riskHTML = "";
  let confidenceClass = "";
  
  switch (riskLevel) {
    case "Low Risk":
      riskHTML = `
        <div class="inline-flex items-center px-6 py-3 bg-green-100 text-green-800 rounded-full">
          <i class="fas fa-check-circle text-2xl mr-3"></i>
          <div>
            <div class="text-2xl font-bold">Low Risk</div>
            <div class="text-sm">You're doing well!</div>
          </div>
        </div>
      `;
      confidenceClass = "bg-green-100 text-green-800";
      break;
      
    case "Moderate Risk":
      riskHTML = `
        <div class="inline-flex items-center px-6 py-3 bg-yellow-100 text-yellow-800 rounded-full">
          <i class="fas fa-exclamation-triangle text-2xl mr-3"></i>
          <div>
            <div class="text-2xl font-bold">Moderate Risk</div>
            <div class="text-sm">Pay attention to warning signs</div>
          </div>
        </div>
      `;
      confidenceClass = "bg-yellow-100 text-yellow-800";
      break;
      
    case "High Risk":
      riskHTML = `
        <div class="inline-flex items-center px-6 py-3 bg-red-100 text-red-800 rounded-full">
          <i class="fas fa-times-circle text-2xl mr-3"></i>
          <div>
            <div class="text-2xl font-bold">High Risk</div>
            <div class="text-sm">Immediate attention recommended</div>
          </div>
        </div>
      `;
      confidenceClass = "bg-red-100 text-red-800";
      break;
  }
  
  riskLevelDiv.innerHTML = riskHTML;
  
  // Display confidence
  const confidenceBadge = document.getElementById("confidenceBadge");
  confidenceBadge.className = `px-3 py-1 rounded-full text-sm font-medium ${confidenceClass}`;
  confidenceBadge.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;
  
  // Display probabilities
  const probabilitiesDiv = document.getElementById("probabilities");
  probabilitiesDiv.innerHTML = `
    <div class="text-center p-4 bg-green-50 rounded-lg">
      <div class="text-2xl font-bold text-green-600">${(data.probabilities.low_risk * 100).toFixed(1)}%</div>
      <div class="text-sm text-green-700">Low Risk</div>
    </div>
    <div class="text-center p-4 bg-yellow-50 rounded-lg">
      <div class="text-2xl font-bold text-yellow-600">${(data.probabilities.moderate_risk * 100).toFixed(1)}%</div>
      <div class="text-sm text-yellow-700">Moderate Risk</div>
    </div>
    <div class="text-center p-4 bg-red-50 rounded-lg">
      <div class="text-2xl font-bold text-red-600">${(data.probabilities.high_risk * 100).toFixed(1)}%</div>
      <div class="text-sm text-red-700">High Risk</div>
    </div>
  `;
  
  // Display explanations
  const explanationList = document.getElementById("explanationList");
  explanationList.innerHTML = data.explanation.map(exp => 
    `<div class="flex items-start">
      <i class="fas fa-info-circle text-blue-500 mt-1 mr-3 flex-shrink-0"></i>
      <p class="text-gray-700">${exp}</p>
    </div>`
  ).join("");
  
  // Display recommendations
  const recommendationsList = document.getElementById("recommendationsList");
  recommendationsList.innerHTML = data.recommendations.map(rec => 
    `<div class="flex items-start">
      <i class="fas fa-arrow-right text-blue-500 mt-1 mr-3 flex-shrink-0"></i>
      <p class="text-gray-700">${rec}</p>
    </div>`
  ).join("");
  
  // Show results
  showResults();
  
  // Smooth scroll to results
  resultArea.scrollIntoView({ behavior: "smooth", block: "start" });
}

function showLoading() {
  loadingState.classList.remove("hidden");
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Analyzing...';
}

function hideLoading() {
  loadingState.classList.add("hidden");
  submitBtn.disabled = false;
  submitBtn.innerHTML = '<i class="fas fa-search mr-2"></i>Analyze Burnout Risk';
}

function showResults() {
  resultArea.classList.remove("hidden");
}

function hideResults() {
  resultArea.classList.add("hidden");
}

function showError(message) {
  const errorMessage = document.getElementById("errorMessage");
  errorMessage.textContent = message;
  errorState.classList.remove("hidden");
  hideLoading();
}

function hideError() {
  errorState.classList.add("hidden");
}

// Add some example text buttons for testing
function addExampleButtons() {
  const examples = [
    {
      text: "I love my job! The team is supportive, workload is manageable, and I feel valued. I'm learning new skills and growing professionally.",
      label: "Positive Experience"
    },
    {
      text: "Work has been busy lately and I'm feeling a bit overwhelmed. Deadlines are tight but achievable. I'm managing okay but could use more breaks.",
      label: "Moderate Stress"
    },
    {
      text: "I'm completely exhausted and burnt out. I can't keep up with the workload anymore. I'm working 12-hour days and still behind. I feel like I'm drowning.",
      label: "High Burnout"
    }
  ];
  
  const exampleDiv = document.createElement("div");
  exampleDiv.className = "bg-gray-50 rounded-lg p-4 mb-6";
  exampleDiv.innerHTML = `
    <h3 class="text-lg font-semibold text-gray-800 mb-3">Try these examples:</h3>
    <div class="flex flex-wrap gap-2">
      ${examples.map(ex => `
        <button 
          onclick="loadExample('${ex.text.replace(/'/g, "\\'")}')"
          class="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm hover:bg-blue-200 transition-colors"
        >
          ${ex.label}
        </button>
      `).join("")}
    </div>
  `;
  
  // Insert after the input section
  const inputSection = document.querySelector(".bg-white.rounded-xl.shadow-lg.p-6.mb-6");
  inputSection.parentNode.insertBefore(exampleDiv, inputSection.nextSibling);
}

function loadExample(text) {
  feedbackInput.value = text;
  feedbackInput.dispatchEvent(new Event("input")); // Trigger word count
}

// Initialize example buttons when page loads
document.addEventListener("DOMContentLoaded", addExampleButtons); 