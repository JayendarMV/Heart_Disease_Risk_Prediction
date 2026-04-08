/**
 * script.js — HeartGuard AI Frontend Logic
 * ==========================================
 * Handles form submission, API communication, result rendering,
 * risk gauge animation, and IoT simulation.
 */

// ── API base URL ──
// If opened via Live Server or file path, point to the Flask backend directly
let API_BASE = window.location.origin;
if (window.location.protocol === 'file:' || window.location.port !== '5000') {
    // Also protect against assuming it's local if deployed. 
    // In production we usually don't have ports like 5500.
    if (window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' || window.location.protocol === 'file:') {
        API_BASE = 'http://127.0.0.1:5000';
    }
}

// ── DOM references ──
const form = document.getElementById("predictionForm");
const submitBtn = document.getElementById("submitBtn");
const btnText = submitBtn.querySelector(".btn-text");
const btnLoader = submitBtn.querySelector(".btn-loader");
const resultsSection = document.getElementById("resultsSection");
const gaugeValue = document.getElementById("gaugeValue");
const riskBadge = document.getElementById("riskBadge");
const explanationList = document.getElementById("explanationList");
const contributionsChart = document.getElementById("contributionsChart");
const recommendationsList = document.getElementById("recommendationsList");
const simulateIoTBtn = document.getElementById("simulateIoT");

// ── Form submission ──
form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Gather form data
    const formData = new FormData(form);
    const data = {};

    // Only include non-empty values
    for (const [key, value] of formData.entries()) {
        if (value !== "" && value !== null) {
            // Convert numeric fields
            if (["age", "trestbps", "chol", "thalch", "oldpeak"].includes(key)) {
                data[key] = parseFloat(value);
            } else if (["fbs", "exang"].includes(key)) {
                data[key] = value === "true";
            } else {
                data[key] = value;
            }
        }
    }

    // Validate mandatory fields
    if (!data.age || !data.sex || !data.cp) {
        showError("Please fill in all mandatory fields (Age, Gender, Chest Pain Type).");
        return;
    }

    // Show loading
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Prediction failed");
        }

        renderResults(result);
    } catch (error) {
        showError(error.message || "Failed to connect to the server. Please ensure the backend is running.");
    } finally {
        setLoading(false);
    }
});

// ── IoT Simulation ──
simulateIoTBtn.addEventListener("click", () => {
    // Simulate different profiles randomly
    const profiles = [
        { restecg: "normal", thalch: 165, exang: "false", oldpeak: 0.2 },
        { restecg: "lv hypertrophy", thalch: 110, exang: "true", oldpeak: 2.5 },
        { restecg: "st-t abnormality", thalch: 95, exang: "true", oldpeak: 3.2 },
        { restecg: "normal", thalch: 175, exang: "false", oldpeak: 0.0 },
        { restecg: "lv hypertrophy", thalch: 130, exang: "false", oldpeak: 1.4 },
    ];

    const profile = profiles[Math.floor(Math.random() * profiles.length)];

    // Fill form fields with simulated values
    document.getElementById("restecg").value = profile.restecg;
    document.getElementById("thalch").value = profile.thalch;
    document.getElementById("exang").value = profile.exang;
    document.getElementById("oldpeak").value = profile.oldpeak;

    // Visual feedback
    simulateIoTBtn.innerHTML = '<span>✅ ECG Data Loaded!</span>';
    simulateIoTBtn.style.borderColor = 'rgba(34, 197, 94, 0.5)';
    simulateIoTBtn.style.color = '#22c55e';

    setTimeout(() => {
        simulateIoTBtn.innerHTML = '<span>Simulate ECG Data</span>';
        simulateIoTBtn.style.borderColor = '';
        simulateIoTBtn.style.color = '';
    }, 2000);

    // Highlight changed fields
    ["restecg", "thalch", "exang", "oldpeak"].forEach((id) => {
        const el = document.getElementById(id);
        el.style.borderColor = "rgba(139, 92, 246, 0.6)";
        el.style.boxShadow = "0 0 0 3px rgba(139, 92, 246, 0.15)";
        setTimeout(() => {
            el.style.borderColor = "";
            el.style.boxShadow = "";
        }, 2500);
    });
});

// ── Results rendering ──
function renderResults(result) {
    // Show results section
    resultsSection.style.display = "block";

    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 100);

    // 1. Risk gauge
    drawGauge(result.risk_score);
    gaugeValue.textContent = `${Math.round(result.risk_score * 100)}%`;

    // Color the gauge value
    const riskLevel = result.risk_level.toLowerCase();
    if (riskLevel === "low") {
        gaugeValue.style.color = "#22c55e";
    } else if (riskLevel === "medium") {
        gaugeValue.style.color = "#f97316";
    } else {
        gaugeValue.style.color = "#ef4444";
    }

    // 2. Risk badge
    riskBadge.textContent = `${result.risk_level} Risk`;
    riskBadge.className = `risk-badge ${riskLevel}`;

    // 3. Explanations
    explanationList.innerHTML = "";
    result.explanation.forEach((exp) => {
        const li = document.createElement("li");
        const isIncrease = exp.toLowerCase().includes("increased");
        li.className = isIncrease ? "risk-increase" : "risk-decrease";
        li.innerHTML = `
            <span class="exp-icon">${isIncrease ? "🔴" : "🟢"}</span>
            <span>${exp}</span>
        `;
        explanationList.appendChild(li);
    });

    // 4. Feature contributions chart
    renderContributions(result.feature_contributions);

    // 5. Recommendations
    recommendationsList.innerHTML = "";
    result.recommendations.forEach((rec, i) => {
        const li = document.createElement("li");
        li.style.animationDelay = `${i * 0.07}s`;
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });
}

// ── Contribution bar chart ──
function renderContributions(contributions) {
    contributionsChart.innerHTML = "";

    // Sort by absolute value
    const sorted = Object.entries(contributions)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
        .slice(0, 10); // Top 10

    const maxAbsVal = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.01);

    sorted.forEach(([feature, value], index) => {
        const row = document.createElement("div");
        row.className = "contrib-row";
        row.style.animationDelay = `${index * 0.06}s`;

        const barWidth = (Math.abs(value) / maxAbsVal) * 48; // max 48% of container
        const isPositive = value > 0;

        row.innerHTML = `
            <span class="contrib-label" title="${feature}">${feature}</span>
            <div class="contrib-bar-container">
                <div class="contrib-bar ${isPositive ? 'positive' : 'negative'}" 
                     style="width: 0%"></div>
            </div>
            <span class="contrib-value">${value > 0 ? '+' : ''}${value.toFixed(3)}</span>
        `;

        contributionsChart.appendChild(row);

        // Animate bar width
        requestAnimationFrame(() => {
            setTimeout(() => {
                const bar = row.querySelector(".contrib-bar");
                bar.style.width = `${barWidth}%`;
            }, 100 + index * 60);
        });
    });
}

// ── Risk gauge drawing (canvas) ──
function drawGauge(score) {
    const canvas = document.getElementById("riskGauge");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size for high DPI
    canvas.width = 260 * dpr;
    canvas.height = 160 * dpr;
    canvas.style.width = "260px";
    canvas.style.height = "160px";
    ctx.scale(dpr, dpr);

    const centerX = 130;
    const centerY = 140;
    const radius = 110;
    const lineWidth = 16;

    // Clear
    ctx.clearRect(0, 0, 260, 160);

    // Background arc
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();

    // Gradient for the gauge
    const gradient = ctx.createLinearGradient(20, 0, 240, 0);
    gradient.addColorStop(0, "#22c55e");
    gradient.addColorStop(0.35, "#eab308");
    gradient.addColorStop(0.65, "#f97316");
    gradient.addColorStop(1, "#ef4444");

    // Animated fill
    animateGauge(ctx, centerX, centerY, radius, lineWidth, gradient, score);
}

function animateGauge(ctx, cx, cy, r, lw, gradient, targetScore) {
    let current = 0;
    const step = targetScore / 60; // 60 frames

    function draw() {
        // Clear only the arc area
        ctx.clearRect(0, 0, 260, 160);

        // Background arc
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
        ctx.lineWidth = lw;
        ctx.lineCap = "round";
        ctx.stroke();

        // Filled arc
        const endAngle = Math.PI + current * Math.PI;
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.PI, endAngle);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = lw;
        ctx.lineCap = "round";
        ctx.stroke();

        // Glow effect
        ctx.beginPath();
        ctx.arc(cx, cy, r, Math.PI, endAngle);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = lw + 6;
        ctx.lineCap = "round";
        ctx.globalAlpha = 0.15;
        ctx.stroke();
        ctx.globalAlpha = 1;

        // Needle dot
        const needleAngle = Math.PI + current * Math.PI;
        const dotX = cx + r * Math.cos(needleAngle);
        const dotY = cy + r * Math.sin(needleAngle);
        ctx.beginPath();
        ctx.arc(dotX, dotY, 6, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "rgba(255, 255, 255, 0.5)";
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;

        // Update gauge text
        gaugeValue.textContent = `${Math.round(current * 100)}%`;

        current += step;
        if (current < targetScore) {
            requestAnimationFrame(draw);
        } else {
            // Final frame
            current = targetScore;
            gaugeValue.textContent = `${Math.round(targetScore * 100)}%`;
        }
    }

    draw();
}

// ── Helpers ──
function setLoading(loading) {
    if (loading) {
        btnText.style.display = "none";
        btnLoader.style.display = "inline-flex";
        submitBtn.disabled = true;
    } else {
        btnText.style.display = "inline";
        btnLoader.style.display = "none";
        submitBtn.disabled = false;
    }
}

function showError(message) {
    // Create toast notification
    const toast = document.createElement("div");
    toast.style.cssText = `
        position: fixed;
        top: 24px;
        right: 24px;
        z-index: 9999;
        padding: 16px 24px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        color: #fca5a5;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        max-width: 400px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        animation: fadeInUp 0.3s ease-out;
    `;
    toast.textContent = `⚠️ ${message}`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(-10px)";
        toast.style.transition = "all 0.3s ease";
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

// ── Initialize particles (subtle floating dots) ──
function initParticles() {
    const container = document.getElementById("bgParticles");
    for (let i = 0; i < 20; i++) {
        const dot = document.createElement("div");
        const size = Math.random() * 3 + 1;
        dot.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: rgba(255, 255, 255, ${Math.random() * 0.08 + 0.02});
            border-radius: 50%;
            left: ${Math.random() * 100}%;
            top: ${Math.random() * 100}%;
            animation: particleFloat ${Math.random() * 20 + 15}s ease-in-out infinite;
            animation-delay: ${Math.random() * -20}s;
        `;
        container.appendChild(dot);
    }

    // Add particle animation
    const style = document.createElement("style");
    style.textContent = `
        @keyframes particleFloat {
            0%, 100% { transform: translate(0, 0) scale(1); opacity: 0.3; }
            25% { transform: translate(${Math.random() * 60 - 30}px, ${Math.random() * 60 - 30}px) scale(1.5); opacity: 0.6; }
            50% { transform: translate(${Math.random() * 80 - 40}px, ${Math.random() * 40 - 20}px) scale(0.8); opacity: 0.2; }
            75% { transform: translate(${Math.random() * 50 - 25}px, ${Math.random() * 70 - 35}px) scale(1.2); opacity: 0.5; }
        }
    `;
    document.head.appendChild(style);
}

// Initialize on load
document.addEventListener("DOMContentLoaded", initParticles);
