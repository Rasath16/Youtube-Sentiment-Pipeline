
document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = "AIzaSyAu7vHREXrcIG3UYOb2ySP6fW6m6ya6uv0"; 
  const API_URL = "http://ec2-54-211-18-166.compute-1.amazonaws.com:8080";

  // Sentiment label mapping (YouTube format: -1, 0, 1)
  const SENTIMENT_LABELS = {
    "-1": { name: "Negative", emoji: "😞", color: "#FF6B6B" },
    0: { name: "Neutral", emoji: "😐", color: "#95A5A6" },
    1: { name: "Positive", emoji: "😊", color: "#4ECDC4" },
  };

  // Show loading screen
  showLoadingScreen();

  // Check API health and get model info
  const modelInfo = await checkAPIHealth();
  if (!modelInfo) {
    showError("Unable to connect to sentiment analysis API.");
    return;
  }

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex =
      /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];

      updateStatus(`📹 Found Video ID: ${videoId}`);
      updateStatus(`💬 Fetching comments from YouTube...`);

      const comments = await fetchComments(videoId);

      if (comments.length === 0) {
        showError(
          "No comments found for this video. The video might have comments disabled."
        );
        return;
      }

      updateStatus(`✅ Fetched ${comments.length} comments`);
      updateStatus(
        `🤖 Analyzing sentiment with ${modelInfo.model_type} model...`
      );

      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        outputDiv.innerHTML = "";

        // Process the predictions
        const sentimentCounts = { 1: 0, 0: 0, "-1": 0 };
        const sentimentData = [];
        let totalSentimentScore = 0;
        let totalConfidence = 0;

        predictions.forEach((item) => {
          const sentiment = String(item.sentiment);
          sentimentCounts[sentiment]++;
          totalSentimentScore += parseInt(item.sentiment);
          totalConfidence += item.confidence || 0;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment),
          });
        });

        // Compute metrics
        const totalComments = comments.length;
        const uniqueCommenters = new Set(
          comments.map((comment) => comment.authorId)
        ).size;
        const totalWords = comments.reduce(
          (sum, comment) =>
            sum +
            comment.text.split(/\s+/).filter((word) => word.length > 0).length,
          0
        );
        const avgWordLength = (totalWords / totalComments).toFixed(1);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(
          2
        );
        const avgConfidence = ((totalConfidence / totalComments) * 100).toFixed(
          1
        );

        // Normalize to 0-10 scale
        const normalizedSentimentScore = (
          ((parseFloat(avgSentimentScore) + 1) / 2) *
          10
        ).toFixed(1);

        // Calculate percentages
        const positivePercent = (
          (sentimentCounts["1"] / totalComments) *
          100
        ).toFixed(1);
        const neutralPercent = (
          (sentimentCounts["0"] / totalComments) *
          100
        ).toFixed(1);
        const negativePercent = (
          (sentimentCounts["-1"] / totalComments) *
          100
        ).toFixed(1);

        // Display results
        displayOverallScore(
          normalizedSentimentScore,
          avgSentimentScore,
          avgConfidence
        );
        displaySentimentBreakdown(
          sentimentCounts,
          positivePercent,
          neutralPercent,
          negativePercent
        );
        displayMetrics(totalComments, uniqueCommenters, avgWordLength);

        // Display visualizations
        await displayChartSection(sentimentCounts);
        await displayTrendGraphSection(sentimentData);
        await displayWordCloudSection(comments.map((c) => c.text));

        // Display top comments
        displayTopComments(predictions);

        // Add footer with model info
        displayFooter(modelInfo);
      }
    } else {
      showError("Please open a YouTube video page to analyze comments.");
    }
  });

  // Check API health and get model info
  async function checkAPIHealth() {
    try {
      const response = await fetch(`${API_URL}/`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });

      if (response.ok) {
        const data = await response.json();
        console.log("API Health:", data);

        if (data.model_loaded && data.vectorizer_loaded) {
          return {
            model_type: data.model_type || "LightGBM",
            model_version: data.model_version || "Unknown",
            accuracy: data.model_accuracy || "Unknown",
          };
        }
      }
      return null;
    } catch (error) {
      console.error("API health check failed:", error);
      return null;
    }
  }

  // Fetch comments from YouTube
  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";

    try {
      while (comments.length < 1000) {
        let url = `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&key=${API_KEY}`;
        if (pageToken) {
          url += `&pageToken=${pageToken}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
          console.error("YouTube API Error:", data);
          showError(`YouTube API Error: ${data.error.message}`);
          return [];
        }

        if (data.items) {
          data.items.forEach((item) => {
            const commentText =
              item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId =
              item.snippet.topLevelComment.snippet.authorChannelId?.value ||
              "Unknown";
            comments.push({
              text: commentText,
              timestamp: timestamp,
              authorId: authorId,
            });
          });
        }

        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      showError(
        "Error fetching comments from YouTube. Please check your API key."
      );
    }

    return comments;
  }

  // Get sentiment predictions from Flask API
  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments }),
      });

      const result = await response.json();

      if (response.ok) {
        return result;
      } else {
        throw new Error(result.error || "Error fetching predictions");
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      showError("Error getting sentiment predictions from API.");
      return null;
    }
  }

  // Display functions
  function showLoadingScreen() {
    outputDiv.innerHTML = `
      <div class="loading-container">
        <div class="spinner"></div>
        <h3>Initializing Sentiment Analysis...</h3>
        <p class="loading-text">Powered by LightGBM + ADASYN from MLflow</p>
      </div>`;
  }

  function updateStatus(message) {
    const loadingText = outputDiv.querySelector(".loading-text");
    if (loadingText) {
      loadingText.textContent = message;
    }
  }

  function showError(message) {
    outputDiv.innerHTML = `
      <div class="error-container">
        <div class="error-icon">⚠️</div>
        <h3>Error</h3>
        <p>${message}</p>
      </div>`;
  }

  function displayOverallScore(normalizedScore, rawScore, avgConfidence) {
    const sentiment =
      rawScore >= 0.3 ? "positive" : rawScore <= -0.3 ? "negative" : "neutral";
    const emoji = rawScore >= 0.3 ? "😊" : rawScore <= -0.3 ? "😞" : "😐";
    const color =
      rawScore >= 0.3 ? "#4ECDC4" : rawScore <= -0.3 ? "#FF6B6B" : "#95A5A6";

    outputDiv.innerHTML += `
      <div class="overall-score" style="border-color: ${color}">
        <div class="score-emoji">${emoji}</div>
        <div class="score-value" style="color: ${color}">${normalizedScore}</div>
        <div class="score-label">Overall Sentiment Score</div>
        <div class="score-subtitle">out of 10 • ${avgConfidence}% confidence</div>
      </div>`;
  }

  function displaySentimentBreakdown(
    counts,
    positivePercent,
    neutralPercent,
    negativePercent
  ) {
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">📊 Sentiment Distribution</div>
        <div class="sentiment-bars">
          <div class="sentiment-bar">
            <div class="bar-label">
              <span>😊 Positive</span>
              <span class="bar-percentage">${positivePercent}%</span>
            </div>
            <div class="bar-background">
              <div class="bar-fill positive" style="width: ${positivePercent}%"></div>
            </div>
            <div class="bar-count">${counts["1"]} comments</div>
          </div>
          
          <div class="sentiment-bar">
            <div class="bar-label">
              <span>😐 Neutral</span>
              <span class="bar-percentage">${neutralPercent}%</span>
            </div>
            <div class="bar-background">
              <div class="bar-fill neutral" style="width: ${neutralPercent}%"></div>
            </div>
            <div class="bar-count">${counts["0"]} comments</div>
          </div>
          
          <div class="sentiment-bar">
            <div class="bar-label">
              <span>😞 Negative</span>
              <span class="bar-percentage">${negativePercent}%</span>
            </div>
            <div class="bar-background">
              <div class="bar-fill negative" style="width: ${negativePercent}%"></div>
            </div>
            <div class="bar-count">${counts["-1"]} comments</div>
          </div>
        </div>
      </div>`;
  }

  function displayMetrics(totalComments, uniqueCommenters, avgWordLength) {
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">📈 Comment Statistics</div>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-icon">💬</div>
            <div class="metric-value">${totalComments}</div>
            <div class="metric-label">Total Comments</div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">👥</div>
            <div class="metric-value">${uniqueCommenters}</div>
            <div class="metric-label">Unique Users</div>
          </div>
          <div class="metric-card">
            <div class="metric-icon">📝</div>
            <div class="metric-value">${avgWordLength}</div>
            <div class="metric-label">Avg Words/Comment</div>
          </div>
        </div>
      </div>`;
  }

  async function displayChartSection(sentimentCounts) {
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">🥧 Sentiment Pie Chart</div>
        <div id="chart-container" class="chart-loading">Loading chart...</div>
      </div>`;

    await fetchAndDisplayChart(sentimentCounts);
  }

  async function displayTrendGraphSection(sentimentData) {
    if (sentimentData.length > 10) {
      outputDiv.innerHTML += `
        <div class="section">
          <div class="section-title">📉 Sentiment Trend Over Time</div>
          <div id="trend-graph-container" class="chart-loading">Loading trend...</div>
        </div>`;

      await fetchAndDisplayTrendGraph(sentimentData);
    }
  }

  async function displayWordCloudSection(comments) {
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">☁️ Word Cloud</div>
        <div id="wordcloud-container" class="chart-loading">Generating word cloud...</div>
      </div>`;

    await fetchAndDisplayWordCloud(comments);
  }

  function displayTopComments(predictions) {
    outputDiv.innerHTML += `
      <div class="section">
        <div class="section-title">💭 Top 15 Comments</div>
        <div class="comments-container">
          ${predictions
            .slice(0, 15)
            .map((item, index) => {
              const sentiment = String(item.sentiment);
              const sentimentInfo = SENTIMENT_LABELS[sentiment];
              const confidence = item.confidence
                ? (item.confidence * 100).toFixed(0)
                : "N/A";
              return `
              <div class="comment-card">
                <div class="comment-header">
                  <span class="comment-number">#${index + 1}</span>
                  <span class="comment-badge" style="background-color: ${
                    sentimentInfo.color
                  }">
                    ${sentimentInfo.emoji} ${sentimentInfo.name} ${
                confidence !== "N/A" ? `(${confidence}%)` : ""
              }
                  </span>
                </div>
                <div class="comment-text">${escapeHtml(item.comment)}</div>
              </div>`;
            })
            .join("")}
        </div>
      </div>`;
  }

  function displayFooter(modelInfo) {
    outputDiv.innerHTML += `
      <div class="footer">
      <div class="footer-title">🤖 Powered by ${
        modelInfo.model_type
      } Model</div>
      <div class="footer-text">Model: final_lightgbm_adasyn_model</div>
      <div class="footer-text">Version: ${
        modelInfo.model_version || "N/A"
      } • Stage: Staging</div>
      <div class="footer-text">Accuracy: ${
        modelInfo.test_metrics?.accuracy ?? "Unknown"
      }</div>
      <div class="footer-text">Loaded from MLflow Registry</div>
    </div>`;
  }

  // Fetch and display visualization images
  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts }),
      });

      if (!response.ok) throw new Error("Failed to fetch chart");

      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);

      const chartContainer = document.getElementById("chart-container");
      chartContainer.innerHTML = `<img src="${imgURL}" alt="Sentiment Chart" class="viz-image">`;
    } catch (error) {
      console.error("Error fetching chart:", error);
      document.getElementById("chart-container").innerHTML =
        '<p class="error-text">Failed to load chart</p>';
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments }),
      });

      if (!response.ok) throw new Error("Failed to fetch word cloud");

      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);

      const wordcloudContainer = document.getElementById("wordcloud-container");
      wordcloudContainer.innerHTML = `<img src="${imgURL}" alt="Word Cloud" class="viz-image">`;
    } catch (error) {
      console.error("Error fetching word cloud:", error);
      document.getElementById("wordcloud-container").innerHTML =
        '<p class="error-text">Failed to load word cloud</p>';
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData }),
      });

      if (!response.ok) throw new Error("Failed to fetch trend graph");

      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);

      const trendGraphContainer = document.getElementById(
        "trend-graph-container"
      );
      trendGraphContainer.innerHTML = `<img src="${imgURL}" alt="Sentiment Trend" class="viz-image">`;
    } catch (error) {
      console.error("Error fetching trend graph:", error);
      document.getElementById("trend-graph-container").innerHTML =
        '<p class="error-text">Failed to load trend graph</p>';
    }
  }

  // Utility function
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
});
