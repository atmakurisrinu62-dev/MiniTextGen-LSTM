import { useState } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [generatedText, setGeneratedText] = useState("");
  const [loading, setLoading] = useState(false);

  const [maxWords, setMaxWords] = useState(50);
  const [temperature, setTemperature] = useState(0.8);

  const generateText = async () => {
    if (!prompt.trim()) return;

    try {
      setLoading(true);
      setGeneratedText("");

      const response = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          prompt: prompt,
          max_words: maxWords,
          temperature: temperature,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate text");
      }

      const data = await response.json();

      setGeneratedText(data.generated_text);
    } catch (error) {
      console.error(error);

      setGeneratedText(
        "Something went wrong while generating text. Please check the backend server.",
      );
    } finally {
      setLoading(false);
    }
  };

  const clearText = () => {
    setPrompt("");
    setGeneratedText("");
  };

  return (
    <div className="app">
      {/* Background Effects */}
      <div className="background-orb orb-one"></div>
      <div className="background-orb orb-two"></div>
      <div className="background-orb orb-three"></div>

      <div className="container">
        {/* Header */}
        <div className="header">
          <div className="badge">PyTorch + LSTM</div>

          <h1>
            MiniTextGen
            <span> LSTM</span>
          </h1>

          <p className="subtitle">
            Generate Shakespeare-style text using a custom LSTM model trained
            with PyTorch.
          </p>
        </div>

        {/* Prompt Section */}
        <div className="section">
          <label className="section-label">Enter your prompt</label>

          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Example: I love..."
          />

          <div className="prompt-footer">
            <span>{prompt.length} characters</span>

            <button className="clear-button" onClick={clearText} type="button">
              Clear
            </button>
          </div>
        </div>

        {/* Controls */}
        <div className="controls">
          <div className="control-card">
            <div className="control-header">
              <label>Max Words</label>

              <span>{maxWords}</span>
            </div>

            <input
              type="range"
              min="10"
              max="150"
              step="10"
              value={maxWords}
              onChange={(e) => setMaxWords(Number(e.target.value))}
            />

            <div className="range-labels">
              <span>10</span>
              <span>150</span>
            </div>
          </div>

          <div className="control-card">
            <div className="control-header">
              <label>Temperature</label>

              <span>{temperature}</span>
            </div>

            <input
              type="range"
              min="0.3"
              max="1.5"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />

            <div className="range-labels">
              <span>Focused</span>
              <span>Creative</span>
            </div>
          </div>
        </div>

        {/* Generate Button */}
        <button
          className="generate-button"
          onClick={generateText}
          disabled={loading}
        >
          {loading ? (
            <>
              <span className="spinner"></span>
              Generating...
            </>
          ) : (
            <>
              <span className="button-icon">✦</span>
              Generate Text
            </>
          )}
        </button>

        {/* Output */}
        {generatedText && (
          <div className="output-card">
            <div className="output-header">
              <div>
                <span className="output-badge">AI Generated</span>

                <h2>Generated Text</h2>
              </div>
            </div>

            <div className="generated-content">{generatedText}</div>
          </div>
        )}

        <div className="footer">
          Built with React • FastAPI • PyTorch • LSTM
        </div>
      </div>
    </div>
  );
}

export default App;
