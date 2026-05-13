import { useState, useRef, useCallback } from 'react'
import './App.css'

const API_URL = 'http://localhost:5000'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFile = useCallback((file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Por favor selecciona un archivo de imagen válido.')
      return
    }
    setSelectedFile(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }, [])

  const handleFileChange = (e) => {
    handleFile(e.target.files[0])
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Primero selecciona una imagen.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('image', selectedFile)

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.error || 'Error en la predicción')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || 'Error conectando con el servidor. ¿Está el servidor ejecutándose?')
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const getBarWidth = (prob) => `${(prob * 100).toFixed(1)}%`
  const getBarColor = (digit, predictedDigit) => {
    return digit === predictedDigit ? 'var(--accent)' : 'var(--bar-bg)'
  }

  return (
    <div className="app">
      {/* Fondo animado */}
      <div className="bg-grid">
        {Array.from({ length: 50 }).map((_, i) => (
          <span key={i} className="grid-dot" style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            animationDelay: `${Math.random() * 5}s`,
            animationDuration: `${3 + Math.random() * 4}s`,
          }} />
        ))}
      </div>

      <header className="header">
        <div className="header-icon">🧠</div>
        <h1>Reconocimiento de Dígitos</h1>
        <p className="subtitle">CNN &middot; Red Neuronal Convolucional</p>
      </header>

      <main className="main-content">
        {/* Panel de carga */}
        <section className="upload-section glass-card">
          <h2>📷 Sube una imagen</h2>
          <p className="hint">
            Sube una imagen que contenga un dígito escrito a mano (0-9). 
            El modelo CNN lo analizará y predecirá qué número es.
          </p>

          <div
            className={`dropzone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {preview ? (
              <img src={preview} alt="Preview" className="preview-image" />
            ) : (
              <div className="dropzone-placeholder">
                <span className="upload-icon">⬆️</span>
                <span>Arrastra una imagen aquí o haz clic para seleccionar</span>
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="file-input-hidden"
          />

          <div className="actions">
            <button
              className="btn btn-primary"
              onClick={handlePredict}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <>
                  <span className="spinner" />
                  Analizando...
                </>
              ) : (
                '🔍 Analizar imagen'
              )}
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              🗑️ Limpiar
            </button>
          </div>

          {error && (
            <div className="error-message">
              ⚠️ {error}
            </div>
          )}
        </section>

        {/* Panel de resultados */}
        {result && (
          <section className="result-section glass-card fade-in">
            <h2>📊 Resultado</h2>

            <div className="predicted-digit-container">
              <div className="predicted-digit-display">
                <span className="digit-value">{result.digit}</span>
              </div>
              <div className="predicted-info">
                <span className="predicted-label">Dígito predicho</span>
                <span className="confidence-badge">
                  {result.confidence}% confianza
                </span>
              </div>
            </div>

            <h3>Probabilidades por clase</h3>
            <div className="probabilities">
              {Object.entries(result.probabilities)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([digit, prob]) => (
                  <div key={digit} className={`prob-row ${Number(digit) === result.digit ? 'highlight' : ''}`}>
                    <span className="prob-digit">{digit}</span>
                    <div className="prob-bar-track">
                      <div
                        className="prob-bar-fill"
                        style={{
                          width: getBarWidth(prob),
                          backgroundColor: getBarColor(Number(digit), result.digit),
                        }}
                      />
                    </div>
                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>P02 &middot; Modelo CNN &middot; MNIST &middot; TensorFlow / Keras</p>
      </footer>
    </div>
  )
}

export default App
