import streamlit as st
import time
from config import PROJECT_NAME

def show_loading_screen():
    """Muestra la pantalla de carga inicial si no se ha cargado la app."""
    if "app_loaded" not in st.session_state:
        st.session_state["app_loaded"] = False

    loading_placeholder = None
    if not st.session_state["app_loaded"]:
        loading_placeholder = st.empty()
        with loading_placeholder:
            st.markdown(
                """
                <link rel="preconnect" href="https://fonts.googleapis.com">
                <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
                <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
                
                <div class="loading-overlay">
                    <div class="glow-bg-1"></div>
                    <div class="glow-bg-2"></div>
                    <div class="loading-card">
                        <div class="app-icon">🌫️</div>
                        <h1 class="loading-title">SisDin</h1>
                        <h2 class="loading-subtitle">Simulador de Calidad del Aire - UANL</h2>
                        <div class="spinner-container">
                            <div class="glow-spinner"></div>
                        </div>
                        <p class="loading-status">Inicializando modelos atmosféricos y red vial...</p>
                        <div class="progress-bar-container">
                            <div class="progress-bar-fill"></div>
                        </div>
                    </div>
                </div>
                
                <style>
                .loading-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    background-color: #0b0b0f;
                    z-index: 9999999;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    overflow: hidden;
                    font-family: 'Outfit', sans-serif;
                }
                
                .glow-bg-1 {
                    position: absolute;
                    width: 400px;
                    height: 400px;
                    background: radial-gradient(circle, rgba(0, 184, 148, 0.15) 0%, rgba(0,0,0,0) 70%);
                    top: 20%;
                    left: 20%;
                    filter: blur(50px);
                    animation: floatGlow1 8s ease-in-out infinite alternate;
                }
                
                .glow-bg-2 {
                    position: absolute;
                    width: 450px;
                    height: 450px;
                    background: radial-gradient(circle, rgba(108, 92, 231, 0.15) 0%, rgba(0,0,0,0) 70%);
                    bottom: 20%;
                    right: 20%;
                    filter: blur(50px);
                    animation: floatGlow2 10s ease-in-out infinite alternate;
                }
                
                @keyframes floatGlow1 {
                    0% { transform: translate(0, 0) scale(1); }
                    100% { transform: translate(50px, 30px) scale(1.1); }
                }
                
                @keyframes floatGlow2 {
                    0% { transform: translate(0, 0) scale(1.1); }
                    100% { transform: translate(-40px, -40px) scale(0.9); }
                }
                
                .loading-card {
                    position: relative;
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.07);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border-radius: 24px;
                    padding: 40px 50px;
                    width: 450px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
                    z-index: 10;
                    text-align: center;
                }
                
                .app-icon {
                    font-size: 3rem;
                    margin-bottom: 15px;
                    animation: floatIcon 3s ease-in-out infinite;
                }
                
                @keyframes floatIcon {
                    0%, 100% { transform: translateY(0); }
                    50% { transform: translateY(-10px); }
                }
                
                .loading-title {
                    color: #ffffff;
                    font-size: 2.5rem;
                    font-weight: 800;
                    margin: 0;
                    letter-spacing: 2px;
                    background: linear-gradient(135deg, #ffffff 0%, #a8a8b3 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }
                
                .loading-subtitle {
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 0.95rem;
                    font-weight: 400;
                    margin-top: 8px;
                    margin-bottom: 30px;
                    letter-spacing: 0.5px;
                }
                
                .spinner-container {
                    position: relative;
                    width: 70px;
                    height: 70px;
                    margin-bottom: 25px;
                }
                
                .glow-spinner {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    border: 3px solid rgba(255, 255, 255, 0.05);
                    border-radius: 50%;
                    border-top: 3px solid #00b894;
                    border-right: 3px solid #00b894;
                    animation: rotate 1.2s cubic-bezier(0.5, 0.1, 0.4, 0.9) infinite;
                    box-shadow: 0 0 15px rgba(0, 184, 148, 0.2);
                }
                
                @keyframes rotate {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .loading-status {
                    color: rgba(255, 255, 255, 0.85);
                    font-size: 0.9rem;
                    font-weight: 600;
                    margin-bottom: 15px;
                    letter-spacing: 0.5px;
                    animation: pulseText 1.5s ease-in-out infinite;
                }
                
                @keyframes pulseText {
                    0%, 100% { opacity: 0.6; }
                    50% { opacity: 1; }
                }
                
                .progress-bar-container {
                    width: 100%;
                    height: 4px;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 2px;
                    overflow: hidden;
                }
                
                .progress-bar-fill {
                    height: 100%;
                    width: 30%;
                    background: linear-gradient(90deg, #00b894, #00cec9);
                    border-radius: 2px;
                    animation: progressMove 2.5s infinite ease-in-out;
                }
                
                @keyframes progressMove {
                    0% {
                        width: 10%;
                        transform: translateX(-100%);
                    }
                    50% {
                        width: 40%;
                    }
                    100% {
                        width: 10%;
                        transform: translateX(1000%);
                    }
                }
                </style>
                """.replace("SisDin", PROJECT_NAME),
                unsafe_allow_html=True,
            )
        # Pequeño delay para asegurar que Streamlit envíe y renderice el HTML del loading screen
        # en el navegador del cliente antes de iniciar la importación pesada de librerías.
        time.sleep(0.1)
    return loading_placeholder

def remove_loading_screen(loading_placeholder):
    """Remueve la pantalla de carga e inicializa la aplicación."""
    if not st.session_state.get("app_loaded", False) and loading_placeholder is not None:
        time.sleep(0.5)  # Breve pausa para suavidad visual
        loading_placeholder.empty()
        st.session_state["app_loaded"] = True
