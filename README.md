# User Story Pro: Video to User Story with Gemini Vision

User Story Pro es una aplicación desarrollada en Streamlit que permite transformar videos de flujos de aplicaciones en historias de usuario profesionales, siguiendo los más altos estándares de agilidad, UX y detalle técnico. Utiliza la API de Google Gemini Vision para analizar visualmente los videos y generar historias de usuario completas, con criterios de aceptación, contexto funcional, requisitos técnicos y detalles de UI.

## Características

- **Sube un video** de tu app (flujos, pantallas, botones, etc.).
- **Extracción automática de frames** relevantes del video.
- **Análisis visual con IA** usando modelos de Google Gemini Vision.
- **Generación automática de historias de usuario** siguiendo el formato ágil clásico, criterios de aceptación (Gherkin), requisitos técnicos y de negocio, y detalles de UI.
- **Prompt editable** para personalizar el enfoque de la IA.
- **Selección de modelo Gemini** desde la interfaz.
- **Interfaz simple y amigable** en Streamlit.

## ¿Para qué sirve?

Ideal para equipos de producto, UX, QA y desarrollo que quieran documentar funcionalidades a partir de prototipos, demos o grabaciones de apps, asegurando historias de usuario claras, completas y listas para desarrollo y testing.

## ¿Cómo usar?

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/luciofondon98/user-stories-ai.git
   cd user-stories-ai
   ```

2. **Crea y activa un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # En Windows
   ```

3. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura tu API Key de Google Gemini:**
   - Crea un archivo `.env` en la raíz con el contenido:
     ```
     GEMINI_API_KEY=tu_api_key_de_gemini
     ```

5. **Ejecuta la app:**
   ```bash
   streamlit run src/main.py
   ```

6. **Carga un video** y sigue las instrucciones en pantalla.

## Estructura del proyecto

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── .env (no incluido, debes crearlo)
└── src/
    └── main.py
```

## Ejemplo de uso

1. Sube un video de tu app mostrando un flujo o funcionalidad.
2. Ajusta el prompt si lo deseas.
3. Selecciona el modelo de Gemini Vision.
4. Haz clic en "Analizar video y generar historia de usuario".
5. Obtén una historia de usuario profesional, lista para tu equipo.
