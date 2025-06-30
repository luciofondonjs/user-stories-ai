import streamlit as st
import os
import tempfile
import cv2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import json
import glob
from datetime import datetime

FRAME_INTERVAL = 2  # segundos entre frames extraídos
MAX_FRAMES = 8      # máximo de frames a analizar para no exceder el límite de la API

# Cargar la API KEY de Gemini desde .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="User Story Pro", layout="centered")
st.title("User Story Pro: Video to User Story with Gemini Vision")

st.write("""
Sube un video de tu app (puede contener flujos, botones, pantallas, etc.). La IA analizará visualmente el video y generará una historia de usuario profesional.
""")

# Prompt editable para la generación de historias de usuario
def default_prompt():
    return """Role:
You are an expert in writing high-quality, agile user stories for digital products, with deep knowledge of best practices, user-centric design, and clear, testable acceptance criteria.

Task:
You will receive a visual analysis of an app feature (screens/flows). Rewrite it into a professional, detailed user story using the structure and standards below, ensuring clarity, completeness, testability, and UI fidelity.

Guidelines:
- Use the classic user story format:
  As a [type of user]
  I want [goal or need]
  So that [benefit or business value]

- Add a Functional Context: Briefly describe how the feature fits in the broader user journey, system, or UI module.

- Acceptance Criteria:
  - Use clear, specific, and testable criteria.
  - Prefer Gherkin syntax (Given / When / Then) or structured bullet points.
  - Break down criteria by section, tab, or interaction step as needed.
  - Include UI behavior details: hover states, color changes, tooltips, visibility toggles, responsiveness, icons, micro-interactions.
  - For each tab/section, list all relevant behaviors and requirements.

- Technical and Business Requirements:
  - Field validations (min/max characters, format, required/optional)
  - Conditional logic (e.g., show X only when Y is selected)
  - Accessibility needs (ARIA labels, keyboard navigation, contrast)
  - Analytics/tracking requirements
  - Design consistency (UI tokens, reusable components)

- User-Centric Focus:
  - Focus on user goals, clarity, autonomy, and accessibility.
  - Include messaging for error/success/empty states.
  - Avoid technical jargon unless necessary.

- INVEST Principle: All stories must be Independent, Negotiable, Valuable, Estimable, Small, Testable.

- UI Detailing Requirement (Front-End Fidelity):
  - Always include detailed UI expectations, even for minor design changes.
  - Cover color scheme and states, font styling, tooltip text and behavior, animations, responsive layout, placement and spacing.

- Follow the JetSMART reference story structure and level of detail.

Reference Example:
As a JetSMART user

I want to clearly view the summary of my purchase, my personal details, the flight information, and have direct access to actions such as check-in, name change, change or refund requests,
So that I can manage my booking independently without relying on support, with all information clear and accessible.

Functional Context
Once the user completes their purchase, they are automatically redirected (or can access through their email or "My Trips") to the itinerary page:
booking.staging.jetsmart.com/V2/Itinerary

The page has 5 tabs:
Your Itinerary
Booking Details
Transactions
Modify Your Booking
Passengers

🎯 Acceptance Criteria by Section/Tab
General Header View
Displays the text "Manage Your Trip" with a calendar icon.
Shows the number of passengers.
Clearly displays the PNR (e.g., D7ICPW).
On the right, within a card-style box, it shows:
Flight route: Santiago (SCL) → Antofagasta (ANF)
Flight date and time: e.g., Sun 29/06, 01:00 - 03:10
Passenger count
Financial breakdown:
Subtotal: $60,900
Taxes and fees: $7,547
Total: $0 CLP (if paid with miles)

Tab: Your Itinerary
Shows outbound flight details (origin, destination, time, flight number).
Indicates whether the flight is direct or has stops/connections.
Options such as:
Resend itinerary email
Display of promotional banners to add:
Bags
Seats
Priority boarding
Highlighted section for the Check-in button (if available):
The button is shown only if there are less than 72h and more than 1h before the flight.
The button updates its state if check-in has already been completed.

Tab: Booking Details
Displays the booking breakdown with the following elements:
Passenger names
Transaction breakdown: base fare, ancillaries (if any), taxes and fees
Subtotals and total amount paid
Indicates if the booking was paid with miles (0 CLP)
Must include clear, readable icons with labels

Tab: Transactions
Technical breakdown of payments:
Payment type (e.g., miles, credit card, voucher)
Transaction status: approved, failed, refunded, etc.
Amount per item
Must include the authorization or reference code for each payment

Tab: Modify Your Booking
Includes 4 interactive actions:
a. "Transfer" button (name change)
Redirects to the "Passenger Information" section
Enables editing of the Passenger Name field
Validates that the flight is more than 2h away
If the change is within the valid window but close to flight time:
Show message: "This change has an additional cost."
Redirect to the payment flow before saving the change
Once saved, the name should update in the general view
b. "Request change" button
Redirects to: JetSMART – Fly SMART, Fly your way
c. "Request refund" button
Same link as "Request change" (unified flow): JetSMART – Fly SMART, Fly your way
d. "Emergency medical change" button
Redirects to the production environment: Changes and Refunds – Fly SMART, Fly your way

Tab: Passengers
Form to complete passenger contact details:
Email
Phone
Country
City
Address
Optional field: AAdvantage number
The "Save Information" button stores the contact data locally
For AAdvantage miles bookings, this step is mandatory to complete the process and enable check-in

🔧 Technical and Business Requirements
The Name field must be locked by default and only unlocked via "Transfer"
The system must verify:
Flight time vs current time
Check-in status
URLs must be dynamically built using PNR and passenger last name
Validations must comply with JetSMART and SSR standards
The design must remain responsive and accessible

Instructions:
Rewrite the visual analysis into a complete, professional user story, following all the above guidelines and the reference example. Be exhaustive and explicit in acceptance criteria and UI details."""

default_gemini_models = [
    "gemini-1.5-flash",
    "gemini-2.5-pro",
]

with st.expander("Mostrar/editar prompt avanzado para la IA", expanded=False):
    extra_context = st.text_area("Contexto extra para la IA (opcional, ej: 'El video muestra el flujo de registro de usuarios en una app de viajes')", value="", height=70)
    prompt = st.text_area("Prompt para la IA (puedes personalizarlo)", value=default_prompt(), height=180)

# Selector de modelo de Gemini Vision
selected_model = st.selectbox(
    "Selecciona el modelo de Gemini Vision a utilizar:",
    default_gemini_models,
    index=0,
    help="Puedes elegir entre los modelos disponibles de Google Gemini Vision."
)

st.sidebar.title("Modo de uso")
modo = st.sidebar.radio("Selecciona el modo de uso:", ["Subir video manualmente", "Procesar carpeta completa"], index=0)

# Información adicional en el sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Información útil")

if modo == "Subir video manualmente":
    st.sidebar.info("""
    **Modo manual:**
    - Sube un video individual
    - Se verifica automáticamente si ya existe HDU
    - Opción de cargar existente o reprocesar
    - Ideal para videos específicos
    """)
else:
    st.sidebar.info("""
    **Modo masivo:**
    - Procesa todos los videos de una carpeta
    - Evita reprocesar videos con HDU existente
    - Opción de forzar reprocesamiento
    - Ideal para lotes grandes
    """)

st.sidebar.markdown("### 💰 Ahorro de costos")
st.sidebar.success("""
✅ **Beneficios:**
- Evita reprocesar videos ya analizados
- Reduce costos de API de Gemini
- Ahorra tiempo de procesamiento
- Mantiene consistencia en análisis
""")

st.sidebar.markdown("### 📁 Archivos generados")
st.sidebar.info("""
Para cada video se crean:
- `video_HDU.txt` - Historia de usuario
- `video_HDU.json` - Metadatos del análisis
""")

# Función para verificar si ya existe HDU para un video
def verificar_hdu_existente(video_path):
    """Verifica si ya existen archivos HDU para el video dado"""
    base_path = os.path.splitext(video_path)[0]
    txt_path = base_path + "_HDU.txt"
    json_path = base_path + "_HDU.json"
    return os.path.exists(txt_path) and os.path.exists(json_path), txt_path, json_path

# Función para cargar HDU existente
def cargar_hdu_existente(video_path):
    """Carga una HDU existente desde los archivos guardados"""
    base_path = os.path.splitext(video_path)[0]
    txt_path = base_path + "_HDU.txt"
    json_path = base_path + "_HDU.json"
    
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            hdu_text = f.read()
        with open(json_path, "r", encoding="utf-8") as f:
            hdu_json = json.load(f)
        return hdu_text, hdu_json, None
    except Exception as e:
        return None, None, f"Error al cargar HDU existente: {str(e)}"

# Función para guardar HDU
def guardar_hdu(video_path, hdu_text, hdu_json):
    base_path = os.path.splitext(video_path)[0]
    txt_path = base_path + "_HDU.txt"
    json_path = base_path + "_HDU.json"
    # Avisar si existen
    avisos = []
    if os.path.exists(txt_path):
        avisos.append(f"⚠️ Se sobreescribirá: {txt_path}")
    if os.path.exists(json_path):
        avisos.append(f"⚠️ Se sobreescribirá: {json_path}")
    for aviso in avisos:
        st.warning(aviso)
    # Guardar archivos
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(hdu_text)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(hdu_json, f, ensure_ascii=False, indent=2)
    return txt_path, json_path

# Función para procesar un video y devolver HDU
@st.cache_data(show_spinner=False)
def procesar_video(video_path, prompt, extra_context, selected_model):
    # Extraer frames
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0
    frames = []
    timestamps = []
    count = 0
    for sec in range(0, int(duration), FRAME_INTERVAL):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            frames.append(pil_img)
            timestamps.append(sec)
            count += 1
            if count >= MAX_FRAMES:
                break
    cap.release()
    if not frames:
        return None, None, "No se pudieron extraer frames."
    # Construir prompt
    global_prompt = (
        (extra_context + "\n\n" if extra_context.strip() else "") +
        prompt +
        "\n\nAnalyze the following sequence of app screens (frames extracted from a video) as a whole. "
        "Describe the user flow, main actions, UI elements, and any relevant details you observe. "
        "Then, based on this visual analysis, generate a single, complete, professional user story following all the guidelines."
    )
    vision_model = genai.GenerativeModel(selected_model)
    try:
        response = vision_model.generate_content([
            global_prompt,
            *frames
        ])
        hdu_text = response.text
        hdu_json = {
            "video": os.path.basename(video_path),
            "ruta": video_path,
            "timestamp": datetime.now().isoformat(),
            "modelo": selected_model,
            "prompt": prompt,
            "extra_context": extra_context,
            "hdu": hdu_text
        }
        return hdu_text, hdu_json, None
    except Exception as e:
        return None, None, str(e)

# Función para obtener información de HDU existente
def obtener_info_hdu_existente(video_path):
    """Obtiene información detallada de una HDU existente"""
    base_path = os.path.splitext(video_path)[0]
    json_path = base_path + "_HDU.json"
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            hdu_json = json.load(f)
        
        info = {
            "video": hdu_json.get("video", "N/A"),
            "modelo": hdu_json.get("modelo", "N/A"),
            "timestamp": hdu_json.get("timestamp", "N/A"),
            "ruta": hdu_json.get("ruta", "N/A")
        }
        return info, None
    except Exception as e:
        return None, f"Error al leer información de HDU: {str(e)}"

if modo == "Subir video manualmente":
    uploaded_file = st.file_uploader("Sube un video (mp4, mov, avi)", type=["mp4", "mov", "avi"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name
        
        st.video(video_path)
        
        # Verificar si ya existe HDU para este video
        existe_hdu, txt_path, json_path = verificar_hdu_existente(video_path)
        
        if existe_hdu:
            st.success("✅ Se encontró una HDU existente para este video.")
            st.info("Para evitar costos adicionales, se cargará la HDU existente en lugar de reprocesar el video.")
            
            # Mostrar información de la HDU existente
            info_hdu, error_info = obtener_info_hdu_existente(video_path)
            if info_hdu and not error_info:
                with st.expander("📋 Información de la HDU existente", expanded=True):
                    st.write(f"**Video:** {info_hdu['video']}")
                    st.write(f"**Modelo usado:** {info_hdu['modelo']}")
                    st.write(f"**Fecha de generación:** {info_hdu['timestamp']}")
                    st.write(f"**Archivos:** {txt_path}, {json_path}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📖 Cargar HDU existente"):
                    hdu_text, hdu_json, error = cargar_hdu_existente(video_path)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.success("Historia de usuario cargada desde archivo existente:")
                        st.markdown(hdu_text)
                        st.info(f"Archivos HDU:\n- {txt_path}\n- {json_path}")
            
            with col2:
                if st.button("🔄 Reprocesar video (costos adicionales)"):
                    st.warning("⚠️ Se procederá a reprocesar el video. Esto generará costos adicionales en la API.")
                    # Continuar con el procesamiento normal
                    st.info("Extrayendo frames del video...")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps if fps else 0
                    frames = []
                    timestamps = []
                    count = 0
                    for sec in range(0, int(duration), FRAME_INTERVAL):
                        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                        ret, frame = cap.read()
                        if ret:
                            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img)
                            frames.append(pil_img)
                            timestamps.append(sec)
                            count += 1
                            if count >= MAX_FRAMES:
                                break
                    cap.release()
                    st.write(f"Se extrajeron {len(frames)} frames para análisis IA.")
                    st.image(frames, caption=[f"Frame {i+1} ({t}s)" for i, t in enumerate(timestamps)], width=180)
                    
                    if st.button("✅ Confirmar reprocesamiento"):
                        st.info("Enviando frames a Gemini Vision para análisis global del video...")
                        hdu_text, hdu_json, error = procesar_video(video_path, prompt, extra_context, selected_model)
                        if error:
                            st.error(f"Error: {error}")
                        else:
                            st.success("Historia de usuario regenerada para el video:")
                            st.markdown(hdu_text)
                            # Guardar archivos en la carpeta temporal
                            txt_path, json_path = guardar_hdu(video_path, hdu_text, hdu_json)
                            st.info(f"Archivos guardados:\n- {txt_path}\n- {json_path}")
        else:
            # No existe HDU, procesar normalmente
            st.info("Extrayendo frames del video...")
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps else 0
            frames = []
            timestamps = []
            count = 0
            for sec in range(0, int(duration), FRAME_INTERVAL):
                cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                ret, frame = cap.read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img)
                    frames.append(pil_img)
                    timestamps.append(sec)
                    count += 1
                    if count >= MAX_FRAMES:
                        break
            cap.release()
            st.write(f"Se extrajeron {len(frames)} frames para análisis IA.")
            st.image(frames, caption=[f"Frame {i+1} ({t}s)" for i, t in enumerate(timestamps)], width=180)
            if st.button("Analizar video y generar historia de usuario"):
                st.info("Enviando frames a Gemini Vision para análisis global del video...")
                hdu_text, hdu_json, error = procesar_video(video_path, prompt, extra_context, selected_model)
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.success("Historia de usuario generada para el video completo:")
                    st.markdown(hdu_text)
                    # Guardar archivos en la carpeta temporal
                    txt_path, json_path = guardar_hdu(video_path, hdu_text, hdu_json)
                    st.info(f"Archivos guardados:\n- {txt_path}\n- {json_path}")

elif modo == "Procesar carpeta completa":
    st.write("### Procesamiento masivo de videos en carpeta")
    carpeta = st.text_input("Ruta absoluta de la carpeta a procesar:", value="")
    
    # Opciones de procesamiento
    st.write("#### Opciones de procesamiento:")
    procesar_existentes = st.checkbox("Reprocesar videos que ya tienen HDU (genera costos adicionales)", value=False)
    mostrar_resumen = st.checkbox("Mostrar resumen de videos encontrados", value=True)
    
    if carpeta:
        if not os.path.isdir(carpeta):
            st.error("La ruta ingresada no es una carpeta válida.")
        else:
            patrones = ["**/*.mp4", "**/*.mov", "**/*.avi"]
            lista_videos = []
            for patron in patrones:
                lista_videos.extend(glob.glob(os.path.join(carpeta, patron), recursive=True))
            
            if not lista_videos:
                st.warning("No se encontraron videos en la carpeta seleccionada.")
            else:
                # Analizar qué videos ya tienen HDU
                videos_con_hdu = []
                videos_sin_hdu = []
                
                for video_path in lista_videos:
                    existe_hdu, _, _ = verificar_hdu_existente(video_path)
                    if existe_hdu:
                        videos_con_hdu.append(video_path)
                    else:
                        videos_sin_hdu.append(video_path)
                
                if mostrar_resumen:
                    st.write(f"📊 **Resumen de videos encontrados:**")
                    st.write(f"- Total de videos: {len(lista_videos)}")
                    st.write(f"- Videos con HDU existente: {len(videos_con_hdu)}")
                    st.write(f"- Videos sin HDU: {len(videos_sin_hdu)}")
                    
                    if videos_con_hdu:
                        st.write("**Videos con HDU existente:**")
                        for video in videos_con_hdu:
                            st.write(f"  ✅ {os.path.basename(video)}")
                    
                    if videos_sin_hdu:
                        st.write("**Videos sin HDU:**")
                        for video in videos_sin_hdu:
                            st.write(f"  ⏳ {os.path.basename(video)}")
                
                # Determinar qué videos procesar
                videos_a_procesar = []
                if procesar_existentes:
                    videos_a_procesar = lista_videos
                    st.warning(f"⚠️ Se procesarán TODOS los {len(lista_videos)} videos (incluyendo {len(videos_con_hdu)} con HDU existente)")
                else:
                    videos_a_procesar = videos_sin_hdu
                    st.success(f"✅ Se procesarán solo los {len(videos_sin_hdu)} videos sin HDU existente")
                
                if videos_a_procesar:
                    if st.button(f"🚀 Iniciar procesamiento de {len(videos_a_procesar)} videos"):
                        st.write(f"Procesando {len(videos_a_procesar)} videos...")
                        log_lines = []
                        progreso = st.progress(0)
                        videos_procesados = 0
                        videos_omitidos = 0
                        errores = 0
                        
                        for idx, video_path in enumerate(videos_a_procesar):
                            existe_hdu, _, _ = verificar_hdu_existente(video_path)
                            
                            if existe_hdu and not procesar_existentes:
                                st.write(f"⏭️ Omitiendo (HDU existente): {os.path.basename(video_path)}")
                                log_lines.append(f"OMITIDO: {video_path} - HDU existente")
                                videos_omitidos += 1
                                continue
                            
                            st.write(f"🔄 Procesando: {os.path.basename(video_path)}")
                            
                            if existe_hdu and procesar_existentes:
                                st.write(f"  ⚠️ Reprocesando video con HDU existente")
                            
                            hdu_text, hdu_json, error = procesar_video(video_path, prompt, extra_context, selected_model)
                            
                            if error:
                                st.error(f"❌ Error en {os.path.basename(video_path)}: {error}")
                                log_lines.append(f"ERROR: {video_path}: {error}")
                                errores += 1
                                continue
                            
                            txt_path, json_path = guardar_hdu(video_path, hdu_text, hdu_json)
                            st.success(f"✅ HDU generada para {os.path.basename(video_path)}")
                            log_lines.append(f"OK: {video_path} -> {txt_path}, {json_path}")
                            videos_procesados += 1
                            progreso.progress((idx+1)/len(videos_a_procesar))
                        
                        # Resumen final
                        st.write("---")
                        st.write("📋 **Resumen del procesamiento:**")
                        st.write(f"- Videos procesados: {videos_procesados}")
                        st.write(f"- Videos omitidos: {videos_omitidos}")
                        st.write(f"- Errores: {errores}")
                        
                        # Guardar log
                        log_path = os.path.join(carpeta, f"procesamiento_hdu_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
                        with open(log_path, "w", encoding="utf-8") as flog:
                            flog.write(f"Resumen del procesamiento:\n")
                            flog.write(f"- Videos procesados: {videos_procesados}\n")
                            flog.write(f"- Videos omitidos: {videos_omitidos}\n")
                            flog.write(f"- Errores: {errores}\n")
                            flog.write(f"- Total de videos encontrados: {len(lista_videos)}\n")
                            flog.write(f"- Videos con HDU existente: {len(videos_con_hdu)}\n")
                            flog.write(f"- Videos sin HDU: {len(videos_sin_hdu)}\n")
                            flog.write(f"- Reprocesar existentes: {procesar_existentes}\n\n")
                            flog.write("Detalle por video:\n")
                            flog.write("\n".join(log_lines))
                        
                        st.info(f"📄 Log guardado en: {log_path}")
                else:
                    st.info("No hay videos para procesar con la configuración actual.") 
