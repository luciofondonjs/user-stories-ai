import streamlit as st
import os
import tempfile
import cv2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

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
    "gemini-1.5-pro"
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

uploaded_file = st.file_uploader("Sube un video (mp4, mov, avi)", type=["mp4", "mov", "avi"])

FRAME_INTERVAL = 2  # segundos entre frames extraídos
MAX_FRAMES = 8      # máximo de frames a analizar para no exceder el límite de la API

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
        tmpfile.write(uploaded_file.read())
        video_path = tmpfile.name

    st.video(video_path)
    st.info("Extrayendo frames del video...")

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

    st.write(f"Se extrajeron {len(frames)} frames para análisis IA.")
    st.image(frames, caption=[f"Frame {i+1} ({t}s)" for i, t in enumerate(timestamps)], width=180)

    if st.button("Analizar video y generar historia de usuario"):
        st.info("Enviando frames a Gemini Vision para análisis global del video...")
        vision_model = genai.GenerativeModel(selected_model)
        # Construir el prompt para análisis global
        global_prompt = (
            (extra_context + "\n\n" if extra_context.strip() else "") +
            prompt +
            "\n\nAnalyze the following sequence of app screens (frames extracted from a video) as a whole. "
            "Describe the user flow, main actions, UI elements, and any relevant details you observe. "
            "Then, based on this visual analysis, generate a single, complete, professional user story following all the guidelines."
        )
        # Enviar todos los frames juntos
        response = vision_model.generate_content([
            global_prompt,
            *frames
        ])
        st.success("Historia de usuario generada para el video completo:")
        st.markdown(response.text) 