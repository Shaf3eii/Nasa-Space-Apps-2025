# app.py

import streamlit as st
import joblib
import numpy as np
import os

# Dark Mode CSS
# This CSS is injected to create the "bare mode" / dark theme.
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #0E1117;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #1a1d24;
}

/* Text color */
body, .st-b7, .st-bb, .st-at, .st-av, .st-ax, .st-ay, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj, .st-ak, .st-al, .st-am, .st-an, .st-ao, .st-ap, .st-aq, .st-ar, .st-as {
    color: #FAFAFA;
}

/* Button styling */
.stButton>button {
    color: #FAFAFA;
    background-color: #262730;
    border: 1px solid #f63366;
}
.stButton>button:hover {
    color: #FFFFFF;
    background-color: #f63366;
    border: 1px solid #f63366;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #FAFAFA;
}

</style>
""", unsafe_allow_html=True)


# Load Model and Scaler 
# Use a function with st.cache_resource to load these files only once,
# which makes the app faster.
@st.cache_resource
def load_model_and_scaler():
    """Loads the pre-trained model and scaler from disk."""
    model_path = 'planet_model.joblib'
    scaler_path = 'scaler.joblib'

    # Check if the files exist before trying to load them
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None


model, scaler = load_model_and_scaler()

#  Web App User Interface 

# Set up the page title and icon
st.set_page_config(page_title="Exoplanet Predictor", page_icon="🪐")

# Main title of the app
st.title("🪐 NASA Exoplanet Prediction AI")

# Check if the model files were loaded successfully. If not, show an error.
if model is None or scaler is None:
    st.error("Error: Model files (`planet_model.joblib`, `scaler.joblib`) not found.")
    st.warning("Please run the `train_model.py` script first to generate the necessary files.", icon="⚠️")
else:
    # If models are loaded, build the main part of the app
    st.markdown("""
    **Is it a planet?** This AI model, trained on NASA's Kepler mission data, can predict whether a star's signal indicates a potential exoplanet. 

    Use the sliders on the left to input the characteristics of a transit signal you want to analyze.
    """)

    #  Sidebar for User Inputs 
    st.sidebar.header("Signal Features")
    st.sidebar.markdown("Adjust the sliders to match the observed stellar signal.")

    # Create sliders and number inputs in the sidebar for user input.
    # These correspond to the features our model was trained on.
    fp_ss = st.sidebar.slider("Stellar Eclipse Flag (koi_fpflag_ss)", 0, 1, 0,
                              help="Flag indicating a stellar eclipse.")
    fp_co = st.sidebar.slider("Centroid Offset Flag (koi_fpflag_co)", 0, 1, 0,
                              help="Flag indicating a centroid offset during transit.")
    fp_ec = st.sidebar.slider("Ephemeris Match Flag (koi_fpflag_ec)", 0, 1, 0,
                              help="Flag indicating a mismatch with the transit ephemeris.")

    st.sidebar.markdown("---")  # Visual separator

    period = st.sidebar.number_input("Orbital Period [days] (koi_period)", min_value=0.0, value=50.0, step=1.0)
    duration = st.sidebar.number_input("Transit Duration [hours] (koi_duration)", min_value=0.0, value=5.0, step=0.1)
    depth = st.sidebar.number_input("Transit Depth [ppm] (koi_depth)", min_value=0.0, value=1000.0, step=10.0)
    prad = st.sidebar.number_input("Planetary Radius [Earth radii] (koi_prad)", min_value=0.0, value=1.0, step=0.1)
    teq = st.sidebar.number_input("Equilibrium Temperature [K] (koi_teq)", min_value=0, value=300, step=10)
    insol = st.sidebar.number_input("Insolation Flux [Earth flux] (koi_insol)", min_value=0.0, value=1.0, step=0.1)

    # Prediction Logic 
    # Create a button that triggers the prediction when clicked.
    if st.button("Analyze Signal and Predict", type="primary"):
        # Collect all the inputs from the sidebar into a single NumPy array
        user_input = np.array([[
            fp_ss, fp_co, fp_ec, period, duration, depth, prad, teq, insol
        ]])

        # Use the loaded scaler to transform the user's input
        # This is a crucial step to match the format of the training data
        scaled_input = scaler.transform(user_input)

        # Make a prediction and get the probabilities
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        st.subheader("Prediction Result:")

        # Display the result based on the model's output
        if prediction[0] == 1:
            st.success(f"**Potential Planet Detected!** 🔭")
            st.write(f"The model is **{prediction_proba[0][1] * 100:.2f}%** confident this is an exoplanet candidate.")
            
            st.image("https://images.pexels.com/photos/39561/solar-flare-sun-eruption-energy-39561.jpeg",
                     caption="An artist's impression of an exoplanet system.")
        else:
            st.error(f"**Likely Not a Planet.** 📉")
            st.write(
                f"The model is **{prediction_proba[0][0] * 100:.2f}%** confident this is a false positive or other stellar phenomenon.")
            
            st.image("https://images.pexels.com/photos/110854/pexels-photo-110854.jpeg",
                     caption="The signal might be caused by other stellar activity.")

