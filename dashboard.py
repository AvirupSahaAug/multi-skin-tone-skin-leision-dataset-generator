import streamlit as st
import json
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Skin GAN Training Dashboard", layout="wide")
st.title("🎛️ Multi-Skin Tone GAN Training Dashboard")

st.markdown("This dashboard tracks the training of the 6-Tone ITA GAN. The training script reads `training_config.json` every epoch, so adjusting the dials below updates the loss weights **live** without restarting the run!")

# Start Training Button
if st.button("🚀 Start Training Process (70 Epochs)"):
    python_exec = "../gpu/Scripts/python.exe"
    script = "src/train_unified_ita.py"
    if os.path.exists(python_exec):
        subprocess.Popen([python_exec, script, "--n_epochs", "70"])
        st.success("✅ Training started in the background! Watch your terminal for the tqdm progress bar.")
    else:
        st.error(f"❌ Could not find python executable at {python_exec}")

st.markdown("---")

# Main Layout: Display the latest generated image
st.header("🖼️ Latest Epoch Generation")
st.markdown("This image shows the synthesized tones side-by-side. The score below each column is baked cleanly into a single preview strip with a reduced display size!")

col1, col2 = st.columns([0.8, 0.2])
with col2:
    if st.button("🔄 Refresh Images"):
        pass

if os.path.exists("latest_sample.png"):
    try:
        img = Image.open("latest_sample.png")
        
        # Load ITAs if available
        itas = ["N/A"] * 6
        if os.path.exists("latest_itas.json"):
            with open("latest_itas.json", "r") as f:
                itas = json.load(f)
                
        targets = [50, 38, 26, 14, 2, -10]
        names = ["Tone 0", "Tone 1", "Tone 2", "Tone 3", "Tone 4", "Tone 5"]
        
        # Create a new padded image to physically draw text onto the image
        orig_w, orig_h = img.size
        # Each column's width is the total image divided by 6 heads
        col_w = orig_w / 6
        
        # Expand height by 60 pixels to give comfortable room for textual annotations
        new_h = orig_h + 60
        annotated_img = Image.new("RGB", (orig_w, new_h), color="black")
        annotated_img.paste(img, (0, 0))
        
        draw = ImageDraw.Draw(annotated_img)
        font = ImageFont.load_default()
        
        for i in range(6):
            cx = (i * col_w) + (col_w / 2)
            score_str = str(itas[i]) if i < len(itas) else "N/A"
            text_top = f"Tgt: {targets[i]} ITA"
            text_bot = f"Act: {score_str} ITA"
            
            # We attempt rough center offsets using a hardcoded pixel width estimate for standard bitmap fonts
            draw.text((cx - 35, orig_h + 10), names[i], fill="white", font=font)
            draw.text((cx - 35, orig_h + 25), text_top, fill="lightgray", font=font)
            draw.text((cx - 35, orig_h + 40), text_bot, fill="yellow", font=font)
            
        # Using a fixed width stringently scales the image down on-screen as requested
        st.image(annotated_img, width=900)
    except Exception as e:
        st.warning(f"Image is currently mid-write or unavailable. Try refreshing. {e}")
else:
    st.info("No generated images yet. Start training and wait for the first epoch to complete!")


st.markdown("---")
st.header("🎚️ Live Tuning Dials")

config_path = "training_config.json"

# Initialize config if not exists
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        json.dump({"lambda_adv": 1.0, "lambda_tone": 5.0, "lambda_cons": 10.0}, f)

# Read current config to populate sliders
with open(config_path, "r") as f:
    config = json.load(f)

# Put dials below the image
dial_col1, dial_col2, dial_col3, dial_col4 = st.columns(4)

with dial_col1:
    lambda_adv = st.slider("🛡️ Base Adv Weight", 0.1, 10.0, float(config.get("lambda_adv", 1.0)), 0.1, key="l_adv")
    st.caption("Weight for fooling the discriminator on base.")
    
with dial_col2:
    lambda_head_adv = st.slider("👾 Head Adv Weight", 0.0, 1.0, float(config.get("lambda_head_adv", 0.05)), 0.01, key="l_head")
    st.caption("Weight applied to the 6 output color heads.")

with dial_col3:
    lambda_tone = st.slider("🎨 Tone Distance (ITA)", 0.1, 50.0, float(config.get("lambda_tone", 5.0)), 0.5, key="l_tone")
    st.caption("Weight for meeting the exact Fitzpatrick ITA.")

with dial_col4:
    lambda_cons = st.slider("🔗 Consistency Weight", 0.1, 50.0, float(config.get("lambda_cons", 10.0)), 0.5, key="l_cons")
    st.caption("Weight for locking identical edge-structures.")

# Save the new config back to the file dynamically (training script reads this every epoch)
new_config = {"lambda_adv": lambda_adv, "lambda_tone": lambda_tone, "lambda_cons": lambda_cons, "lambda_head_adv": lambda_head_adv}
if new_config != config:
    with open(config_path, "w") as f:
        json.dump(new_config, f)
    # Give a small visual cue that it was updated
    st.toast("✅ Config file updated! The training loop will use these weights next epoch.", icon="👍")
