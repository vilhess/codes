import streamlit as st 
import torch 
import matplotlib.pyplot as plt 

DEVICE="mps"

model = torch.load("checkpoint/model.pkl").to(DEVICE)
decoder = model.decoder

st.title("Jouer avec l'espace latent")

st.image("figures/latent_space.png")

st.write('')

col1, col2 = st.columns(2)

with col1:
    st.write('')
    st.write('')

    z1 = st.slider("z1 value", min_value=-6., max_value=6., step=0.1, value=0.)
    z2 = st.slider("z2 value", min_value=-6., max_value=6., step=0.1, value=0.)

z = torch.Tensor([z1, z2]).unsqueeze(0).to(DEVICE)
generated = decoder(z)
img_generated = generated.squeeze(0).detach().cpu().view(1, 28, 28).permute(1, 2, 0)

with col2:
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img_generated, cmap="gray")
    st.pyplot(fig)