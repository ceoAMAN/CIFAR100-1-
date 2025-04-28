import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from model import CustomCNN


def load_model(model_version="trained_model.pth"):
    model = CustomCNN(num_classes=10)
    model_path = f'model/{model_version}'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


model_version = st.sidebar.selectbox("Select Model Version", ["trained_model.pth", "alternate_model.pth"])
model = load_model(model_version)


class_labels = [
    "Class 0", "Class 1", "Class 2", "Class 3", "Class 4",
    "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"
]

st.title("Image Classifier")
st.write("Upload one or more images to classify them into one of the 10 classes!")

uploaded_images = st.file_uploader("Choose images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        try:
            
            with st.spinner("Processing image..."):
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)

               
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image_tensor = transform(image).unsqueeze(0)

               
                with torch.no_grad():
                    output = model(image_tensor)
                    _, predicted_class = torch.max(output, 1)
                    predicted_label = class_labels[predicted_class.item()]

                
                st.success(f"Predicted Class: {predicted_label} (Class {predicted_class.item()})")
        except UnidentifiedImageError:
            st.error(f"Error: The file '{uploaded_image.name}' is not a valid image.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please upload one or more images to get started.")