import io
import os
import tempfile
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import onnxruntime as ort
import streamlit as st
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from PIL import Image

st.set_page_config(
    page_title="Сегментация коронарных артерий",
    page_icon="🫀",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model(model_path=None):
    with st.spinner("Загрузка модели..."):
        if model_path is None:
            model_path = os.getenv("MODEL_PATH", "data/models/model.onnx")

        try:
            if os.path.exists(model_path):
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if ort.get_device() == "GPU"  # type: ignore
                    else ["CPUExecutionProvider"]
                )
                session = ort.InferenceSession(model_path, providers=providers)
                st.success(f"Модель загружена из: {model_path}")
                return session
            else:
                st.warning(
                    f"Модель не найдена по пути: {model_path}. Загрузите модель через боковую панель."
                )
                return None
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {str(e)}. Проверьте формат файла.")
            return None


def load_model_from_upload(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


if "model" not in st.session_state:
    st.session_state.model = load_model()
if "model_path" not in st.session_state:
    st.session_state.model_path = os.getenv("MODEL_PATH", "data/models/model.onnx")

with st.sidebar:
    st.header("⚙️ Настройки модели")

    model_source = st.radio(
        "Источник модели:", ["Использовать путь", "Загрузить файл"], index=0
    )

    if model_source == "Загрузить файл":
        uploaded_model = st.file_uploader(
            "Загрузите ONNX модель", type=["onnx"], key="model_uploader"
        )

        if uploaded_model is not None:
            if st.button("Применить модель"):
                model_path = load_model_from_upload(uploaded_model)
                st.session_state.model = load_model(model_path)
                st.session_state.model_path = model_path
                st.rerun()
    else:
        custom_path = st.text_input(
            "Путь к модели:",
            value=st.session_state.model_path,
            help="Укажите путь к файлу .onnx или используйте переменную окружения MODEL_PATH",
        )

        if st.button("Загрузить модель"):
            st.session_state.model = load_model(custom_path)
            st.session_state.model_path = custom_path
            st.rerun()

    st.divider()

    if st.session_state.model is not None:
        st.success("✅ Модель активна")
    else:
        st.error("❌ Модель не загружена")

val_test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        CropForegroundd(keys=["image"], source_key="image"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        Resized(keys=["image"], spatial_size=(128, 128, 64), mode="trilinear"),
        ToTensord(keys=["image"]),
    ]
)


def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def predict_mask(image_path):
    if st.session_state.model is None:
        st.error("Модель не загружена! Загрузите модель через боковую панель.")
        return None, 0.0

    data_dict = {"image": image_path}
    dataset = Dataset(data=[data_dict], transform=val_test_transforms)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    start_time = time.time()

    for batch in loader:
        inputs = batch["image"].numpy()
        input_name = st.session_state.model.get_inputs()[0].name
        outputs = st.session_state.model.run(None, {input_name: inputs})
        mask = (outputs[0] > 0.5).astype(np.float32)[0, 0]
        original_shape = nib.load(image_path).shape  # type: ignore
        mask_resized = resize_mask_to_original(mask, original_shape)

    inference_time = time.time() - start_time

    return mask_resized, inference_time


def resize_mask_to_original(mask, target_shape):
    from skimage.transform import resize

    return resize(
        mask,
        output_shape=target_shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )


def visualize_slice(image, mask, slice_idx, opacity=0.5):
    img_slice = image[:, :, slice_idx]
    mask_slice = mask[:, :, slice_idx]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(img_slice.T, cmap="gray", origin="lower")
    ax.imshow(mask_slice.T, cmap="Reds", alpha=opacity, origin="lower")
    ax.axis("off")
    return fig


def main_page():
    st.title("Сегментация коронарных артерий на КТ-сканах")
    st.write("""
    **Проект:** Использование 3D U-Net для автоматической сегментации коронарных артерий
    на основе датасета ImageCAS. Модель анализирует КТ-изображения и выделяет области артерий.
    """)

    st.subheader("Пример данных из датасета")
    st.write("Для демонстрации загрузите свой файл в разделе Предсказание")

    st.info("💡 Загрузите модель через боковую панель, если она еще не загружена.")


def prediction_section():
    st.header("Предсказание сегментации")

    if st.session_state.model is None:
        st.warning("⚠️ Сначала загрузите модель через боковую панель!")
        return

    uploaded_file = st.file_uploader(
        "Загрузите КТ-изображение (.nii.gz)",
        type=["nii.gz"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        with st.spinner("Обработка файла..."):
            file_path = process_uploaded_file(uploaded_file)
            nifti_img = nib.load(file_path)
            image_data = nifti_img.get_fdata()  # type: ignore

            st.success("Файл успешно загружен!")

            mask, inf_time = predict_mask(file_path)

            if mask is None:
                return

            st.success(f"Сегментация завершена! Время: {inf_time:.2f} сек")

            slice_idx = st.slider(
                "Выберите срез (ось Z)",
                0,
                image_data.shape[2] - 1,
                image_data.shape[2] // 2,
            )
            opacity = st.slider("Прозрачность маски", 0, 100, 20) / 100.0

            fig = visualize_slice(image_data, mask, slice_idx, opacity)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = Image.open(buf)

            st.image(img, width=500)

            mask_img = nib.Nifti1Image(mask, nifti_img.affine)  # type: ignore
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp:
                nib.save(mask_img, tmp.name)
                with open(tmp.name, "rb") as f:
                    st.download_button(
                        label="Скачать маску сегментации",
                        data=f,
                        file_name="artery_mask.nii.gz",
                        mime="application/gzip",
                    )
            os.unlink(tmp.name)
        os.unlink(file_path)


def metrics_section():
    st.header("Метрики качества модели")
    st.subheader("Производительность")
    col1, col2 = st.columns(2)
    col1.metric("Средний Dice Score", "0.21")
    col2.metric("Время инференса (GPU)", "~5.2 сек")


menu = {
    "Главная": main_page,
    "Предсказание": prediction_section,
    "Метрики": metrics_section,
}

selected = st.sidebar.selectbox("Навигация", list(menu.keys()))
menu[selected]()
