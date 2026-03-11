import os
from datetime import datetime

import requests
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.getenv("METER_API_URL", "http://localhost:8000/infer")
API_INPUT_SIZE = (400, 100)  # pipeline.py 側の resize と合わせる


def resize_for_api_display(image: Image.Image) -> Image.Image:
    return image.resize(API_INPUT_SIZE)


def draw_bboxes(image: Image.Image, digits: list[dict]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for d in digits:
        x, y, w, h = d["bbox"]
        value = str(d["value"])
        score = float(d["score"])

        draw.rectangle((x, y, x + w, y + h), outline="lime", width=3)
        draw.text((x, max(0, y - 16)), f"{value} ({score:.2f})", fill="lime")

    return img


def normalize_meter_value(result: dict) -> str:
    """
    APIの meter_value が空でも、digits から表示用文字列を再構成する。
    """
    raw_value = result.get("meter_value", None)

    # まず meter_value をそのまま使えるか判定
    if raw_value is not None:
        text = str(raw_value).strip()
        if text != "":
            return text

    # meter_value が空なら digits から復元
    digits = result.get("digits", [])
    values = []

    for d in digits:
        v = str(d.get("value", "")).strip()
        if v == "" or v.lower() == "blank":
            continue
        values.append(v)

    return "".join(values)


def init_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "latest_filename" not in st.session_state:
        st.session_state.latest_filename = None
    if "latest_image" not in st.session_state:
        st.session_state.latest_image = None


def add_history(filename: str, result: dict):
    meter_value = normalize_meter_value(result)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.history.insert(
        0,
        {
            "timestamp": timestamp,
            "filename": filename,
            "meter_value": meter_value,
            "ok": result.get("ok", False),
            "reason": result.get("reason"),
            "digits": result.get("digits", []),
        },
    )


st.set_page_config(page_title="Meter Reader Demo", layout="wide")
init_session_state()

st.title("7セグメーター認識デモ")
st.write("画像をアップロードして推論すると、認識結果と履歴を表示します。")

uploaded_file = st.file_uploader(
    "パネル画像をアップロードしてください",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    display_image = resize_for_api_display(original_image)

    with col1:
        st.subheader("入力画像")
        st.image(display_image, width="stretch")

    if st.button("推論を実行"):
        try:
            uploaded_file.seek(0)
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type or "application/octet-stream",
                )
            }

            with st.spinner("推論中..."):
                response = requests.post(API_URL, files=files, timeout=30)

            if response.status_code != 200:
                st.error(f"APIエラー: status={response.status_code}")
                st.code(response.text)
            else:
                result = response.json()

                st.session_state.latest_result = result
                st.session_state.latest_filename = uploaded_file.name
                st.session_state.latest_image = display_image

                add_history(uploaded_file.name, result)

        except requests.exceptions.ConnectionError:
            st.error("FastAPI サーバに接続できません。先に `uvicorn api.main:app --reload` を起動してください。")
        except Exception as e:
            st.error(f"予期しないエラー: {e}")

# 最新結果の表示
if st.session_state.latest_result is not None and st.session_state.latest_image is not None:
    result = st.session_state.latest_result
    display_image = st.session_state.latest_image
    digits = result.get("digits", [])
    meter_value = normalize_meter_value(result)
    ok = result.get("ok", False)
    reason = result.get("reason")

    annotated = draw_bboxes(display_image, digits)

    with col2:
        st.subheader("認識結果画像")
        st.image(annotated, caption="bbox付き認識結果", width="stretch")

    st.markdown("---")
    st.subheader("認識数値")

    if meter_value != "":
        st.markdown(
            f"""
            <div style="
                font-size: 2.8rem;
                font-weight: bold;
                padding: 0.8rem 1rem;
                border-radius: 0.5rem;
                border: 1px solid #ccc;
                background-color: #f7f7f7;
                text-align: center;
                margin-bottom: 1rem;
            ">
                {meter_value}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("数値を認識できませんでした。")

    st.write(f"**判定**: {'OK' if ok else 'NG'}")
    st.write(f"**理由**: {reason}")

    st.subheader("桁ごとの結果")
    st.json(digits)

st.markdown("---")
st.subheader("認識履歴")

if st.button("履歴をクリア"):
    st.session_state.history = []
    st.session_state.latest_result = None
    st.session_state.latest_filename = None
    st.session_state.latest_image = None
    st.rerun()

if not st.session_state.history:
    st.info("まだ履歴はありません。")
else:
    for i, item in enumerate(st.session_state.history, start=1):
        with st.expander(
            f"{i}. [{item['timestamp']}] {item['filename']} → {item['meter_value']} ({'OK' if item['ok'] else 'NG'})",
            expanded=(i == 1),
        ):
            st.write(f"**時刻**: {item['timestamp']}")
            st.write(f"**ファイル名**: {item['filename']}")
            st.write(f"**認識結果**: {item['meter_value']}")
            st.write(f"**判定**: {'OK' if item['ok'] else 'NG'}")
            st.write(f"**理由**: {item['reason']}")
            st.json(item["digits"])

