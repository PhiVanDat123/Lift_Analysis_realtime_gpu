"""
Gradio UI entry point.
All business logic lives in core/.
"""
import asyncio
import cv2
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from core.processor import (
    get_latest_frame,
    get_raw_frame,
    set_exercise,
    get_current_state,
    reset_counter,
    process_video,
)

fastapi_app = FastAPI()


def _mjpeg_generator(frame_fn, fps: float = 30.0):
    """Async MJPEG generator — encodes frames from frame_fn at given fps."""
    async def generate():
        interval = 1.0 / fps
        while True:
            frame = frame_fn()
            if frame is not None:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       jpeg.tobytes() + b"\r\n")
            await asyncio.sleep(interval)
    return generate


@fastapi_app.get("/stream")
async def video_stream():
    """Annotated feed: raw frame + pose/barbell overlay, 30fps."""
    return StreamingResponse(
        _mjpeg_generator(get_latest_frame)(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


@fastapi_app.get("/raw_stream")
async def raw_stream():
    """Raw camera feed, 30fps."""
    return StreamingResponse(
        _mjpeg_generator(get_raw_frame)(),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


_IMG_STYLE = "width:100%;border-radius:4px;display:block;"

with gr.Blocks(title="AI Exercise Pose Feedback") as demo:
    gr.Markdown("# AI Exercise Pose Feedback")
    gr.Markdown(
        "Chọn bài tập rồi dùng **webcam** hoặc **upload video** để nhận phản hồi tư thế."
    )

    exercise_radio = gr.Radio(
        choices=["Bench Press", "Back Squat", "Deadlift"],
        value="Back Squat",
        label="Bài tập",
    )

    with gr.Tabs():

        # ── Tab 1: Webcam ──────────────────────────────────────────────────────
        with gr.Tab("Webcam (real-time)"):
            session_state = gr.State(
                {"counter": 0, "stage": "", "feedback": "", "last_score": None}
            )

            with gr.Row():
                gr.HTML(f'<img src="/raw_stream" style="{_IMG_STYLE}" />')
                gr.HTML(f'<img src="/stream"     style="{_IMG_STYLE}" />')

            webcam_feedback = gr.Textbox(
                label="Phản hồi tư thế",
                lines=4,
                interactive=False,
                placeholder="Phản hồi sẽ hiển thị ở đây sau mỗi rep...",
            )
            reset_btn = gr.Button("Reset rep counter", variant="secondary")

            # Update exercise in worker when radio changes
            exercise_radio.change(
                fn=set_exercise,
                inputs=[exercise_radio],
                outputs=[],
            )

            # Poll feedback + state every second via Timer
            timer = gr.Timer(value=1.0)
            timer.tick(
                fn=get_current_state,
                outputs=[webcam_feedback, session_state],
            )

            reset_btn.click(
                fn=reset_counter,
                inputs=[session_state],
                outputs=[session_state],
            )

        # ── Tab 2: Upload Video ────────────────────────────────────────────────
        with gr.Tab("Upload Video"):
            with gr.Row():
                video_input  = gr.Video(label="Video đầu vào", sources=["upload"])
                video_output = gr.Video(label="Video đã phân tích")

            video_feedback = gr.Textbox(
                label="Tổng kết tư thế",
                lines=10,
                interactive=False,
                placeholder="Kết quả phân tích sẽ hiển thị ở đây sau khi xử lý xong...",
            )
            analyze_btn = gr.Button("Phân tích video", variant="primary")

            analyze_btn.click(
                fn=process_video,
                inputs=[video_input, exercise_radio],
                outputs=[video_output, video_feedback],
            )

app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
