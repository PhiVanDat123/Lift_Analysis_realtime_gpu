"""
Gradio UI entry point.
All business logic lives in core/.

FastAPI app is created first, /stream route added, then Gradio mounted —
so MJPEG stream works on the same port as Gradio on any host (RunPod, etc.).
"""
import queue

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from core.processor import process_video_streaming, _stream_state


# ── FastAPI app with MJPEG route ──────────────────────────────────────────────

app = FastAPI()


def _mjpeg_generator():
    while True:
        active = _stream_state["active"].is_set()
        try:
            frame_jpg = _stream_state["queue"].get(timeout=0.1)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_jpg
                + b"\r\n"
            )
        except queue.Empty:
            if not active and _stream_state["queue"].empty():
                break


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Exercise Pose Feedback") as demo:
    gr.Markdown("# AI Exercise Pose Feedback")
    gr.Markdown("Chọn bài tập, upload video và nhận phản hồi tư thế theo thời gian thực.")

    exercise_radio = gr.Radio(
        choices=["Bench Press", "Back Squat", "Deadlift"],
        value="Back Squat",
        label="Bài tập",
    )

    with gr.Row():
        video_input   = gr.Video(label="Video đầu vào", sources=["upload"], height=480)
        video_display = gr.HTML(
            '<p style="color:#888;font-size:14px;padding:16px;">Upload video và nhấn Phân tích để bắt đầu.</p>',
            label="Phân tích real-time",
        )

    video_feedback = gr.Textbox(
        label="Phản hồi tư thế",
        lines=10,
        interactive=False,
        placeholder="Phản hồi sẽ hiển thị ở đây sau mỗi rep...",
    )
    analyze_btn = gr.Button("Phân tích video", variant="primary")

    analyze_btn.click(
        fn=process_video_streaming,
        inputs=[video_input, exercise_radio],
        outputs=[video_display, video_feedback],
    )


# ── Mount Gradio onto FastAPI, then run ───────────────────────────────────────

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
