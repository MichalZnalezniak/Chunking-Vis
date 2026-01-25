import atexit
import gc
import threading
from pathlib import Path

import onnxruntime as ort
import onnxruntime_genai as og
from django.apps import AppConfig
from django.core.signals import got_request_exception
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer


class ChunkerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chunker"

    semantic_model = None
    semantic_model_loaded = False
    agentic_model = None
    agentic_model_loaded = False

    def ready(self):
        # Load the model in a separate thread
        # download onnx - phi model
        base_path = Path(__file__).resolve().parent.parent
        model_path = base_path / "utils" / "Phi3"
        model_file = model_path / "model.onnx"
        model_path.mkdir(parents=True, exist_ok=True)
        if not model_file.exists():
            snapshot_download(
                repo_id="microsoft/Phi-3-mini-4k-instruct-onnx",
                local_dir=model_path,
                allow_patterns=["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*"],
            )

        def load_model_async():
            ### Load semantic model
            ChunkerConfig.tokenizer_semantic = Tokenizer.from_file("utils/MiniLM/tokenizer/tokenizer.json")
            ChunkerConfig.session_semnatic = ort.InferenceSession(
                "utils/MiniLM/all-MiniLM-L6-v2_int8.onnx/onnx/model_quint8_avx2.onnx"
            )
            ChunkerConfig.semantic_model_loaded = True
            ### Load agentic model
            ChunkerConfig.model_agentic = og.Model("utils/Phi3/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4")
            ChunkerConfig.params = og.GeneratorParams(ChunkerConfig.model_agentic)
            ChunkerConfig.tokenizer_agentic = og.Tokenizer(ChunkerConfig.model_agentic)

        # Clean up functions
        atexit.register(cleanup_genai)
        got_request_exception.connect(cleanup_genai)
        threading.Thread(target=load_model_async, daemon=True).start()


def cleanup_genai(sender=None, **kwargs):
    cfg = ChunkerConfig
    try:
        del cfg.model_agentic
        del cfg.params
        del cfg.tokenizer_agentic
        del cfg.session_semnatic
        del cfg.tokenizer_semantic
    except Exception as e:
        print("Cleanup error:", e)
    gc.collect()
