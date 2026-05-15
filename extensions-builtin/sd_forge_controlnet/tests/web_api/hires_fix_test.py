import importlib.util
import os

from .template import (
    APITestTemplate,
    girl_img,
    disable_in_cq,
    get_model,
)


def _load_kohya_hrfix():
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", "..", ".."))
    target = os.path.join(repo_root, "extensions-builtin", "reForge-KohyaHRFix",
                          "scripts", "kohya_hrfix.py")
    spec = importlib.util.spec_from_file_location("reforge_kohya_hrfix_module", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.KohyaHRFixForForge


KohyaHRFixForForge = _load_kohya_hrfix()

@disable_in_cq
def test_kohya_hrfix_with_sd15_controlnet():
    APITestTemplate(
        "test_kohya_hrfix_with_sd15_controlnet",
        "txt2img", 
        payload_overrides={
            "prompt": "a cat",
            "steps": 20,
            "width": 1024,
            "height": 1024,
        },
        unit_overrides={
            "image": girl_img,
            "module": "canny",
            "model": get_model("control_v11p_sd15_canny"),
        },
        script_args=(True, 3, 2.0, 0.0, 0.35, True, "bicubic", "bicubic"),
        scripts=[KohyaHRFixForForge()],
    ).exec()
