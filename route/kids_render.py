from __future__ import annotations

import os
import io
import re
import json
import base64
import uuid
import traceback
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List

import anyio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from openai import OpenAI
from tripo3d import TripoClient
from tripo3d.models import TaskStatus

from service.nano_banana import render_one_image


router = APIRouter(prefix="/api/v1/kids", tags=["kids"])


# -----------------------------
# Utils
# -----------------------------
def _is_truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

DEBUG = _is_truthy(os.environ.get("DEBUG", "false"))

def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent

    markers = ("pyproject.toml", "requirements.txt", ".git")
    for p in [cur] + list(cur.parents):
        for m in markers:
            if (p / m).exists():
                return p
    return cur

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "")).expanduser()
if str(PROJECT_ROOT).strip():
    PROJECT_ROOT = PROJECT_ROOT.resolve()
else:
    PROJECT_ROOT = _find_project_root(Path(__file__))

PUBLIC_DIR = Path(os.environ.get("PUBLIC_DIR", PROJECT_ROOT / "public")).resolve()
GENERATED_DIR = Path(os.environ.get("GENERATED_DIR", PUBLIC_DIR / "generated")).resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

STATIC_PREFIX = os.environ.get("GENERATED_URL_PREFIX", "/api/generated").rstrip("/")

def _mime_to_ext(mime: str) -> str:
    m = (mime or "").lower()
    if "png" in m:
        return ".png"
    if "jpeg" in m or "jpg" in m:
        return ".jpg"
    if "webp" in m:
        return ".webp"
    return ".png"

def _write_error_log(out_dir: Path, text: str) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "error.log").write_text(text, encoding="utf-8")
    except Exception:
        pass

def _to_generated_url(p: Path, out_dir: Path) -> str:
    p = Path(p).resolve()
    gen = GENERATED_DIR.resolve()

    try:
        rel = p.relative_to(gen)
        return f"{STATIC_PREFIX}/" + rel.as_posix()
    except ValueError:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / p.name
        if p != dst:
            dst.write_bytes(p.read_bytes())
        rel = dst.relative_to(gen)
        return f"{STATIC_PREFIX}/" + rel.as_posix()

def _pick_model_url(files_url: Dict[str, str]) -> str:
    for k, u in files_url.items():
        if "glb" in k.lower() or u.lower().endswith(".glb"):
            return u
    for k, u in files_url.items():
        if "pbr" in k.lower():
            return u
    return next(iter(files_url.values()))

async def _download_http_to_file(url: str, dst: Path) -> Path:
    import httpx
    dst.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return dst

def _one_line(s: str) -> str:
    return " ".join((s or "").split())

def _clip(s: str, n: int) -> str:
    s = _one_line(s)
    return s[:n]

def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return (s[:48] or "result")


# -----------------------------
# ✅ Background Removal + Color Hints (폴백 안전)
# -----------------------------
def _remove_bg_and_hints(img_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
    """
    - 흰 배경/단색 배경에 강한 "가벼운" 배경 제거 (PIL+numpy만)
    - 실패하면 원본 그대로 반환
    - 힌트: bg_hex, fg_hexes(top3), is_line_art
    """
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return img_bytes, {}

    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        arr = np.array(im, dtype=np.uint8)
        h, w = arr.shape[:2]
        if h < 8 or w < 8:
            return img_bytes, {}

        # border 샘플로 배경색 추정 (median)
        margin = max(4, min(h, w) // 25)
        top = arr[:margin, :, :3].reshape(-1, 3)
        bot = arr[h-margin:, :, :3].reshape(-1, 3)
        lef = arr[:, :margin, :3].reshape(-1, 3)
        rig = arr[:, w-margin:, :3].reshape(-1, 3)
        border = np.vstack([top, bot, lef, rig])
        bg_rgb = np.median(border, axis=0).astype(np.int32)

        # 거리 임계값(단색 배경용)
        thr = 38  # 너무 낮으면 배경 남고, 너무 높으면 내부 흰색도 날아감
        rgb = arr[:, :, :3].astype(np.int32)
        dist = np.sqrt(np.sum((rgb - bg_rgb[None, None, :]) ** 2, axis=2))

        # flood fill: "테두리에서 연결된 배경만" 제거
        bgmask = np.zeros((h, w), dtype=np.uint8)

        from collections import deque
        q = deque()

        def push_if_bg(y: int, x: int):
            if 0 <= y < h and 0 <= x < w and bgmask[y, x] == 0 and dist[y, x] <= thr:
                bgmask[y, x] = 1
                q.append((y, x))

        # border seed
        for x in range(w):
            push_if_bg(0, x); push_if_bg(h-1, x)
        for y in range(h):
            push_if_bg(y, 0); push_if_bg(y, w-1)

        while q:
            y, x = q.popleft()
            push_if_bg(y-1, x); push_if_bg(y+1, x); push_if_bg(y, x-1); push_if_bg(y, x+1)

        # alpha 적용
        out = arr.copy()
        out[bgmask == 1, 3] = 0

        # fg 팔레트 추출
        fg = out[out[:, :, 3] > 0][:, :3]
        total = h * w
        fg_count = int(fg.shape[0])
        fg_ratio = fg_count / float(total)

        bg_hex = "#{:02x}{:02x}{:02x}".format(int(bg_rgb[0]), int(bg_rgb[1]), int(bg_rgb[2]))

        fg_hexes: List[str] = []
        is_line_art = False

        if fg_count > 50:
            # quantize(거칠게)해서 top-k
            qrgb = (fg // 32) * 32
            keys, counts = np.unique(qrgb, axis=0, return_counts=True)
            order = np.argsort(-counts)[:5]
            top = keys[order]
            for c in top:
                fg_hexes.append("#{0:02x}{1:02x}{2:02x}".format(int(c[0]), int(c[1]), int(c[2])))

            # 라인아트 판정(대충): fg 적고, fg 평균이 어둡다
            mean = fg.mean(axis=0)
            mean_v = float(mean.max())
            if fg_ratio < 0.22 and mean_v < 120:
                is_line_art = True

        # png bytes로 저장
        out_im = Image.fromarray(out, mode="RGBA")
        buf = io.BytesIO()
        out_im.save(buf, format="PNG")
        return buf.getvalue(), {
            "bg_hex": bg_hex,
            "fg_hexes": fg_hexes[:3],
            "fg_ratio": fg_ratio,
            "is_line_art": is_line_art,
        }
    except Exception:
        return img_bytes, {}


# -----------------------------
# ✅ GLB 무광/언릿 후처리 (옵션, 없으면 자동 스킵)
# -----------------------------
def _try_make_glb_matte(src: Path, dst: Path) -> Optional[Path]:
    try:
        from pygltflib import GLTF2
    except Exception:
        return None

    try:
        gltf = GLTF2().load(str(src))

        if gltf.materials:
            if gltf.extensionsUsed is None:
                gltf.extensionsUsed = []
            if "KHR_materials_unlit" not in gltf.extensionsUsed:
                gltf.extensionsUsed.append("KHR_materials_unlit")

            for m in gltf.materials:
                if m is None:
                    continue

                if getattr(m, "pbrMetallicRoughness", None) is not None:
                    pbr = m.pbrMetallicRoughness
                    if pbr is not None:
                        pbr.metallicFactor = 0.0
                        pbr.roughnessFactor = 1.0
                        # ✅ 가능하면 텍스처로 인한 음영/반사 샘플링 줄이기
                        try:
                            pbr.baseColorTexture = None
                            pbr.metallicRoughnessTexture = None
                        except Exception:
                            pass

                m.extensions = (m.extensions or {})
                m.extensions["KHR_materials_unlit"] = {}

                for ext_key in (
                    "KHR_materials_clearcoat",
                    "KHR_materials_specular",
                    "KHR_materials_transmission",
                    "KHR_materials_ior",
                    "KHR_materials_sheen",
                    "KHR_materials_iridescence",
                ):
                    if ext_key in m.extensions:
                        m.extensions.pop(ext_key, None)

                if hasattr(m, "normalTexture"):
                    m.normalTexture = None
                if hasattr(m, "occlusionTexture"):
                    m.occlusionTexture = None
                if hasattr(m, "alphaMode"):
                    m.alphaMode = "OPAQUE"

        dst.parent.mkdir(parents=True, exist_ok=True)
        gltf.save(str(dst))
        if dst.exists() and dst.stat().st_size > 0:
            return dst
        return None
    except Exception:
        return None


# -----------------------------
# OpenAI (prompt + objectName JSON)
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_INSTRUCTIONS = """You are an expert prompt engineer for 3D modeling from a reference image.
Return a COMPACT JSON ONLY, with keys:
- "objectName": a short English object name (1-3 words, lowercase, like "bunny", "rabbit", "cat", "car", "chair")
- "prompt": a single-paragraph English prompt for creating a SIMPLE mesh suitable for LEGO building instructions.

Rules for "prompt":
- Low-poly hard-surface, blocky primitives, clean planar surfaces.
- Keep geometry simple: large primitives, straight edges, shallow insets only.
- Remove micro details: logos, text, tiny vents, patterns, textures, stickers, decals.
- Prefer symmetry when possible.
- Materials must be MATTE and non-reflective: no glossy/shiny surfaces, no specular highlights, no metallic/chrome, no mirror reflections.
- Use FLAT / UNLIT-like simple shading if possible; avoid strong lighting effects baked into the look.
- Avoid inventing extra colors. Use the provided palette guidance if present.
- Output JSON ONLY. No markdown, no extra text.
"""

def _extract_json(resp) -> Dict[str, Any]:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        raw = t.strip()
    else:
        out = getattr(resp, "output", []) or []
        texts: List[str] = []
        for item in out:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", "") in ("output_text", "text"):
                    val = getattr(c, "text", "") or ""
                    if val.strip():
                        texts.append(val.strip())
        raw = "\n".join(texts).strip()

    if not raw:
        return {}

    # 혹시 앞뒤에 잡텍스트 끼면 JSON 부분만 뜯기
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{"): raw.rfind("}") + 1]

    try:
        return json.loads(raw)
    except Exception:
        return {}

def build_lego_prompt_and_name_sync(image_bytes: bytes, mime: str, hints: Dict[str, Any]) -> Tuple[str, str]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # ✅ 힌트 문장(라인아트면 bg색=바디색으로 유도)
    bg_hex = hints.get("bg_hex")
    fg_hexes = hints.get("fg_hexes") or []
    is_line_art = bool(hints.get("is_line_art"))

    hint_text = ""
    if bg_hex or fg_hexes:
        hint_text = f"Palette guidance: backgroundColor={bg_hex}, foregroundColors={fg_hexes}. "
    if is_line_art and bg_hex:
        hint_text += (
            f"This looks like monochrome line art; make the main body color match the backgroundColor ({bg_hex}) "
            f"with minimal small details using the darkest foreground color. "
        )

    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_INSTRUCTIONS,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "The image may be a 2x2 collage showing the same object from four views "
                            "(front, left 3/4, right 3/4, back). Infer one consistent simple blocky model. "
                            "Ignore lighting, shadows, and rendering effects; focus on shape and solid colors. "
                            + hint_text +
                            "Return JSON {objectName, prompt}."
                        ),
                    },
                    {"type": "input_image", "image_url": f"data:{mime};base64,{b64}", "detail": "low"},
                ],
            }
        ],
        max_output_tokens=260,
    )

    data = _extract_json(resp)
    obj = _one_line(str(data.get("objectName", "") or "")).lower()
    prm = _one_line(str(data.get("prompt", "") or ""))

    if not prm:
        raise RuntimeError("Empty prompt from OpenAI")

    if not obj:
        # 최소 폴백: prompt에서 첫 명사 비슷한 단어 뽑기
        m = re.search(r"\b(bunny|rabbit|cat|dog|bear|fox|car|plane|ship|house|robot|duck)\b", prm.lower())
        obj = m.group(1) if m else "object"

    return prm, obj


# -----------------------------
# 1) 2D 콜라주 생성
# -----------------------------
@router.post("/render-image")
async def render_image(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    try:
        img_bytes = await file.read()

        # ✅ 배경 제거 (실패하면 원본)
        cut_bytes, _hints = _remove_bg_and_hints(img_bytes)

        out_bytes = await render_one_image(cut_bytes, "image/png")
        return Response(content=out_bytes, media_type="image/png")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        tb = traceback.format_exc()
        print("[kids.render_image] ERROR\n", tb)
        raise HTTPException(status_code=500, detail="Internal Server Error during generation")


# -----------------------------
# 2) 프롬프트 + 객체명 생성
# -----------------------------
class PromptResp(BaseModel):
    prompt: str
    objectName: str
    hints: Dict[str, Any] = {}

@router.post("/generate-prompt", response_model=PromptResp)
async def generate_prompt(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    try:
        img_bytes = await file.read()

        # ✅ 배경 제거 + 힌트
        cut_bytes, hints = _remove_bg_and_hints(img_bytes)

        prompt, obj = await anyio.to_thread.run_sync(
            build_lego_prompt_and_name_sync, cut_bytes, "image/png", hints
        )
        return {"prompt": prompt, "objectName": obj, "hints": hints}
    except Exception as e:
        tb = traceback.format_exc()
        print("[kids.generate_prompt] ERROR\n", tb)
        if DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error during prompt generation")


# -----------------------------
# 3) 이미지 -> prompt -> Tripo -> 다운로드 -> URL
# -----------------------------
class Generate3DResp(BaseModel):
    prompt: str
    objectName: str
    taskId: str
    modelUrl: str
    files: Dict[str, str]
    hints: Dict[str, Any] = {}

@router.post("/generate-3d", response_model=Generate3DResp)
async def generate_3d(
    file: UploadFile = File(...),
    referenceLdr: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    objectName: str | None = Form(default=None),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="image only")

    TRIPO_API_KEY = os.environ.get("TRIPO_API_KEY", "")
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="TRIPO_API_KEY is not set")

    req_id = uuid.uuid4().hex
    ext = _mime_to_ext(file.content_type or "image/png")

    raw_path = GENERATED_DIR / f"{req_id}_raw{ext}"
    out_dir = GENERATED_DIR / f"tripo_{req_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_bytes = await file.read()
        raw_path.write_bytes(img_bytes)

        # ✅ 배경 제거 + 힌트
        cut_bytes, hints = _remove_bg_and_hints(img_bytes)

        # (1) prompt/objectName: 프론트가 준 게 있으면 우선 사용
        if prompt and prompt.strip() and objectName and objectName.strip():
            prompt = _one_line(prompt.strip())
            obj = _one_line(objectName.strip()).lower()
        elif prompt and prompt.strip():
            prompt = _one_line(prompt.strip())
            obj = _slugify(prompt.split(" ")[0]) or "object"
        else:
            prompt, obj = await anyio.to_thread.run_sync(
                build_lego_prompt_and_name_sync, cut_bytes, "image/png", hints
            )

        prompt = _clip(prompt, 320)
        obj = _slugify(obj)

        # ✅ meta 저장 (brickify에서 파일명 자동으로 따라가게)
        try:
            (out_dir / "meta.json").write_text(
                json.dumps({"objectName": obj, "prompt": prompt, "hints": hints}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        # (2) Tripo 3D
        negative_prompt = _clip(
            "low quality, blurry, noisy, over-detailed, high poly, thin parts, tiny features, "
            "text, logos, patterns, textures, glossy, shiny, reflections, metallic, chrome, mirror, glass, transparent, "
            "strong ambient occlusion, heavy shadows, gradient lighting, multicolor random",
            220,
        )

        async with TripoClient(api_key=TRIPO_API_KEY) as client:
            try:
                task_id = await client.text_to_model(prompt=prompt, negative_prompt=negative_prompt)
            except Exception as e:
                msg = str(e)
                if "code\":1004" in msg or "[1004]" in msg or "parameter is invalid" in msg:
                    task_id = await client.text_to_model(prompt=prompt)
                else:
                    raise

            task = await client.wait_for_task(task_id, verbose=DEBUG)
            if task.status != TaskStatus.SUCCESS:
                raise RuntimeError(f"Tripo task failed: status={task.status}")
            downloaded = await client.download_task_models(task, str(out_dir))

        fixed_downloaded: Dict[str, str] = {}
        for model_type, path_or_url in (downloaded or {}).items():
            if not path_or_url:
                continue
            s = str(path_or_url)
            if s.startswith(("http://", "https://")):
                ext_guess = ".glb" if ".glb" in s.lower() else ".bin"
                dst = out_dir / f"{model_type}{ext_guess}"
                await _download_http_to_file(s, dst)
                fixed_downloaded[model_type] = str(dst)
            else:
                fixed_downloaded[model_type] = s

        # 존재 체크
        missing = []
        for k, v in fixed_downloaded.items():
            pv = Path(v)
            if not pv.exists():
                missing.append((k, v, "NOT_EXISTS"))
            elif pv.stat().st_size == 0:
                missing.append((k, v, "ZERO_SIZE"))
        if missing:
            raise RuntimeError(f"Downloaded files missing: {missing}")

        # ✅ GLB 무광/언릿 후처리 (옵션)
        matte_key: Optional[str] = None
        for k, v in list(fixed_downloaded.items()):
            pv = Path(v)
            if pv.suffix.lower() == ".glb" and pv.exists() and pv.stat().st_size > 0:
                matte_path = out_dir / f"{pv.stem}_matte.glb"
                made = _try_make_glb_matte(pv, matte_path)
                if made:
                    mk = f"{k}_matte"
                    fixed_downloaded[mk] = str(made)
                    matte_key = mk
                break

        files_url: Dict[str, str] = {}
        for model_type, path_str in fixed_downloaded.items():
            files_url[model_type] = _to_generated_url(Path(path_str), out_dir=out_dir)

        if not files_url:
            glbs = [p for p in out_dir.rglob("*.glb") if p.is_file()]
            if glbs:
                files_url["glb"] = _to_generated_url(glbs[0], out_dir=out_dir)

        if not files_url:
            raise RuntimeError("No downloadable model files found in out_dir")

        model_url = files_url.get(matte_key) if matte_key else None
        if not model_url:
            model_url = _pick_model_url(files_url)

        return {
            "prompt": prompt,
            "objectName": obj,
            "taskId": str(task_id),
            "modelUrl": model_url,
            "files": files_url,
            "hints": hints,
        }

    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[kids.generate_3d] ERROR req_id={req_id}\n{tb}")
        _write_error_log(out_dir, tb)

        if DEBUG:
            raise HTTPException(status_code=500, detail={"reqId": req_id, "error": str(e)})

        raise HTTPException(status_code=500, detail="Internal Server Error during 3D generation")


# -----------------------------
# 4) GLB -> LDR (brickify)
# -----------------------------
class BrickifyResp(BaseModel):
    ldrUrl: str
    parts: int
    finalTarget: int
    objectName: str = "result"

AGE_TO_BUDGET = {"4-5": 20, "6-7": 60, "8-10": 120}

def _sanitize_glb_url(u: str) -> str:
    u = (u or "").strip()
    if u.startswith("/api/api/"):
        u = u.replace("/api/api/", "/api/", 1)
    return u

def _local_generated_path_from_url(u: str) -> Optional[Path]:
    u = _sanitize_glb_url(u)
    if u.startswith("/api/generated/"):
        rel = u[len("/api/generated/"):]
        return (GENERATED_DIR / rel).resolve()
    if u.startswith("/generated/"):
        rel = u[len("/generated/"):]
        return (GENERATED_DIR / rel).resolve()
    return None

def _read_meta_object_name(glb_path: Path) -> Optional[str]:
    # tripo_xxx 폴더에 meta.json 저장해둔 거 읽기
    try:
        for cand in [glb_path.parent / "meta.json", glb_path.parent.parent / "meta.json"]:
            if cand.exists():
                data = json.loads(cand.read_text(encoding="utf-8"))
                obj = str(data.get("objectName", "")).strip()
                if obj:
                    return obj
    except Exception:
        pass
    return None

def _load_engine_convert():
    import importlib.util

    engine_path = (PROJECT_ROOT / "brick-engine" / "glb_to_ldr_embedded.py").resolve()
    if not engine_path.exists():
        raise RuntimeError(f"engine file missing: {engine_path}")

    spec = importlib.util.spec_from_file_location("glb_to_ldr_embedded", str(engine_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load spec for glb_to_ldr_embedded")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "convert_glb_to_ldr"):
        raise RuntimeError("convert_glb_to_ldr not found in engine module")

    return mod.convert_glb_to_ldr

_CONVERT_FN = None

@router.post("/brickify", response_model=BrickifyResp)
async def brickify(
    glbUrl: str = Form(...),
    age: str | None = Form(default=None),
    budget: int | None = Form(default=None),
    objectName: str | None = Form(default=None),  # ✅ 프론트가 보내도 되고, 안 보내도 됨
):
    global _CONVERT_FN
    glbUrl = _sanitize_glb_url(glbUrl)

    eff_budget = budget if budget is not None else AGE_TO_BUDGET.get((age or "").strip(), 60)

    req_id = uuid.uuid4().hex
    out_dir = GENERATED_DIR / f"brickify_{req_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_path: Optional[Path] = None
    try:
        local = _local_generated_path_from_url(glbUrl)
        if local and local.exists():
            glb_path = local
        else:
            if not glbUrl.startswith(("http://", "https://")):
                raise RuntimeError(f"Cannot resolve glbUrl: {glbUrl}")
            glb_path = out_dir / "input.glb"
            await _download_http_to_file(glbUrl, glb_path)

        if not glb_path.exists() or glb_path.stat().st_size == 0:
            raise RuntimeError(f"GLB missing/empty: {glb_path}")

        if _CONVERT_FN is None:
            _CONVERT_FN = _load_engine_convert()

        if int(eff_budget) <= 25:
            start_target = 24
        elif int(eff_budget) <= 70:
            start_target = 45
        else:
            start_target = 60

        # ✅ objectName 결정: (1) 폼 > (2) meta.json > (3) result
        obj = _slugify(objectName or "")
        if obj == "result":
            meta_obj = _read_meta_object_name(glb_path)
            if meta_obj:
                obj = _slugify(meta_obj)

        out_ldr = out_dir / f"{obj}.ldr"

        result: Dict[str, Any] = await anyio.to_thread.run_sync(
            lambda: _CONVERT_FN(  # type: ignore
                str(glb_path),
                str(out_ldr),
                budget=int(eff_budget),
                target=int(start_target),
                min_target=5,
                shrink=0.85,
                search_iters=6,
                kind="brick",
                plates_per_voxel=3,
                interlock=True,
                max_area=20,
                solid_color=4,
                use_mesh_color=True,
                invert_y=False,
                smart_fix=True,
                step_order="bottomup",
                title=f"{obj} (tgt={start_target})",
                # 아래 인자들은 엔진에서 받아서 무시(호환용)
                span=4,
                max_new_voxels=12000,
                refine_iters=8,
                ensure_connected=True,
                min_embed=2,
                erosion_iters=1,
                fast_search=True,
                extend_catalog=True,
                max_len=8,
            )
        )

    except Exception as e:
        tb = traceback.format_exc()
        _write_error_log(out_dir, tb)
        print("[kids.brickify] ERROR\n", tb)
        if DEBUG:
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail="brickify failed")

    ldr_url = _to_generated_url(out_ldr, out_dir=out_dir)
    return BrickifyResp(
        ldrUrl=ldr_url,
        parts=int(result.get("parts", 0)),
        finalTarget=int(result.get("final_target", 0)),
        objectName=obj,
    )
