"""MODE=local — inferensi LLM lokal (Qwen2.5 4-bit + adapter LoRA hasil fine-tune).

Parameter bersifat STATIS saat demo: model & adapter dimuat sekali di proses yang
sama, tidak ada training, auto-tuning, maupun feedback loop (batasan MVP rulebook).

Satu model, dua peran, dibedakan lewat prompt di `prompts.py`:
  * mode Pelanggan — balasan role-play, di-sample supaya tidak monoton.
  * mode Pelatih  — penilaian sesi dalam JSON, greedy supaya hasilnya stabil dan
                    bisa direproduksi saat juri mengulang demo yang sama.

Kalau GPU/model/adapter tidak tersedia, JANGAN diam-diam jatuh ke mock: hasil mock
bukan penilaian AI, dan menyajikannya seolah-olah hasil model akan menyesatkan.
Kegagalan dimunculkan sebagai error yang jelas; operator memilih MODE=mock secara sadar.
"""
from __future__ import annotations

import json
import logging
import re
import threading

from . import prompts
from .config import settings
from .schemas import TEKNIK, ChatMessage, EvaluateResponse, Scenario, TeknikScore

log = logging.getLogger(__name__)

_TEKNIK_VALID = set(TEKNIK)

_model = None
_tokenizer = None
# FastAPI sinkron melayani request di threadpool, jadi dua request bisa masuk bersamaan.
# _load_lock: supaya model tidak dimuat dua kali. _gen_lock: satu GPU, satu generate.
_load_lock = threading.Lock()
_gen_lock = threading.Lock()


class LLMError(RuntimeError):
    """Model tidak bisa dimuat atau keluarannya tidak terpakai."""


# --------------------------------------------------------------------------- #
# Pemuatan model (sekali, malas)
# --------------------------------------------------------------------------- #

def _load() -> tuple[object, object]:
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    with _load_lock:
        if _model is not None:  # sudah dimuat thread lain saat kita menunggu
            return _model, _tokenizer
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as e:
            raise LLMError(
                "Dependensi MODE=local belum terpasang. "
                "Jalankan: pip install -r requirements-model.txt (atau pakai JAGOJUAL_MODE=mock)."
            ) from e

        from pathlib import Path

        adapter = Path(settings.adapter_path)
        punya_adapter = (adapter / "adapter_config.json").exists()
        if not punya_adapter:
            log.warning(
                "Adapter LoRA tidak ditemukan di %s — memakai base model tanpa fine-tune. "
                "Mutu penilaian akan jauh di bawah hasil yang dilaporkan.",
                adapter,
            )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = None
        if settings.load_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )

        log.info("Memuat %s (4bit=%s, adapter=%s)", settings.base_model_id, settings.load_4bit, punya_adapter)
        try:
            tok = AutoTokenizer.from_pretrained(str(adapter) if punya_adapter else settings.base_model_id)
            model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_id, quantization_config=quant, torch_dtype=dtype, device_map={"": 0}
            )
            if punya_adapter:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(adapter))
        except Exception as e:  # noqa: BLE001 — apa pun sebabnya, pesannya harus bisa ditindaklanjuti
            raise LLMError(f"Gagal memuat model/adapter: {e}") from e

        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()
        _model, _tokenizer = model, tok
        return _model, _tokenizer


def _generate(messages: list[dict], max_new_tokens: int, sample: bool) -> str:
    import torch

    model, tok = _load()
    teks = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(teks, return_tensors="pt").to(model.device)
    kwargs = {"do_sample": True, "temperature": settings.temperature, "top_p": 0.9} if sample else {"do_sample": False}
    with _gen_lock, torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kwargs)
    return tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- #
# Mode Pelanggan
# --------------------------------------------------------------------------- #

def generate_customer_reply(scenario: Scenario, history: list[ChatMessage], message: str) -> str:
    balasan = _generate(
        prompts.roleplay_messages(scenario, history),
        max_new_tokens=settings.max_new_tokens_chat,
        sample=True,
    )
    # Model kadang ikut menuliskan label pembicara; buang supaya tidak muncul di gelembung chat.
    balasan = re.sub(r"^(Pelanggan|Sales)\s*:\s*", "", balasan).strip()
    if not balasan:
        raise LLMError("Model tidak menghasilkan balasan pelanggan.")
    return balasan


# --------------------------------------------------------------------------- #
# Mode Pelatih
# --------------------------------------------------------------------------- #

def _parse_json(teks: str) -> dict | None:
    teks = re.sub(r"^```(?:json)?|```$", "", teks.strip(), flags=re.MULTILINE).strip()
    awal, akhir = teks.find("{"), teks.rfind("}")
    if awal < 0 or akhir <= awal:
        return None
    try:
        obj = json.loads(teks[awal : akhir + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _int_terbatas(nilai: object) -> int | None:
    try:
        return max(0, min(100, int(round(float(nilai)))))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def normalisasi_penilaian(obj: dict | None) -> EvaluateResponse:
    """Ubah keluaran mentah model jadi EvaluateResponse.

    Yang dibuang: teknik di luar taksonomi, skor non-numerik, saran non-teks. Yang
    TIDAK dilakukan: mengarang skor untuk teknik yang tidak dinilai model — teknik
    yang absen memang tidak ditampilkan, lebih baik daripada angka karangan.
    """
    if not obj:
        raise LLMError("Keluaran mode Pelatih bukan JSON yang bisa dibaca.")

    per_teknik: list[TeknikScore] = []
    terlihat: set[str] = set()
    for item in obj.get("per_teknik") or []:
        if not isinstance(item, dict):
            continue
        teknik = item.get("teknik")
        skor = _int_terbatas(item.get("skor"))
        if teknik not in _TEKNIK_VALID or teknik in terlihat or skor is None:
            continue
        terlihat.add(teknik)
        catatan = item.get("catatan")
        per_teknik.append(TeknikScore(teknik=teknik, skor=skor, catatan=str(catatan) if catatan else ""))

    if not per_teknik:
        raise LLMError("Mode Pelatih tidak mengembalikan satu pun penilaian teknik yang valid.")

    # Urutkan sesuai taksonomi supaya tampilan konsisten antar sesi.
    per_teknik.sort(key=lambda t: TEKNIK.index(t.teknik))

    skor_total = _int_terbatas(obj.get("skor_total"))
    if skor_total is None:
        skor_total = round(sum(t.skor for t in per_teknik) / len(per_teknik))

    saran = [str(s).strip() for s in (obj.get("saran") or []) if isinstance(s, (str, int, float)) and str(s).strip()]
    return EvaluateResponse(skor_total=skor_total, per_teknik=per_teknik, saran=saran[:3])


def evaluate_conversation(scenario: Scenario, history: list[ChatMessage]) -> EvaluateResponse:
    mentah = _generate(
        prompts.coach_messages(scenario, history),
        max_new_tokens=settings.max_new_tokens_evaluate,
        sample=False,
    )
    hasil = _parse_json(mentah)
    if hasil is None:
        log.warning("Keluaran mode Pelatih gagal di-parse: %.300s", mentah)
    return normalisasi_penilaian(hasil)
