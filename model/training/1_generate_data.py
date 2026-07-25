"""Generate dialog jualan berlabel dari matriks skenario (OFFLINE, mis. di Kaggle).

Alur: data/scenario_matrix.json -> prompt (prompts.py) -> LLM -> JSON dialog ->
validasi (data/dialogs/schema.json + aturan bisnis) -> tulis data/dialogs/<id>.json.

Client bersifat OpenAI-compatible (endpoint /chat/completions), jadi bisa diarahkan
ke provider mana pun ATAU server lokal (vLLM/Ollama). Jalankan beberapa kali dengan
model berbeda untuk MULTI-LLM generation (rekomendasi riset: kurangi bias satu model).

Konfigurasi via env:
    JAGO_LLM_BASE_URL   (mis. https://api.provider.com/v1  atau  http://localhost:11434/v1)
    JAGO_LLM_API_KEY
    JAGO_LLM_MODEL      (mis. qwen2.5-72b-instruct)

Contoh:
    python build_scenario_matrix.py --per-bidang 150
    python 1_generate_data.py --dry-run                 # cek prompt, tanpa panggil API
    python 1_generate_data.py --limit 5                 # smoke test 5 dialog
    python 1_generate_data.py                            # generate semua sel yang belum ada
    python 1_generate_data.py --validate-only           # validasi semua file dialog yang ada
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import prompts  # modul lokal

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DIALOGS_DIR = DATA_DIR / "dialogs"
MATRIX = DATA_DIR / "scenario_matrix.json"
SCHEMA = DIALOGS_DIR / "schema.json"

VALID_TEKNIK = {"sapa_rapport", "gali_kebutuhan", "presentasi_manfaat", "atasi_keberatan", "closing", "upsell"}
VALID_KEBERATAN = {"harga", "bandingkan_kompetitor", "ragu_kualitas", "tidak_butuh", "mau_pikir_pikir", "cuma_lihat_lihat"}


# --------------------------------------------------------------------------- #
# Validasi
# --------------------------------------------------------------------------- #

def validate_dialog(d: dict, expect: dict | None = None) -> list[str]:
    """Validasi 1 dialog. Return daftar pesan error (kosong = valid).

    Pakai jsonschema bila terpasang; selalu tambah aturan bisnis (label per-speaker
    & kehadiran keberatan utama) yang tak tercakup schema.
    """
    errors: list[str] = []
    try:
        import jsonschema  # opsional

        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        v = jsonschema.Draft7Validator(schema)
        errors += [f"schema: {e.message}" for e in v.iter_errors(d)]
    except ImportError:
        # Fallback ringan bila jsonschema tak ada.
        for key in ("scenario_id", "bidang", "produk", "persona", "turns"):
            if key not in d:
                errors.append(f"field wajib hilang: {key}")

    for i, t in enumerate(d.get("turns", [])):
        spk = t.get("speaker")
        if spk == "sales":
            if t.get("teknik") not in VALID_TEKNIK:
                errors.append(f"turn {i}: teknik tidak valid ({t.get('teknik')!r})")
            if t.get("kualitas") not in {"baik", "lemah"}:
                errors.append(f"turn {i}: kualitas wajib baik/lemah")
        elif spk == "pelanggan":
            keb = t.get("keberatan")
            if keb is not None and keb not in VALID_KEBERATAN:
                errors.append(f"turn {i}: keberatan tidak valid ({keb!r})")
        else:
            errors.append(f"turn {i}: speaker tidak valid ({spk!r})")

    if expect:
        keberatan_muncul = {t.get("keberatan") for t in d.get("turns", [])}
        if expect["keberatan_global"] not in keberatan_muncul:
            errors.append(
                f"keberatan utama '{expect['keberatan_global']}' tidak muncul di dialog"
            )
    return errors


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible)
# --------------------------------------------------------------------------- #

def call_llm(messages: list[dict], model: str, base_url: str, api_key: str,
             temperature: float = 0.8, retries: int = 5) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "response_format": {"type": "json_object"}}
    ).encode("utf-8")
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        # Beberapa provider (mis. Groq di balik Cloudflare) memblok User-Agent default
        # urllib (error 1010). Samarkan sebagai browser biasa.
        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            # 429 = rate-limit; tunggu makin lama tiap percobaan (5,10,20,40s) lalu ulang.
            if e.code == 429 and attempt < retries - 1:
                jeda = 5 * (2 ** attempt)
                print(f"    [429] kena rate-limit, tunggu {jeda}s lalu coba lagi...", file=sys.stderr)
                time.sleep(jeda)
                continue
            raise
    raise RuntimeError("gagal setelah beberapa kali retry")


def parse_json_lenient(text: str) -> dict:
    """Ambil objek JSON pertama, toleran terhadap fence markdown yang bocor."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        text = text[4:] if text.lstrip().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def cmd_validate_only() -> int:
    files = sorted(DIALOGS_DIR.glob("*.json"))
    files = [f for f in files if f.name != "schema.json"]
    bad = 0
    for f in files:
        errs = validate_dialog(json.loads(f.read_text(encoding="utf-8")))
        status = "OK " if not errs else "ERR"
        print(f"[{status}] {f.name}")
        for e in errs:
            print(f"        - {e}")
            bad += 1
    print(f"\n{len(files)} file, {bad} error.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="Cetak prompt sel pertama, tanpa panggil API.")
    ap.add_argument("--validate-only", action="store_true", help="Validasi file dialog yang sudah ada, lalu keluar.")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah sel yang digenerate (0 = semua).")
    ap.add_argument("--overwrite", action="store_true", help="Timpa dialog yang sudah ada (default: skip = resume).")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--delay", type=float, default=4.0,
                    help="Jeda (detik) antar request supaya tidak kena rate-limit. Default 4.")
    ap.add_argument("--model", default=os.getenv("JAGO_LLM_MODEL", ""))
    args = ap.parse_args()

    if args.validate_only:
        return cmd_validate_only()

    if not MATRIX.exists():
        print(f"ERROR: {MATRIX} belum ada. Jalankan build_scenario_matrix.py dulu.", file=sys.stderr)
        return 2

    tax = prompts.load_taxonomy()
    cells = json.loads(MATRIX.read_text(encoding="utf-8"))["sel"]
    if args.limit:
        cells = cells[: args.limit]

    if args.dry_run:
        msgs = prompts.build_generation_messages(cells[0], tax)
        print("=== SYSTEM ===\n" + msgs[0]["content"])
        print("\n=== USER ===\n" + msgs[1]["content"])
        print(f"\n[dry-run] {len(cells)} sel siap digenerate.")
        return 0

    base_url = os.getenv("JAGO_LLM_BASE_URL", "")
    api_key = os.getenv("JAGO_LLM_API_KEY", "")
    if not base_url or not args.model:
        print("ERROR: set JAGO_LLM_BASE_URL & --model / JAGO_LLM_MODEL.", file=sys.stderr)
        return 2

    DIALOGS_DIR.mkdir(parents=True, exist_ok=True)
    ok = skip = fail = 0
    for cell in cells:
        out = DIALOGS_DIR / f"{cell['scenario_id']}.json"
        if out.exists() and not args.overwrite:
            skip += 1
            continue
        try:
            msgs = prompts.build_generation_messages(cell, tax)
            raw = call_llm(msgs, args.model, base_url, api_key, args.temperature)
            dialog = parse_json_lenient(raw)
            dialog.setdefault("grounding", {})["generator"] = args.model
            errs = validate_dialog(dialog, expect=cell)
            if errs:
                fail += 1
                print(f"[SKIP-INVALID] {cell['scenario_id']}: {errs[0]}", file=sys.stderr)
                continue
            out.write_text(json.dumps(dialog, ensure_ascii=False, indent=2), encoding="utf-8")
            ok += 1
            print(f"[OK] {cell['scenario_id']}")
        except Exception as e:  # noqa: BLE001 — generator batch, lanjut sel berikutnya
            fail += 1
            print(f"[ERROR] {cell['scenario_id']}: {e}", file=sys.stderr)
        time.sleep(args.delay)  # pacing antar request agar ramah rate-limit

    print(f"\nSelesai. ok={ok} skip={skip} gagal={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
